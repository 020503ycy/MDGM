##导入包##
import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import MolDraw2DCairo
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from torch_geometric.nn import SAGEConv,global_mean_pool
from torch.nn.init import kaiming_uniform_
import random
from cairosvg import svg2png
from scipy.ndimage import gaussian_filter1d
import matplotlib.cm as cm
import matplotlib.colors as colors
from PIL import Image
from matplotlib.colors import TwoSlopeNorm
import tempfile
import io

##定义模型与功能函数##
# 定义超参数
N = 450
H_0 = 90
H_1 = 90
H_2 = 90
H_3 = 90
train_frac = 0.8
batch_size = 32
shuffle_seed = 0
learning_rate = 0.003
epochs = 1000
# 早停相关参数
patience = 20  # 原代码为10，改为20
best_val_loss = float('inf')
counter = 0


# 定义 Standardizer 类
class Standardizer:
    def __init__(self, X):
        self.mean = torch.mean(X)
        self.std = torch.std(X)

    def standardize(self, X):
        Z = (X - self.mean) / (self.std)
        return Z

    def restore(self, Z):
        X = self.mean + Z * self.std
        return X

    def state(self):
        return {"mean": self.mean, "std": self.std}

    def load(self, state):
        self.mean = state["mean"]
        self.std = state["std"]


# 定义 smiles_to_data 函数
def smiles_to_data(smiles, N=450):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES provided")

    # Adjacency Matrix
    adj = rdmolops.GetAdjacencyMatrix(mol, useBO=True)
    adj_matrix = np.zeros((N, N))
    s0, s1 = adj.shape
    if s0 > N:
        raise ValueError("Molecule too large")
    adj_matrix[:s0, :s1] = adj

    # Feature Matrix (One-hot encoding)
    atoms = mol.GetAtoms()
    n_atoms = 450
    node_mat = np.zeros((n_atoms, 90))
    for atom in atoms:
        atom_index = atom.GetIdx()
        atom_no = atom.GetAtomicNum()
        node_mat[atom_index, atom_no] = 1

    # Labels
    labels = []  # Assuming labels are not available in this function

    # Convert to PyTorch tensors
    X = torch.from_numpy(node_mat).float()
    A = torch.from_numpy(adj_matrix).float()
    y = torch.Tensor([[]]).float()  # Replace with actual labels if available
    mol_num = torch.Tensor([0])  # Assuming no molecular number available
    from scipy.sparse import coo_matrix
    A_coo = coo_matrix(A)
    edge_index = torch.from_numpy(np.vstack([A_coo.row, A_coo.col])).long()
    edge_weight = torch.from_numpy(A_coo.data).float()
    num_atoms = mol.GetNumAtoms()  # 获取实际原子数

    return Data(
        x=X,
        edge_index=edge_index,
        edge_attr=edge_weight,
        y=y,
        A=A,
        mol_num=mol_num,
        num_atoms=num_atoms,  # 新增属性
    )


# 定义 load_data 函数
def load_data(N=450):
    df = pd.read_csv('GraphSAGE_weight/train.csv')
    dataset = []
    labels = []
    for index, row in df.iterrows():
        mol = Chem.MolFromSmiles(row.smiles)
        if mol is None:
            continue
        adj = rdmolops.GetAdjacencyMatrix(mol, useBO=True)
        s0, s1 = adj.shape
        if s0 > N:
            continue
        adj_matrix = np.zeros((N, N))
        adj_matrix[:s0, :s1] = adj

        atoms = mol.GetAtoms()
        n_atoms = 450
        node_mat = np.zeros((n_atoms, 90))
        for atom in atoms:
            atom_index = atom.GetIdx()
            atom_no = atom.GetAtomicNum()
            node_mat[atom_index, atom_no] = 1

        X = torch.from_numpy(node_mat).float()
        A = torch.from_numpy(adj_matrix).float()
        y = torch.tensor([row.energy]).float()
        mol_num = torch.tensor([row.num]).float()
        from scipy.sparse import coo_matrix
        A_coo = coo_matrix(A)
        edge_index = torch.from_numpy(np.vstack([A_coo.row, A_coo.col])).long()
        edge_weight = torch.from_numpy(A_coo.data).float()
        num_atoms = mol.GetNumAtoms()  # 获取实际原子数

        dataset.append(Data(
            x=X,
            edge_index=edge_index,
            edge_attr=edge_weight,
            y=y,
            A=A,
            mol_num=mol_num,
            num_atoms=num_atoms  # 新增属性
        ))
        labels.append(row.energy)
    return dataset


# 定义GraphSAGE模型类
class GraphSAGE(torch.nn.Module):
    def __init__(self, H_0, H_1, H_2, H_3):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(H_0, H_1)
        self.conv2 = SAGEConv(H_1, H_2)
        self.conv3 = SAGEConv(H_2, H_3)
        self.fc1 = torch.nn.Linear(H_3, 32)
        self.fc2 = torch.nn.Linear(32, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, SAGEConv):
                if hasattr(m, 'lin') and m.lin is not None:
                    kaiming_uniform_(m.lin.weight, nonlinearity='relu')
                    if m.lin.bias is not None:
                        torch.nn.init.zeros_(m.lin.bias)
            elif isinstance(m, torch.nn.Linear):
                kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, data):
        h0, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        h1 = F.relu(self.conv1(h0, edge_index))
        h2 = F.relu(self.conv2(h1, edge_index))
        h3 = F.relu(self.conv3(h2, edge_index))

        # 保存激活值（与GIN类似逻辑）
        self.activations = h3

        # 注册钩子以捕获梯度（与GIN类似逻辑）
        def save_gradients(grad):
            self.gradients = grad

        if h3.requires_grad:
            h3.register_hook(save_gradients)

        h4 = global_mean_pool(h3, data.batch)
        h4 = F.relu(self.fc1(h4))
        out = self.fc2(h4)
        return out


def train(model, loader, device, optimizer, standardizer):
    model.train()
    total_loss = 0
    total_mae = 0
    all_y_true = []
    all_y_pred = []
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)

        y_std = standardizer.standardize(data.y.view(-1, 1))
        loss = F.mse_loss(out, y_std)

        mae = F.l1_loss(standardizer.restore(out), data.y.view(-1, 1))

        loss.backward()
        # 统计梯度方向
        total_positive = 0
        total_negative = 0
        for param in model.parameters():
            if param.grad is not None:
                total_positive += (param.grad > 0).sum().item()
                total_negative += (param.grad < 0).sum().item()

        total_loss += loss.item() * data.num_graphs
        total_mae += mae.item() * data.num_graphs

        optimizer.step()

        all_y_true.append(data.y.view(-1, 1).cpu())
        all_y_pred.append(standardizer.restore(out).cpu())

    average_mae = total_mae / len(loader.dataset)
    average_loss = total_loss / len(loader.dataset)
    print(f'Train MAE: {average_mae:.4f}')
    return average_loss, average_mae, torch.cat(all_y_true, dim=0), torch.cat(all_y_pred, dim=0)


def test(model, loader, device, standardizer):
    model.eval()
    total_loss = 0
    total_mae = 0
    all_y_true = []
    all_y_pred = []

    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            output = model(data)
            y_std = standardizer.standardize(data.y.view(-1, 1))

            total_loss += F.mse_loss(output, y_std, reduction='sum').item()
            total_mae += F.l1_loss(standardizer.restore(output),
                                   data.y.view(-1, 1), reduction='sum').item()

            all_y_true.append(data.y.view(-1, 1).cpu())
            all_y_pred.append(standardizer.restore(output).cpu())

    all_y_true = torch.cat(all_y_true, dim=0)
    all_y_pred = torch.cat(all_y_pred, dim=0)

    average_loss = total_loss / len(loader.dataset)
    average_mae = total_mae / len(loader.dataset)
    rmse = torch.sqrt(torch.tensor(average_loss))  # 计算RMSE

    y_mean = all_y_true.mean()
    total_sum_of_squares = ((all_y_true - y_mean) ** 2).sum().item()
    residual_sum_of_squares = ((all_y_true - all_y_pred) ** 2).sum().item()
    r2_score = 1 - (residual_sum_of_squares / total_sum_of_squares)

    all_y_true_np = all_y_true.numpy().ravel()
    all_y_pred_np = all_y_pred.numpy().ravel()
    pearson_corr, _ = pearsonr(all_y_true_np, all_y_pred_np)

    print(f'Test MAE: {average_mae:.4f}')
    print(f'R² Score: {r2_score:.4f}')
    print(f'Pearson Correlation Coefficient: {pearson_corr:.4f}')
    print(f'RMSE: {rmse.item():.4f}')  # 打印RMSE

    return average_loss, average_mae, r2_score, pearson_corr, all_y_pred, all_y_true, rmse


# 修改后的预测函数
def predict(model, smiles, device, standardizer):
    data = smiles_to_data(smiles)
    # 自动生成batch属性
    loader = DataLoader([data], batch_size=1, shuffle=False)
    model.eval()
    with torch.no_grad():
        for data_in_loader in loader:
            data_in_loader = data_in_loader.to(device)
            output = model(data_in_loader)
    return standardizer.restore(output).item()


def compute_global_cam_stats(model, dataset, device):
    model.eval()
    all_cams = []
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    for data in loader:
        data = data.to(device)
        model.zero_grad()
        output = model(data)
        output.backward()

        acts = model.activations.detach().cpu().numpy()
        grads = model.gradients.detach().cpu().numpy()
        num_atoms = data.num_atoms

        cam = grad_cam(acts, grads, num_atoms)
        all_cams.append(cam)

    all_cams = np.concatenate(all_cams)
    # 只考虑正向贡献的范围
    global_min = np.percentile(all_cams, 1)
    global_max = np.percentile(all_cams, 99)
    return global_min, global_max  # 不再取对称范围

def grad_cam(activations, gradients, num_atoms):
    if activations.ndim != 2 or gradients.ndim != 2:
        raise ValueError(f"输入应为二维数组，但获得：activations {activations.shape}，gradients {gradients.shape}")
    
    # 保持原始梯度计算（不再取符号）
    weights = np.mean(gradients, axis=0)
    cam = np.dot(activations, weights)
    cam = cam[:num_atoms] if len(cam) >= num_atoms else cam
    return gaussian_filter1d(cam, sigma=1.5)

def plot_atom_contributions(model, data, smiles, num_value, output_folder, global_min, global_max):
    mol = Chem.MolFromSmiles(smiles)
    num_atoms = data.num_atoms

    acts = model.activations.detach().cpu().numpy()
    grads = model.gradients.detach().cpu().numpy()

    atom_weights = grad_cam(acts, grads, num_atoms)
    
    # 线性映射到0-1范围
    atom_weights = (atom_weights - global_min) / (global_max - global_min)
    atom_weights = np.clip(atom_weights, 0, 1)  # 限制在[0,1]

    assert len(atom_weights) == num_atoms, f"权重数量{len(atom_weights)}与原子数{num_atoms}不匹配"

    img = visualize_atom_weights(mol, atom_weights)
    img_path = os.path.join(output_folder, f'{num_value}_atom_contributions.png')
    plt.imsave(img_path, img)

def visualize_atom_weights(mol, weights):
    assert len(weights) == mol.GetNumAtoms(), "权重数量与分子原子数不匹配"

    # 使用从蓝到红的双色渐变
    norm = TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1)
    cmap = plt.get_cmap('bwr')

    drawer = MolDraw2DCairo(400, 400)
    rdDepictor.Compute2DCoords(mol)

    atom_colors = {}
    for i in range(len(weights)):
        color = cmap(norm(weights[i]))
        atom_colors[i] = (color[0], color[1], color[2], 0.7)  # 70%透明度

    # 渲染分子图像
    drawer.DrawMolecule(mol,
                        highlightAtoms=list(range(len(weights))),
                        highlightAtomColors=atom_colors)
    drawer.FinishDrawing()
    img_data = drawer.GetDrawingText()
    img = plt.imread(io.BytesIO(img_data))

    # 生成颜色条
    fig, ax = plt.subplots(figsize=(1, 4))
    im = ax.imshow(np.array([[0, 0.5, 1]]), cmap=cmap, norm=norm, aspect='auto')
    ax.set_visible(False)
    cbar = fig.colorbar(im, orientation='vertical', label='Promotion Effect', shrink=1)
    cbar.ax.set_ylabel('Promotion Strength', rotation=270, labelpad=15)

    # 处理颜色条图像
    colorbar_buffer = io.BytesIO()
    plt.savefig(colorbar_buffer, format='png', bbox_inches='tight')
    colorbar_buffer.seek(0)
    colorbar_img = plt.imread(colorbar_buffer)
    plt.close(fig)

    # 确保颜色条图像高度与分子图像一致
    if colorbar_img.shape[0] != img.shape[0]:
        colorbar_pil = Image.fromarray((colorbar_img * 255).astype(np.uint8))
        colorbar_pil = colorbar_pil.resize((colorbar_pil.width, img.shape[0]))
        colorbar_img = np.array(colorbar_pil) / 255

    # 将颜色条图像从 RGBA 转换为 RGB 格式
    if colorbar_img.shape[2] == 4:
        colorbar_pil = Image.fromarray((colorbar_img * 255).astype(np.uint8))
        colorbar_pil = colorbar_pil.convert('RGB')
        colorbar_img = np.array(colorbar_pil) / 255

    # 合并图像和颜色条
    combined_img = np.hstack((img, colorbar_img))
    return combined_img

## 训练模型部分（数据划分与加载修改）
# 加载数据集
dataset = load_data(N)
random.Random(shuffle_seed).shuffle(dataset)

# 划分训练集、验证集、测试集（7:2:1）
train_size = int(0.7 * len(dataset))
val_size = int(0.2 * len(dataset))  # 从10%改为20%
test_size = len(dataset) - train_size - val_size

train_dataset = dataset[:train_size]
val_dataset = dataset[train_size:train_size + val_size]
test_dataset = dataset[train_size + val_size:]

# 准备数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  # 新增验证集加载器
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 准备标准化器（基于训练集）
output = [data.y for data in train_dataset]
standardizer = Standardizer(torch.Tensor(output))

# 初始化模型与优化器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GraphSAGE(H_0, H_1, H_2, H_3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)

# 记录训练和验证损失
train_losses = []
val_losses = []  # 改为记录验证集损失
best_train_y_true = None
best_train_y_pred = None
best_val_y_true = None
best_val_y_pred = None
best_test_y_true = None
best_test_y_pred = None

## 训练模型（早停基于验证集，增加多集合指标输出）
for epoch in range(1, epochs + 1):
    print(f'Epoch {epoch}')
    
    # 训练集评估
    train_loss, train_mae, train_y_true, train_y_pred = train(model, train_loader, device, optimizer, standardizer)
    
    # 验证集评估
    val_loss, val_mae, val_r2, val_pearson, val_y_pred, val_y_true, _ = test(
        model, val_loader, device, standardizer
    )
    
    # 测试集评估（训练过程中同步输出，非最终测试）
    test_loss, test_mae, test_r2, test_pearson, test_y_pred, test_y_true, _ = test(
        model, test_loader, device, standardizer
    )
    
    # 统一输出格式
    print(f'Train Loss: {train_loss:.4f} | MAE: {train_mae:.4f}')
    print(f'Val Loss: {val_loss:.4f} | MAE: {val_mae:.4f} | R²: {val_r2:.4f}')
    print(f'Test Loss: {test_loss:.4f} | MAE: {test_mae:.4f} | Pearson: {test_pearson:.4f}\n')

    train_losses.append(train_loss)
    val_losses.append(val_loss)  # 记录验证损失

    # 早停判断（基于验证集损失）
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        state = {
            'model_state_dict': model.state_dict(),  # 模型参数
            'standardizer_state': standardizer.state(),  # 标准化器参数（包含mean/std）
        }
        torch.save(state, 'GraphSAGE_weight/best_graphsage_model.pth')
        # 保存最优模型对应的真实值和预测值
        best_train_y_true = train_y_true
        best_train_y_pred = train_y_pred
        best_val_y_true = val_y_true
        best_val_y_pred = val_y_pred
        best_test_y_true = test_y_true
        best_test_y_pred = test_y_pred
    else:
        counter += 1
        if counter >= patience:
            print(f'Early stopping at epoch {epoch}')
            break

# 绘图部分
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import linregress, t

# 期刊级配色方案（Seaborn风格优化）
COLORS = {
    'train': '#81021f',  
    'val': '#435f87',    
    'test': '#fdd95f'     
}

# 创建画布及主坐标轴
fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
fig.subplots_adjust(left=0.15, right=0.9, bottom=0.15, top=0.9)

# 初始化图例元素
legend_elements = []

def plot_dataset(y_true, y_pred, dataset_name, color):
    """绘制数据集散点、拟合线及置信区间，并生成图例句柄"""
    # 数据预处理（确保一维输入）
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()

    # 创建图例句柄（带白色描边的圆形标记）
    handle = plt.Line2D(
        [], [],
        marker='o',
        color='none',
        markerfacecolor=color,
        markeredgecolor='white',
        markeredgewidth=0.8,
        markersize=8,
    )
    legend_elements.append(handle)

    # 绘制散点图
    ax.scatter(
        y_true, y_pred,
        c=color,
        alpha=0.8,
        s=40,
        edgecolor='white',
        linewidth=0.8,
        zorder=2  # 确保散点在最上层
    )

    # 线性回归拟合
    slope, intercept, r_value, p_value, std_err = linregress(y_true, y_pred)[:5]
    x_min = min(y_true.min(), y_pred.min())
    x_max = max(y_true.max(), y_pred.max())
    x_fit = np.linspace(x_min, x_max, 100)  # 根据数据实际范围生成x_fit
    y_fit = slope * x_fit + intercept

    # 绘制拟合线
    ax.plot(
        x_fit, y_fit,
        color=color,
        linestyle='--',
        linewidth=1.2,
        alpha=0.7,
        zorder=1  # 拟合线在散点下方
    )

    # 计算并填充置信区间
    residuals = y_pred - y_true
    se = np.std(residuals, ddof=2)
    n = len(y_true)
    t_val = t.ppf(0.975, n - 2)
    ci_low = y_fit - t_val * se * np.sqrt(1 / n + (x_fit - y_true.mean()) ** 2 / ((n - 1) * y_true.var()))
    ci_high = y_fit + t_val * se * np.sqrt(1 / n + (x_fit - y_true.mean()) ** 2 / ((n - 1) * y_true.var()))
    ax.fill_between(
        x_fit, ci_low, ci_high,
        color=color,
        alpha=0.15,
        zorder=0  # 置信区间在最底层
    )

# 绘制各数据集（统一管理数据集参数）
DATASETS = [
    (best_train_y_true, best_train_y_pred, 'Training Set', COLORS['train']),
    (best_val_y_true, best_val_y_pred, 'Validation Set', COLORS['val']),
    (best_test_y_true, best_test_y_pred, 'Test Set', COLORS['test'])
]

# 计算性能指标
def compute_metrics(yt, yp):
    yt = yt.ravel()
    yp = yp.ravel()
    mae = mean_absolute_error(yt, yp)
    rmse = np.sqrt(mean_squared_error(yt, yp))
    return f'MAE={mae:.2f} | RMSE={rmse:.2f}'

legend_labels = []
for yt, yp, name, color in DATASETS:
    # 修改部分：添加 detach() 方法
    plot_dataset(yt.detach().cpu().numpy(), yp.detach().cpu().numpy(), name, color)
    metrics = compute_metrics(yt.detach().cpu().numpy(), yp.detach().cpu().numpy())
    legend_labels.append(f'{name}: {metrics}')

# 获取所有数据的范围
all_y_true = np.concatenate([yt.detach().cpu().numpy().ravel() for yt, _, _, _ in DATASETS])
all_y_pred = np.concatenate([yp.detach().cpu().numpy().ravel() for _, yp, _, _ in DATASETS])
xmin = min(all_y_true.min(), all_y_pred.min())
xmax = max(all_y_true.max(), all_y_pred.max())

# 设置坐标轴范围
ax.set_xlim(xmin, xmax)
ax.set_ylim(xmin, xmax)

# 添加 1:1 对角线（关键参考线），改为虚线
lim = [xmin, xmax]
ax.plot(
    lim, lim,
    linestyle='--',  # 明确设置为虚线
    color='k',
    alpha=0.8,
    linewidth=1.5,
    zorder=3  # 对角线在最上层
)

# 创建合并后的左上角图例（包含所有数据集标识和指标文本）
ax.legend(
    handles=legend_elements,
    labels=legend_labels,
    loc='upper left',
    bbox_to_anchor=(0.02, 0.98),
    frameon=True,
    framealpha=0.9,
    edgecolor='white',
    fontsize=10,
    title_fontsize=11
)


# 图表细节优化（期刊级格式）
ax.set(
    xlabel='True Values (kcal/mol)',
    ylabel='Predicted Values (kcal/mol)',
    aspect='equal',  # 强制等比例坐标轴
)

# 设置标题
ax.grid(linestyle=':', linewidth=0.5, alpha=0.7, zorder=-1)  # 网格在底层
ax.spines[['top', 'right']].set_visible(False)  # 隐藏右上边框

# 保存图表为 PDF 文件
plt.savefig('model_performance_plot.pdf', format='pdf')

plt.show()

# 只计算一次全局参数
global_min, global_max = compute_global_cam_stats(model, train_dataset, device)
print(f"Global CAM - Min: {global_min}, Max: {global_max}")
os.makedirs('GraphSAGE_weight', exist_ok=True)

##train
##绘制热度图##
# 创建保存原子贡献图的文件夹
output_folder = 'atom_contribution_images_train'
os.makedirs(output_folder, exist_ok=True)

# 加载预测数据，修正分隔符错误
predict_df = pd.read_csv('GraphSAGE_weight/train.csv')
# 按照 num 列进行排序
predict_df = predict_df.sort_values(by='num')

predictions = []

# 初始化模型和标准化器（从训练保存的文件加载）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GraphSAGE(H_0, H_1, H_2, H_3).to(device)
standardizer = Standardizer(torch.tensor([0.0]))  # 临时初始化用于加载状态

# 加载完整的 state 字典（包含 'model_state_dict' 和 'standardizer_state' 键）
state = torch.load(
    'GraphSAGE_weight/best_graphsage_model.pth',
    map_location=device
)

# 1. 加载模型状态：通过 'model_state_dict' 键获取模型参数
model.load_state_dict(state['model_state_dict'])

# 2. 加载标准化器状态：通过 'standardizer_state' 键获取均值/标准差
standardizer.load(state['standardizer_state'])

model.eval()  # 评估模式

# 预测并生成贡献图
for index, row in predict_df.iterrows():
    smiles = row['smiles']
    num_value = row['num']  # 获取 num 列的值
    data = smiles_to_data(smiles)
    loader = DataLoader([data], batch_size=1, shuffle=False)
    for data_in_loader in loader:
        data_in_loader = data_in_loader.to(device)
        model.zero_grad()  # 增加梯度清零
        output = model(data_in_loader)
        output.backward()  # 触发梯度计算
        plot_atom_contributions(model, data_in_loader, smiles, num_value, output_folder, global_min, global_max)
    pred = predict(model, smiles, device, standardizer)
    predictions.append(pred)

# 保存预测结果
predict_df['predicted_energy'] = predictions
predict_df.to_csv('MDGM_GraphSAGE.csv', index=False)
model.eval()

##BDBS
## 训练完成后，加载最佳模型并执行纯预测 ##
# 加载预测数据
predict_df = pd.read_csv('GraphSAGE_weight/test.csv')
predictions = []

# 初始化模型和标准化器（从训练保存的文件加载）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GraphSAGE(H_0, H_1, H_2, H_3).to(device)
standardizer = Standardizer(torch.tensor([0.0]))  # 临时初始化用于加载状态

state = torch.load(
    'GraphSAGE_weight/best_graphsage_model.pth',
    map_location=device
)
model.load_state_dict(state['model_state_dict'])
standardizer.load(state['standardizer_state'])
model.eval()  # 评估模式

# 纯预测逻辑（无梯度、无可视化）
for smiles in predict_df['smiles']:
    data = smiles_to_data(smiles)
    loader = DataLoader([data], batch_size=1, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch)
            pred = standardizer.restore(output).item()
            predictions.append(pred)

# 保存结果
predict_df['predicted_energy'] = predictions
predict_df.to_csv('BDBS_GraphSAGE.csv', index=False)

##BBBP
## 训练完成后，加载最佳模型并执行纯预测 ##
# 加载预测数据
# 加载预测数据
try:
    predict_df = pd.read_csv('GraphSAGE_weight/BBBP.csv', encoding='gbk')
except UnicodeDecodeError:
    try:
        predict_df = pd.read_csv('GraphSAGE_weight/BBBP.csv', encoding='latin1')
    except Exception as e:
        print(f"仍然无法读取文件: {e}")
predictions = []

# 初始化模型和标准化器（从训练保存的文件加载）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GraphSAGE(H_0, H_1, H_2, H_3).to(device)
standardizer = Standardizer(torch.tensor([0.0]))  # 临时初始化用于加载状态

state = torch.load(
    'GraphSAGE_weight/best_graphsage_model.pth',
    map_location=device
)
model.load_state_dict(state['model_state_dict'])
standardizer.load(state['standardizer_state'])
model.eval()  # 评估模式

# 纯预测逻辑（无梯度、无可视化）
for smiles in predict_df['smiles']:
    try:
        data = smiles_to_data(smiles)
        loader = DataLoader([data], batch_size=1, shuffle=False)
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                output = model(batch)
                pred = standardizer.restore(output).item()
                predictions.append(pred)
    except (ValueError, RuntimeError) as e:  # 扩大异常处理范围
        print(f"遇到不合理的数据，已跳过：{smiles}，错误信息：{e}")
        predictions.append(np.nan)  # 填充缺失值
# 保存结果
predict_df['predicted_energy'] = predictions
predict_df.to_csv('BBBP_GraphSAGE.csv', index=False)

##B3DB
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
import numpy as np

predict_df = pd.read_csv('GraphSAGE_weight/B3DB.csv')
predictions = []

# 初始化模型和标准化器（从训练保存的文件加载）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GraphSAGE(H_0, H_1, H_2, H_3).to(device)
standardizer = Standardizer(torch.tensor([0.0]))  # 临时初始化用于加载状态

state = torch.load(
    'GraphSAGE_weight/best_graphsage_model.pth',
    map_location=device,
    weights_only=True  # 处理未来警告
)
model.load_state_dict(state['model_state_dict'])
standardizer.load(state['standardizer_state'])
model.eval()  # 评估模式

# 纯预测逻辑（无梯度、无可视化）
for smiles in predict_df['smiles']:
    try:
        data = smiles_to_data(smiles)
        loader = DataLoader([data], batch_size=1, shuffle=False)
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                output = model(batch)
                pred = standardizer.restore(output).item()
                predictions.append(pred)
    except (ValueError, RuntimeError) as e:  # 扩大异常处理范围
        print(f"遇到不合理的数据，已跳过：{smiles}，错误信息：{e}")
        predictions.append(np.nan)  # 填充缺失值

# 保存结果
predict_df['predicted_energy'] = predictions
predict_df.to_csv('B3DB_GraphSAGE.csv', index=False)
    
