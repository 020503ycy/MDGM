import os
import torch
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdmolops, rdDepictor
from rdkit.Chem.Draw import MolDraw2DCairo
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from torch.nn.init import kaiming_uniform_
from scipy.ndimage import gaussian_filter1d
import matplotlib.cm as cm
import matplotlib.colors as colors
from PIL import Image
from matplotlib.colors import TwoSlopeNorm
import io


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


# 定义 GraphSAGE 模型类
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
        h1 = torch.relu(self.conv1(h0, edge_index))
        h2 = torch.relu(self.conv2(h1, edge_index))
        h3 = torch.relu(self.conv3(h2, edge_index))

        # 保存激活值（与GIN类似逻辑）
        self.activations = h3

        # 注册钩子以捕获梯度（与GIN类似逻辑）
        def save_gradients(grad):
            self.gradients = grad

        if h3.requires_grad:
            h3.register_hook(save_gradients)

        h4 = global_mean_pool(h3, data.batch)
        h4 = torch.relu(self.fc1(h4))
        out = self.fc2(h4)
        return out


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

    # Convert to PyTorch tensors
    X = torch.from_numpy(node_mat).float()
    A = torch.from_numpy(adj_matrix).float()
    from scipy.sparse import coo_matrix
    A_coo = coo_matrix(A)
    edge_index = torch.from_numpy(np.vstack([A_coo.row, A_coo.col])).long()
    edge_weight = torch.from_numpy(A_coo.data).float()
    num_atoms = mol.GetNumAtoms()  # 获取实际原子数

    return Data(
        x=X,
        edge_index=edge_index,
        edge_attr=edge_weight,
        num_atoms=num_atoms,
    )


# 定义 load_data 函数
def load_data(N=450):
    df = pd.read_csv('GraphSAGE_weight/train.csv')
    dataset = []
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
        from scipy.sparse import coo_matrix
        A_coo = coo_matrix(A)
        edge_index = torch.from_numpy(np.vstack([A_coo.row, A_coo.col])).long()
        edge_weight = torch.from_numpy(A_coo.data).float()
        num_atoms = mol.GetNumAtoms()  # 获取实际原子数

        dataset.append(Data(
            x=X,
            edge_index=edge_index,
            edge_attr=edge_weight,
            num_atoms=num_atoms
        ))
    return dataset


# 定义 grad_cam 函数
def grad_cam(activations, gradients, num_atoms):
    if activations.ndim != 2 or gradients.ndim != 2:
        raise ValueError(f"输入应为二维数组，但获得：activations {activations.shape}，gradients {gradients.shape}")

    # 保持原始梯度计算（不再取符号）
    weights = np.mean(gradients, axis=0)
    cam = np.dot(activations, weights)
    cam = cam[:num_atoms] if len(cam) >= num_atoms else cam
    return gaussian_filter1d(cam, sigma=1.5)


# 定义 compute_global_cam_stats 函数
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


# 定义 visualize_atom_weights 函数
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


# 定义 plot_atom_contributions 函数
def plot_atom_contributions(model, data, smiles, num_value, output_folder, global_min, global_max):
    mol = Chem.MolFromSmiles(smiles)
    num_atoms = data.num_atoms

    acts = model.activations.detach().cpu().numpy()
    grads = model.gradients.detach().cpu().numpy()

    atom_weights = grad_cam(acts, grads, num_atoms)

    # 线性映射到0 - 1范围
    atom_weights = (atom_weights - global_min) / (global_max - global_min)
    atom_weights = np.clip(atom_weights, 0, 1)  # 限制在[0,1]

    assert len(atom_weights) == num_atoms, f"权重数量{len(atom_weights)}与原子数{num_atoms}不匹配"

    img = visualize_atom_weights(mol, atom_weights)

    # 创建一个新的图像对象
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img)
    ax.axis('off')

    # 保存为 PDF 格式，设置 DPI 为 600
    img_path = os.path.join(output_folder, f'{num_value}_atom_contributions.pdf')
    plt.savefig(img_path, format='pdf', dpi=600, bbox_inches='tight')
    plt.close(fig)


## 加载模型和标准化器配置
# 超参数保持与训练时一致
N = 450
H_0 = 90
H_1 = 90
H_2 = 90
H_3 = 90

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 初始化模型和标准化器
model = GraphSAGE(H_0, H_1, H_2, H_3).to(device)
standardizer = Standardizer(torch.tensor([0.0]))  # 临时初始化用于加载状态

# 加载训练好的权重
state = torch.load(
    'GraphSAGE_weight/best_graphsage_model.pth',
    map_location=device
)
model.load_state_dict(state['model_state_dict'])
standardizer.load(state['standardizer_state'])
model.eval()  # 评估模式

## 计算全局CAM标准化参数（用于热图颜色映射）
# 加载训练数据集计算全局范围
train_dataset_for_cam = load_data(N)
global_min, global_max = compute_global_cam_stats(model, train_dataset_for_cam, device)

## 读取smiles文件
# 假设文件格式：第一列为num，第二列为smiles，表头为'num'和'smiles'
input_file = 'smiles.txt'
df = pd.read_csv(input_file, delimiter='\t')  # 假设用制表符分隔，根据实际情况调整

## 预测并生成热图
predictions = []
output_folder = 'atom_contribution_images_pred'
os.makedirs(output_folder, exist_ok=True)

for idx, row in df.iterrows():
    num_value = row['num']
    smiles = row['smiles']

    try:
        # 生成图数据
        data = smiles_to_data(smiles)
        loader = DataLoader([data], batch_size=1, shuffle=False)

        for batch in loader:
            batch = batch.to(device)
            model.zero_grad()  # 梯度清零

            # 前向传播并计算梯度
            output = model(batch)
            output.backward()  # 触发梯度计算，用于热图生成

            # 保存预测结果
            pred = standardizer.restore(output).item()
            predictions.append(pred)

            # 生成原子贡献热图
            plot_atom_contributions(
                model,
                batch,
                smiles,
                num_value,
                output_folder,
                global_min,
                global_max
            )

    except Exception as e:
        print(f"处理SMILES {smiles} 时出错: {str(e)}")
        predictions.append(None)
        continue

## 保存预测结果
df['predicted_energy'] = predictions
df.to_csv('predicted_results.csv', index=False)

print("预测及热图生成完成！结果保存在predicted_results.csv和atom_contribution_images_pred文件夹中")