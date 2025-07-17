###完美
import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdmolops
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from torch_geometric.nn import GINConv, global_mean_pool, GCNConv, SAGEConv, GATConv
import random
from torch.nn.init import kaiming_uniform_  # 导入He初始化函数

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
patience = 10  # 当验证集损失在连续10个epoch没有下降时停止训练

# 定义 Standardizer 类
class Standardizer:
    def __init__(self, X):
        self.mean = torch.mean(X)
        self.std = torch.std(X)

    def standardize(self, X):
        Z = (X - self.mean) / self.std
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
        y=torch.tensor([0.0]),  # 这里暂时用 0.0 占位，后续可替换
        A=A,
        mol_num=torch.tensor([0]),
        num_atoms=num_atoms
    )

# 定义 load_data 函数
def load_data(N=450):
    df = pd.read_csv('train.csv')
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
        y = torch.tensor([row.energy]).float()
        mol_num = torch.tensor([row.num]).float()
        from scipy.sparse import coo_matrix
        A_coo = coo_matrix(A)
        edge_index = torch.from_numpy(np.vstack([A_coo.row, A_coo.col])).long()
        edge_weight = torch.from_numpy(A_coo.data).float()
        num_atoms = mol.GetNumAtoms()

        dataset.append(Data(
            x=X,
            edge_index=edge_index,
            edge_attr=edge_weight,
            y=y,
            A=A,
            mol_num=mol_num,
            num_atoms=num_atoms
        ))
    return dataset

# 基于图同构网络（Graph Isomorphism Network, GIN）的神经网络模型
class GIN(torch.nn.Module):
    def __init__(self, H_0, H_1, H_2, H_3):
        super(GIN, self).__init__()
        self.conv1 = GINConv(torch.nn.Linear(H_0, H_1))
        self.conv2 = GINConv(torch.nn.Linear(H_1, H_2))
        self.conv3 = GINConv(torch.nn.Linear(H_2, H_3))
        self.fc1 = torch.nn.Linear(H_3, 32)
        self.fc2 = torch.nn.Linear(32, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, GINConv):
                # 访问GINConv内部的nn属性
                kaiming_uniform_(m.nn.weight, nonlinearity='relu')
                if m.nn.bias is not None:
                    torch.nn.init.zeros_(m.nn.bias)
            elif isinstance(m, torch.nn.Linear):
                kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, data):
        h0, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        h1 = F.elu(self.conv1(h0, edge_index))
        h2 = F.elu(self.conv2(h1, edge_index))
        h3 = F.elu(self.conv3(h2, edge_index))
        h4 = global_mean_pool(h3, data.batch)
        h4 = F.elu(self.fc1(h4))
        out = self.fc2(h4)
        return out

# 基于图卷积网络（Graph Convolutional Network, GCN）的神经网络模型
class GCN(torch.nn.Module):
    def __init__(self, H_0, H_1, H_2, H_3):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(H_0, H_1)
        self.conv2 = GCNConv(H_1, H_2)
        self.conv3 = GCNConv(H_2, H_3)
        self.fc1 = torch.nn.Linear(H_3, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, 1)
        self._init_weights()  # 调用初始化函数

    def _init_weights(self):
        """对GCN模型的参数进行He初始化"""
        for m in self.modules():
            if isinstance(m, GCNConv):
                # 访问GCNConv内部的线性层参数
                kaiming_uniform_(m.lin.weight, nonlinearity='relu')
                if m.lin.bias is not None:
                    torch.nn.init.zeros_(m.lin.bias)
            elif isinstance(m, torch.nn.Linear):
                kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, data):
        h0, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        h1 = F.leaky_relu(self.conv1(h0, edge_index, edge_weight))
        h2 = F.leaky_relu(self.conv2(h1, edge_index, edge_weight))
        h3 = F.leaky_relu(self.conv3(h2, edge_index, edge_weight))
        h4 = global_mean_pool(h3, data.batch)
        h4 = F.leaky_relu(self.fc1(h4))
        h4 = F.leaky_relu(self.fc2(h4))
        out = self.fc3(h4)
        return out

## 修改后的GraphSAGE初始化部分
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
                # 检查是否存在lin层并初始化
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
        h4 = global_mean_pool(h3, data.batch)
        h4 = F.relu(self.fc1(h4))
        out = self.fc2(h4)
        return out

# 基于图注意力网络（Graph Attention Network, GAT）的神经网络模型
class GAT(torch.nn.Module):
    def __init__(self, H_0, H_1, H_2, H_3):
        super(GAT, self).__init__()
        self.conv1 = GATConv(H_0, H_1)
        self.conv2 = GATConv(H_1, H_2)
        self.conv3 = GATConv(H_2, H_3)
        self.fc1 = torch.nn.Linear(H_3, 32)
        self.fc2 = torch.nn.Linear(32, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, GATConv):
                # 初始化GATConv的线性层（如果存在）
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
        h1 = F.elu(self.conv1(h0, edge_index))
        h2 = F.elu(self.conv2(h1, edge_index))
        h3 = F.elu(self.conv3(h2, edge_index))
        h4 = global_mean_pool(h3, data.batch)
        h4 = F.elu(self.fc1(h4))
        out = self.fc2(h4)
        return out

# 修改后的train函数
def train(model, loader, device, optimizer, standardizer):
    model.train()
    total_loss = 0  # 记录原始尺度的MSE
    total_mae = 0
    min_loss = float('inf')
    max_loss = float('-inf')

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        
        # 标准化后的损失用于反向传播
        y_std = standardizer.standardize(data.y.view(-1, 1))
        loss = F.mse_loss(out, y_std)
        
        # 计算恢复后的输出用于记录原始尺度损失
        out_restored = standardizer.restore(out)
        mae = F.l1_loss(out_restored, data.y.view(-1, 1))
        mse_restored = F.mse_loss(out_restored, data.y.view(-1, 1))
        
        loss.backward()
        
        # 累加原始尺度的MSE和MAE
        total_loss += mse_restored.item() * data.num_graphs
        total_mae += mae.item() * data.num_graphs

        # 更新最小和最大损失值
        if mse_restored.item() < min_loss:
            min_loss = mse_restored.item()
        if mse_restored.item() > max_loss:
            max_loss = mse_restored.item()
        
        optimizer.step()
    
    average_mae = total_mae / len(loader.dataset)
    average_loss = total_loss / len(loader.dataset)
    
    # 标准化平均损失值
    if max_loss != min_loss:
        normalized_loss = (average_loss - min_loss) / (max_loss - min_loss)
    else:
        normalized_loss = 0.0  # 避免除以零
    
    print(f'Train MAE: {average_mae:.4f}')
    print(f'Normalized Train Loss: {normalized_loss:.4f}')
    
    return average_loss, min_loss, max_loss  # 返回原始损失、最小和最大值

# 修改后的test函数
def test(model, loader, device, standardizer):
    model.eval()
    total_loss = 0
    total_mae = 0
    total_mse = 0
    all_y_true = []
    all_y_pred = []
    min_loss = float('inf')
    max_loss = float('-inf')

    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            output = model(data)
            y_pred_restore = standardizer.restore(output)

            # 计算 MAE 和 MSE（恢复后的值）
            mae = F.l1_loss(y_pred_restore, data.y.view(-1, 1))
            mse = F.mse_loss(y_pred_restore, data.y.view(-1, 1))

            total_mae += mae.item() * data.num_graphs
            total_mse += mse.item() * data.num_graphs
            total_loss += mse.item() * data.num_graphs

            # 更新最小和最大损失值
            if mse.item() < min_loss:
                min_loss = mse.item()
            if mse.item() > max_loss:
                max_loss = mse.item()

            all_y_true.append(data.y.view(-1, 1).cpu())
            all_y_pred.append(y_pred_restore.cpu())

    average_loss = total_loss / len(loader.dataset)
    average_mae = total_mae / len(loader.dataset)
    average_mse = total_mse / len(loader.dataset)

    all_y_true = torch.cat(all_y_true, dim=0)
    all_y_pred = torch.cat(all_y_pred, dim=0)

    y_mean = all_y_true.mean()
    total_sum_of_squares = ((all_y_true - y_mean) ** 2).sum().item()
    residual_sum_of_squares = ((all_y_true - all_y_pred) ** 2).sum().item()
    r2_score = 1 - (residual_sum_of_squares / total_sum_of_squares)

    all_y_true_np = all_y_true.numpy().ravel()
    all_y_pred_np = all_y_pred.numpy().ravel()
    pearson_corr, _ = pearsonr(all_y_true_np, all_y_pred_np)

    # 标准化平均损失值
    if max_loss != min_loss:
        normalized_loss = (average_loss - min_loss) / (max_loss - min_loss)
    else:
        normalized_loss = 0.0  # 避免除以零

    print(f'Test MAE: {average_mae:.4f}')
    print(f'R² Score: {r2_score:.4f}')
    print(f'Pearson Correlation Coefficient: {pearson_corr:.4f}')
    print(f'Normalized Test Loss: {normalized_loss:.4f}')

    return average_loss, average_mae, average_mse, r2_score, pearson_corr, all_y_pred, all_y_true, min_loss, max_loss

# 修改后的train_and_test函数
def train_and_test(model, model_name, train_loader, test_loader, device, standardizer):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)
    train_losses = []  # 存储 (average_loss, min_loss, max_loss)
    test_losses = []   # 存储 (average_loss, min_loss, max_loss)
    best_val_loss = float('inf')
    counter = 0
    epochs_run = 0

    for epoch in range(1, epochs + 1):
        epochs_run = epoch
        print(f'Epoch {epoch} for {model_name}')
        train_loss, train_min, train_max = train(model, train_loader, device, optimizer, standardizer)
        test_loss, test_mae, test_mse, test_r2, test_pearson, _, _, test_min, test_max = test(model, test_loader, device, standardizer)

        train_losses.append( (train_loss, train_min, train_max) )
        test_losses.append( (test_loss, test_min, test_max) )

        if test_loss < best_val_loss:
            best_val_loss = test_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping at epoch {epoch} for {model_name}')
                break

    return test_mae, test_mse, test_r2, epochs_run, train_losses, test_losses

# 加载数据集
dataset = load_data(N)
random.Random(shuffle_seed).shuffle(dataset)

# 五折交叉验证
k = 5
fold_size = len(dataset) // k
results = {
    'GIN': {'MAE': [], 'MSE': [], 'R^2': [], 'epochs_run': []},
    'GCN': {'MAE': [], 'MSE': [], 'R^2': [], 'epochs_run': []},
    'GraphSAGE': {'MAE': [], 'MSE': [], 'R^2': [], 'epochs_run': []},
    'GAT': {'MAE': [], 'MSE': [], 'R^2': [], 'epochs_run': []}
}
model_losses = {
    'GIN': {'train_loss': [], 'test_loss': []},
    'GCN': {'train_loss': [], 'test_loss': []},
    'GraphSAGE': {'train_loss': [], 'test_loss': []},
    'GAT': {'train_loss': [], 'test_loss': []}
}

for fold in range(k):
    print(f"Fold {fold + 1}")
    test_indices = list(range(fold * fold_size, (fold + 1) * fold_size))
    train_indices = [i for i in range(len(dataset)) if i not in test_indices]

    train_dataset = [dataset[i] for i in train_indices]
    test_dataset = [dataset[i] for i in test_indices]

    # 准备数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 准备标准化器
    output = torch.cat([data.y for data in train_dataset], dim=0)
    standardizer = Standardizer(output)

    # 设备选择
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化模型
    models = {
        'GIN': GIN(H_0, H_1, H_2, H_3).to(device),
        'GCN': GCN(H_0, H_1, H_2, H_3).to(device),
        'GraphSAGE': GraphSAGE(H_0, H_1, H_2, H_3).to(device),
        'GAT': GAT(H_0, H_1, H_2, H_3).to(device)
    }

    for model_name, model in models.items():
        print(f"Training {model_name} in fold {fold + 1}")
        mae, mse, r2, epochs_run, train_losses, test_losses = train_and_test(
            model, model_name, train_loader, test_loader, device, standardizer
        )
        results[model_name]['MAE'].append(mae)
        results[model_name]['MSE'].append(mse)
        results[model_name]['R^2'].append(r2)
        results[model_name]['epochs_run'].append(epochs_run)
        model_losses[model_name]['train_loss'].append(train_losses)
        model_losses[model_name]['test_loss'].append(test_losses)

# 输出各模型的评估指标（五折交叉验证的平均值）
print("各模型评估指标（五折交叉验证平均值）：")
print("模型\t\t平均MAE\t\t平均MSE\t\t平均R²\t\t平均训练轮数")
for model_name, metrics in results.items():
    avg_mae = np.mean(metrics['MAE'])
    avg_mse = np.mean(metrics['MSE'])
    avg_r2 = np.mean(metrics['R^2'])
    avg_epochs = np.mean(metrics['epochs_run'])
    print(f"{model_name}\t\t{avg_mae:.4f}\t\t{avg_mse:.4f}\t\t{avg_r2:.4f}\t\t{avg_epochs:.0f}")

# 找出各项指标表现最佳的模型（基于五折交叉验证平均值）
best_mae_model = min(results, key=lambda x: np.mean(results[x]['MAE']))
best_mse_model = min(results, key=lambda x: np.mean(results[x]['MSE']))
best_r2_model = max(results, key=lambda x: np.mean(results[x]['R^2']))

print("\n各项指标表现最佳的模型（基于五折交叉验证平均值）：")
print(f"MAE 最小的模型: {best_mae_model} (平均 MAE: {np.mean(results[best_mae_model]['MAE']):.4f})")
print(f"MSE 最小的模型: {best_mse_model} (平均 MSE: {np.mean(results[best_mse_model]['MSE']):.4f})")
print(f"R² 最大的模型: {best_r2_model} (平均 R²: {np.mean(results[best_r2_model]['R^2']):.4f})")

# 提取模型名称
model_names = list(results.keys())

# 提取各指标平均值
mae_values = [np.mean(results[model]['MAE']) for model in model_names]
mse_values = [np.mean(results[model]['MSE']) for model in model_names]
r2_values = [np.mean(results[model]['R^2']) for model in model_names]

x = np.arange(len(model_names))
width = 0.25

fig, ax = plt.subplots()
rects1 = ax.bar(x - width, mse_values, width, label='MSE')
rects2 = ax.bar(x, mae_values, width, label='MAE')
rects3 = ax.bar(x + width, r2_values, width, label='R²')

ax.set_ylabel('Scores')
ax.set_title('Model Performance Comparison (5-fold CV)')
ax.set_xticks(x)
ax.set_xticklabels(model_names)
ax.legend()

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.4f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

fig.tight_layout()

# 保存为 SVG 图片
plt.savefig('model_performance_2.svg', format='svg')
plt.show()

# 修改后的绘图部分，处理每个fold的每个epoch的数据
for model_name, data in model_losses.items():
    all_train_losses = data['train_loss']  # 每个元素是一个fold的每个epoch的损失列表
    all_test_losses = data['test_loss']
    
    plt.figure(figsize=(10, 6))
    
    # 遍历每个fold的训练和测试损失数据
    for fold_idx, (fold_train_losses, fold_test_losses) in enumerate(zip(all_train_losses, all_test_losses)):
        # 提取每个epoch的average_loss并标准化
        train_loss_values = [loss for loss, _, _ in fold_train_losses]
        test_loss_values = [loss for loss, _, _ in fold_test_losses]
        
        # 计算该fold的标准化训练损失
        if len(train_loss_values) > 0:
            train_min = min(train_loss_values)
            train_max = max(train_loss_values)
            normalized_train = [(loss - train_min) / (train_max - train_min) if (train_max - train_min) != 0 else 0.0 
                                for loss in train_loss_values]
            plt.plot(normalized_train, label=f'Fold {fold_idx+1} Train', alpha=0.5)
        
        # 计算该fold的标准化测试损失
        if len(test_loss_values) > 0:
            test_min = min(test_loss_values)
            test_max = max(test_loss_values)
            normalized_test = [(loss - test_min) / (test_max - test_min) if (test_max - test_min) != 0 else 0.0 
                               for loss in test_loss_values]
            plt.plot(normalized_test, label=f'Fold {fold_idx+1} Test', alpha=0.5)
    
    plt.xlabel('Epoch')
    plt.ylabel('Normalized Loss')
    plt.title(f'{model_name} Normalized Loss per Fold')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{model_name}_normalized_loss_per_fold.svg', format='svg')
    plt.close()