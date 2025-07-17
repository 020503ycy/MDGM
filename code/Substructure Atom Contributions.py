##统计
import os
import torch
import numpy as np
import pandas as pd
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool
from torch.nn.init import kaiming_uniform_
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d

# 定义超参数
N = 450
H_0 = 90
H_1 = 90
H_2 = 90
H_3 = 90

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
    
    # 新增load方法用于加载保存的标准化器状态
    def load(self, state):
        self.mean = state["mean"]
        self.std = state["std"]

# 定义 smiles_to_data 函数
def smiles_to_data(smiles, N=450):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES provided")
    adj = Chem.rdmolops.GetAdjacencyMatrix(mol, useBO=True)
    adj_matrix = np.zeros((N, N))
    s0, s1 = adj.shape
    if s0 > N:
        raise ValueError("Molecule too large")
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
    num_atoms = mol.GetNumAtoms()
    return Data(
        x=X,
        edge_index=edge_index,
        edge_attr=edge_weight,
        A=A,
        num_atoms=num_atoms,
        smiles=smiles
    )

# 定义 load_data 函数
def load_data(N=450):
    df = pd.read_csv('GraphSAGE_weight/train.csv')
    dataset = []
    smiles_list = []
    for index, row in df.iterrows():
        try:
            mol = Chem.MolFromSmiles(row.smiles)
            if mol is None:
                continue
            adj = Chem.rdmolops.GetAdjacencyMatrix(mol, useBO=True)
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
            y = torch.tensor([float(row.energy)]).float()
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
                num_atoms=num_atoms,
                smiles=row.smiles
            ))
            smiles_list.append(row.smiles)
        except ValueError:
            print(f"无法处理样本 {row.smiles}，跳过。")
    return dataset, smiles_list

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

        # 保存激活值（与SAGEConv类似逻辑）
        self.activations = h3

        # 注册钩子以捕获梯度（与SAGEConv类似逻辑）
        def save_gradients(grad):
            self.gradients = grad

        if h3.requires_grad:
            h3.register_hook(save_gradients)

        h4 = global_mean_pool(h3, data.batch)
        h4 = F.relu(self.fc1(h4))
        out = self.fc2(h4)
        return out

def grad_cam(activations, gradients, num_atoms):
    weights = np.mean(gradients, axis=0)
    cam = np.dot(activations, weights)
    cam = cam[:num_atoms] if len(cam) >= num_atoms else cam
    return gaussian_filter1d(cam, sigma=1.5)

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
    global_min = np.percentile(all_cams, 1)
    global_max = np.percentile(all_cams, 99)
    return global_min, global_max

# 新增函数：统计特定子结构的平均贡献值
def calculate_substructure_contribution(model, data, device, substructure_smiles, global_min, global_max):
    print(f"正在处理子结构 SMILES: {substructure_smiles}")
    substructure = Chem.MolFromSmiles(substructure_smiles)
    if substructure is None:
        print(f"无效的子结构 SMILES: {substructure_smiles}")
        return 0  # 这里返回 0 表示跳过该无效子结构，也可以根据需求修改处理方式
    data = data.to(device)
    model.zero_grad()
    output = model(data)
    output.backward()

    acts = model.activations.detach().cpu().numpy()
    grads = model.gradients.detach().cpu().numpy()
    num_atoms = data.num_atoms

    atom_weights = grad_cam(acts, grads, num_atoms)

    # 线性映射到0-1范围
    atom_weights = (atom_weights - global_min) / (global_max - global_min)
    atom_weights = np.clip(atom_weights, 0, 1)  # 限制在[0,1]

    smiles = data.smiles
    mol = Chem.MolFromSmiles(smiles)
    matches = mol.GetSubstructMatches(substructure)
    all_contributions = []
    if matches:
        for match in matches:
            match_contribution = np.mean([atom_weights[i] for i in match])
            all_contributions.append(match_contribution)

    if all_contributions:
        average_contribution = np.mean(all_contributions)
        return average_contribution
    else:
        return 0

# 统计多个子结构并保存结果到 CSV
def calculate_and_save_multiple_substructures(model, smiles_list, device, substructure_list, global_min, global_max, output_csv):
    results = []
    for smiles in smiles_list:
        data = smiles_to_data(smiles)
        for substructure_smiles in substructure_list:
            average_contribution = calculate_substructure_contribution(model, data, device, substructure_smiles, global_min, global_max)
            results.append({
                'smiles': smiles,
                'substructure': substructure_smiles,
                'average_contribution': average_contribution
            })
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_csv, index=False)

# 加载数据集和 SMILES 列表
dataset, smiles_list = load_data(N)

# 准备标准化器
output = [data.y for data in dataset]
standardizer = Standardizer(torch.Tensor(output))

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

# 计算全局参数
global_min, global_max = compute_global_cam_stats(model, dataset, device)
print(f"Global CAM - Min: {global_min}, Max: {global_max}")

# 定义要分析的子结构列表
substructure_list = [
    # 烃类官能团
    "C", "CC", "C=C", "C#C",
    # 含氧化合物官能团
    "CO", "CCO", "c1ccccc1O", "COC", "C=O", "CC=O", "CC(=O)C", "C(=O)O", "CC(=O)O", "COC(=O)C",
    # 含氮化合物官能团
    "CN", "CCN", "CC(=O)N", "CC#N",
    # 含卤素化合物官能团
    "CCl", "CCBr",
    # 其他官能团
    "c1ccccc1S(=O)(=O)O", "c1ccccc1[N+](=O)[O-]",
    # 含苯环结构
    "c1ccccc1", "Cc1ccccc1", "C=CC1=CC=CC=C1", "Nc1ccccc1",
    # 含萘环结构
    "c1ccc2ccccc2c1", "Oc1ccc2ccccc2c1", "Nc1ccc2cccc(c2)c1",
    # 含杂环结构
    "n1ccccc1", "o1cccc1", "s1cccc1", "n1cccc1"
]
output_csv = 'GraphSAGE_weight/substructure_contributions.csv'
calculate_and_save_multiple_substructures(model, smiles_list, device, substructure_list, global_min, global_max, output_csv)
print(f"子结构贡献值已保存到 {output_csv}")    

import pandas as pd

# 读取结果文件
df = pd.read_csv('GraphSAGE_weight/substructure_contributions.csv')

# 去掉第三列（索引为 2）值为 0 的行
df = df[df.iloc[:, 2] != 0]

# 统计每个子结构的出现次数和平均贡献
substructure_stats = df.groupby('substructure').agg(
    出现次数=('smiles', 'nunique'),
    平均贡献=('average_contribution', 'mean')
).reset_index()

# 保存统计结果到新的 CSV 文件
substructure_stats.to_csv('GraphSAGE_weight/substructure_statistics.csv', index=False)

print("统计结果已保存到 substructure_statistics.csv")
    
#绘图
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows 系统可以使用 SimHei
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 读取结果文件
df = pd.read_csv('substructure_contributions.csv')

# 去掉第三列（索引为 2）值为 0 的行
df = df[df.iloc[:, 2] != 0]

# 筛选出特定的烃类子结构
hydrocarbon_substructures = ['C', 'CC', 'C=C', 'C#C']
df = df[df['substructure'].isin(hydrocarbon_substructures)]

# 统计每个子结构的出现次数和四分位信息
substructure_stats = df.groupby('substructure').agg(
    出现次数=('smiles', 'nunique'),
    平均贡献=('average_contribution', 'mean'),
    下四分位=('average_contribution', lambda x: x.quantile(0.25)),  # 25% 分位数
    上四分位=('average_contribution', lambda x: x.quantile(0.75))   # 75% 分位数
).reset_index()

# 按平均贡献对数据进行排序
substructure_stats = substructure_stats.sort_values(by='平均贡献')

# 定义颜色映射，改为双色映射 'RdBu'
norm = plt.Normalize(vmin=0, vmax=1138)
cmap = plt.get_cmap('magma')

# 创建颜色列表
colors = [cmap(norm(count)) for count in substructure_stats['出现次数']]

# 固定柱子宽度
bar_width = 0.6
# 根据柱子数量动态调整图片宽度
num_bars = len(substructure_stats)
plt.figure(figsize=(num_bars * 1.5, 6))  # 这里 1.5 是一个调整因子，可以根据需要修改

ax = plt.gca()  # 获取当前 Axes

# 计算误差范围并取绝对值
lower_error = np.abs(substructure_stats['平均贡献'] - substructure_stats['下四分位'])
upper_error = np.abs(substructure_stats['上四分位'] - substructure_stats['平均贡献'])
yerr = [lower_error, upper_error]

# 使用索引作为 x 位置绘制柱状图
bars = ax.bar(
    np.arange(num_bars),
    substructure_stats['平均贡献'],
    width=bar_width,
    color=colors,
    yerr=yerr,
    capsize=5,  # 误差条顶部的横线长度
    alpha=0.7
)

# 设置 x 轴标签
ax.set_xticks(np.arange(num_bars))
ax.set_xticklabels(substructure_stats['substructure'], rotation=45)

plt.ylabel('平均贡献')
plt.title('烷烃类')

# 设置 y 轴范围为 0-1
ax.set_ylim(0, 1)

# 添加颜色条
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)  # 指定颜色条的 Axes
cbar.set_label('出现次数')

plt.tight_layout()

# 保存图形为 SVG 格式
plt.savefig('GraphSAGE_weight/烷烃类.pdf', format='pdf')

plt.show()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows 系统可以使用 SimHei
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 读取结果文件
df = pd.read_csv('GraphSAGE_weight/substructure_contributions.csv')

# 去掉第三列（索引为 2）值为 0 的行
df = df[df.iloc[:, 2] != 0]

# 筛选出特定的烃类子结构
hydrocarbon_substructures = ['CO', 'CCO', 'c1ccccc1O', 'COC', 'C=O', 'CC=O', 'CC(=O)C', 'C(=O)O','CC(=O)O','CC(=O)N','o1cccc1']
df = df[df['substructure'].isin(hydrocarbon_substructures)]

# 统计每个子结构的出现次数和四分位信息
substructure_stats = df.groupby('substructure').agg(
    出现次数=('smiles', 'nunique'),
    平均贡献=('average_contribution', 'mean'),
    下四分位=('average_contribution', lambda x: x.quantile(0.25)),  # 25% 分位数
    上四分位=('average_contribution', lambda x: x.quantile(0.75))   # 75% 分位数
).reset_index()

# 按平均贡献对数据进行排序
substructure_stats = substructure_stats.sort_values(by='平均贡献')

# 定义颜色映射，改为双色映射 'RdBu'
norm = plt.Normalize(vmin=0, vmax=1138)
cmap = plt.get_cmap('magma')

# 创建颜色列表
colors = [cmap(norm(count)) for count in substructure_stats['出现次数']]

# 固定柱子宽度
bar_width = 0.6
# 根据柱子数量动态调整图片宽度
num_bars = len(substructure_stats)
plt.figure(figsize=(num_bars * 1.5, 6))  # 这里 1.5 是一个调整因子，可以根据需要修改

ax = plt.gca()  # 获取当前 Axes

# 计算误差范围并取绝对值
lower_error = np.abs(substructure_stats['平均贡献'] - substructure_stats['下四分位'])
upper_error = np.abs(substructure_stats['上四分位'] - substructure_stats['平均贡献'])
yerr = [lower_error, upper_error]

# 使用索引作为 x 位置绘制柱状图
bars = ax.bar(
    np.arange(num_bars),
    substructure_stats['平均贡献'],
    width=bar_width,
    color=colors,
    yerr=yerr,
    capsize=5,  # 误差条顶部的横线长度
    alpha=0.7
)

# 设置 x 轴标签
ax.set_xticks(np.arange(num_bars))
ax.set_xticklabels(substructure_stats['substructure'], rotation=45)

plt.ylabel('平均贡献')
plt.title('含氧官能团')

# 设置 y 轴范围为 0-1
ax.set_ylim(0, 1)

# 添加颜色条
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)  # 指定颜色条的 Axes
cbar.set_label('出现次数')

plt.tight_layout()

# 保存图形为 SVG 格式
plt.savefig('GraphSAGE_weight/含氧官能团.pdf', format='pdf')

plt.show()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows 系统可以使用 SimHei
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 读取结果文件
df = pd.read_csv('GraphSAGE_weight/substructure_contributions.csv')

# 去掉第三列（索引为 2）值为 0 的行
df = df[df.iloc[:, 2] != 0]

# 筛选出特定的烃类子结构
hydrocarbon_substructures = ['c1ccccc1', 'Cc1ccccc1', 'C=CC1=CC=CC=C1', 'Nc1ccccc1','c1ccc2ccccc2c1','c1ccccc1O']
df = df[df['substructure'].isin(hydrocarbon_substructures)]

# 统计每个子结构的出现次数和四分位信息
substructure_stats = df.groupby('substructure').agg(
    出现次数=('smiles', 'nunique'),
    平均贡献=('average_contribution', 'mean'),
    下四分位=('average_contribution', lambda x: x.quantile(0.25)),  # 25% 分位数
    上四分位=('average_contribution', lambda x: x.quantile(0.75))   # 75% 分位数
).reset_index()

# 按平均贡献对数据进行排序
substructure_stats = substructure_stats.sort_values(by='平均贡献')

# 定义颜色映射，改为双色映射 'RdBu'
norm = plt.Normalize(vmin=0, vmax=1138)
cmap = plt.get_cmap('magma')

# 创建颜色列表
colors = [cmap(norm(count)) for count in substructure_stats['出现次数']]

# 固定柱子宽度
bar_width = 0.6
# 根据柱子数量动态调整图片宽度
num_bars = len(substructure_stats)
plt.figure(figsize=(num_bars * 1.5, 6))  # 这里 1.5 是一个调整因子，可以根据需要修改

ax = plt.gca()  # 获取当前 Axes

# 计算误差范围并取绝对值
lower_error = np.abs(substructure_stats['平均贡献'] - substructure_stats['下四分位'])
upper_error = np.abs(substructure_stats['上四分位'] - substructure_stats['平均贡献'])
yerr = [lower_error, upper_error]

# 使用索引作为 x 位置绘制柱状图
bars = ax.bar(
    np.arange(num_bars),
    substructure_stats['平均贡献'],
    width=bar_width,
    color=colors,
    yerr=yerr,
    capsize=5,  # 误差条顶部的横线长度
    alpha=0.7
)

# 设置 x 轴标签
ax.set_xticks(np.arange(num_bars))
ax.set_xticklabels(substructure_stats['substructure'], rotation=45)

plt.ylabel('平均贡献')
plt.title('含苯环')

# 设置 y 轴范围为 0-1
ax.set_ylim(0, 1)

# 添加颜色条
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)  # 指定颜色条的 Axes
cbar.set_label('出现次数')

plt.tight_layout()

# 保存图形为 SVG 格式
plt.savefig('GraphSAGE_weight/含苯环.pdf', format='pdf')

plt.show()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows 系统可以使用 SimHei
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 读取结果文件
df = pd.read_csv('GraphSAGE_weight/substructure_contributions.csv')

# 去掉第三列（索引为 2）值为 0 的行
df = df[df.iloc[:, 2] != 0]

# 筛选出特定的烃类子结构
hydrocarbon_substructures = ['n1ccccc1', 'o1cccc1']
df = df[df['substructure'].isin(hydrocarbon_substructures)]

# 统计每个子结构的出现次数和四分位信息
substructure_stats = df.groupby('substructure').agg(
    出现次数=('smiles', 'nunique'),
    平均贡献=('average_contribution', 'mean'),
    下四分位=('average_contribution', lambda x: x.quantile(0.25)),  # 25% 分位数
    上四分位=('average_contribution', lambda x: x.quantile(0.75))   # 75% 分位数
).reset_index()

# 按平均贡献对数据进行排序
substructure_stats = substructure_stats.sort_values(by='平均贡献')

# 定义颜色映射，改为双色映射 'RdBu'
norm = plt.Normalize(vmin=0, vmax=1138)
cmap = plt.get_cmap('magma')

# 创建颜色列表
colors = [cmap(norm(count)) for count in substructure_stats['出现次数']]

# 固定柱子宽度
bar_width = 0.6
# 根据柱子数量动态调整图片宽度
num_bars = len(substructure_stats)
plt.figure(figsize=(num_bars * 1.5, 6))  # 这里 1.5 是一个调整因子，可以根据需要修改

ax = plt.gca()  # 获取当前 Axes

# 计算误差范围并取绝对值
lower_error = np.abs(substructure_stats['平均贡献'] - substructure_stats['下四分位'])
upper_error = np.abs(substructure_stats['上四分位'] - substructure_stats['平均贡献'])
yerr = [lower_error, upper_error]

# 使用索引作为 x 位置绘制柱状图
bars = ax.bar(
    np.arange(num_bars),
    substructure_stats['平均贡献'],
    width=bar_width,
    color=colors,
    yerr=yerr,
    capsize=5,  # 误差条顶部的横线长度
    alpha=0.7
)

# 设置 x 轴标签
ax.set_xticks(np.arange(num_bars))
ax.set_xticklabels(substructure_stats['substructure'], rotation=45)

plt.ylabel('平均贡献')
plt.title('含杂环')

# 设置 y 轴范围为 0-1
ax.set_ylim(0, 1)

# 添加颜色条
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)  # 指定颜色条的 Axes
cbar.set_label('出现次数')

plt.tight_layout()

# 保存图形为 SVG 格式
plt.savefig('GraphSAGE_weight/含杂环.pdf', format='pdf')

plt.show()