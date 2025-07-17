import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import shap
import matplotlib
matplotlib.use('Agg')  # 强制使用非交互式后端（适用于保存文件）
import matplotlib.pyplot as plt
import os
import joblib
from rdkit.Chem.rdMolDescriptors import CalcTPSA
from scipy.stats import linregress, t

# 定义标准分子描述符（与之前保持一致）
common_descriptors = [
    'MolWt',       # 分子量
    'NumHDonors',  # 氢键供体数
    'NumHAcceptors',# 氢键受体数
    'TPSA',        # 莫尔极性表面积
    'LabuteASA',    # 可近似表示分子表面积
    'MolLogP',     # 脂溶性
    'NumRotatableBonds',
    'NumAliphaticRings',
    'NumAromaticRings',
    'NumSaturatedRings'
]

# 计算分子描述符（仅保留标准描述符）
def calculate_descriptors(smiles, descriptor_list):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        descriptors = []
        for descriptor_name in descriptor_list:
            if descriptor_name == 'TPSA':
                descriptors.append(CalcTPSA(mol))
            else:
                descriptor_function = getattr(Descriptors, descriptor_name, None)
                if descriptor_function:
                    descriptors.append(descriptor_function(mol))
                else:
                    descriptors.append(np.nan)
        return descriptors
    return [np.nan] * len(descriptor_list)

# 训练模型
def train_model():
    # 加载数据
    train_data = pd.read_csv('RF_weight/train.csv')
    num_ids = train_data['num'].values  # 加载 num 列
    smiles = train_data['smiles']
    labels = train_data['energy'].values  # 目标列

    # 计算特征
    features = []
    for s in smiles:
        features.append(calculate_descriptors(s, common_descriptors))
    features = np.array(features)

    # 处理缺失值
    features = np.nan_to_num(features, nan=0)  # 用0填充缺失值（或根据需求调整）

    # 数据划分（7:2:1）
    X_train_val, X_test, y_train_val, y_test, num_train_val, num_test = train_test_split(
        features, labels, num_ids, test_size=0.1, random_state=42
    )
    X_train, X_val, y_train, y_val, num_train, num_val = train_test_split(
        X_train_val, y_train_val, num_train_val, test_size=0.2222, random_state=42  # 20% of 90% = 18%总数据，最终7:2:1
    )

    # 特征缩放
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # 训练模型
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # 预测
    y_train_pred = rf_model.predict(X_train)
    y_val_pred = rf_model.predict(X_val)
    y_test_pred = rf_model.predict(X_test)

    # 评估指标
    def print_metrics(y_true, y_pred, name):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        print(f"{name} Metrics: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")

    print_metrics(y_train, y_train_pred, "Training Set")
    print_metrics(y_val, y_val_pred, "Validation Set")
    print_metrics(y_test, y_test_pred, "Test Set")

    # 保存模型和缩放器
    joblib.dump(rf_model, 'RF_weight/random_forest_model.joblib')
    joblib.dump(scaler, 'RF_weight/scaler.joblib')
    joblib.dump(common_descriptors, 'RF_weight/common_descriptors.joblib')

    return rf_model, scaler, X_train, common_descriptors, \
           y_train, y_train_pred, y_val, y_val_pred, y_test, y_test_pred, num_train

# 期刊级绘图函数（整合用户提供的绘图代码）
def plot_performance(y_train, y_train_pred,
                     y_val, y_val_pred,
                     y_test, y_test_pred):

    # 期刊级配色方案
    COLORS = {'train': '#81021f', 'val': '#435f87', 'test': '#fdd95f'}

    # 创建画布
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    fig.subplots_adjust(left=0.15, right=0.9, bottom=0.15, top=0.9)

    # 定义绘图函数
    def plot_dataset(y_true, y_pred, name, color):
        y_true = y_true.ravel()
        y_pred = y_pred.ravel()

        # 散点图
        scatter = ax.scatter(y_true, y_pred, c=color, alpha=0.8, s=40,
                             edgecolor='white', linewidth=0.8, zorder=2)

        # 拟合线
        slope, intercept, r_value, p_value, std_err = linregress(y_true, y_pred)
        x_range = np.linspace(min(y_true.min(), y_pred.min()),
                              max(y_true.max(), y_pred.max()), 100)
        line, = ax.plot(x_range, slope * x_range + intercept, '--',
                        color=color, linewidth=1.2, alpha=0.7, zorder=1)

        # 置信区间
        residuals = y_pred - y_true
        se = np.std(residuals, ddof=2)
        n = len(y_true)
        t_val = t.ppf(0.975, n - 2)
        ci_low = slope * x_range + intercept - t_val * se * np.sqrt(1 / n + (x_range - np.mean(y_true)) ** 2 / ((n - 1) * np.var(y_true)))
        ci_high = slope * x_range + intercept + t_val * se * np.sqrt(1 / n + (x_range - np.mean(y_true)) ** 2 / ((n - 1) * np.var(y_true)))
        fill = ax.fill_between(x_range, ci_low, ci_high, color=color, alpha=0.15, zorder=0)

        # 指标标签
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        label_text = f'{name}: MAE={mae:.2f} | RMSE={rmse:.2f}'

        # 创建图例句柄
        handle = plt.Line2D([], [], marker='o', color='none',
                            markerfacecolor=color, markeredgecolor='white',
                            markeredgewidth=0.8, markersize=8)
        return handle, label_text

    # 绘制各数据集
    datasets = [
        (y_train, y_train_pred, 'Training Set', COLORS['train']),
        (y_val, y_val_pred, 'Validation Set', COLORS['val']),
        (y_test, y_test_pred, 'Test Set', COLORS['test'])
    ]

    legend_handles = []
    legend_labels = []
    for yt, yp, name, color in datasets:
        handle, label = plot_dataset(yt, yp, name, color)
        legend_handles.append(handle)
        legend_labels.append(label)

    # 对角线
    lim = [ax.get_xlim()[0], ax.get_xlim()[1]]
    ax.plot(lim, lim, 'k--', alpha=0.8, linewidth=1.5, zorder=3)

    # 坐标轴设置
    ax.set(xlabel='True Values (kcal/mol)',
           ylabel='Predicted Values (kcal/mol)',
           aspect='equal', xlim=lim, ylim=lim)

    # 图例
    ax.legend(legend_handles, legend_labels, loc='upper left', bbox_to_anchor=(0.02, 0.98),
              frameon=True, framealpha=0.9, edgecolor='white', fontsize=10)

    # 细节优化
    ax.grid(linestyle=':', linewidth=0.5, alpha=0.7, zorder=-1)
    ax.spines[['top', 'right']].set_visible(False)

    # 保存和显示
    plt.savefig('RF_MDGM_predict.pdf', format='pdf', bbox_inches='tight')
    plt.show()

def shap_analysis(rf_model, X_train, common_descriptors, num_train):
    # 创建 SHAP 解释器
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_train)

    # 绘制 SHAP 总图
    shap.summary_plot(shap_values, X_train, feature_names=common_descriptors)
    plt.savefig('shap_summary_plot.pdf', format='pdf', dpi=300, bbox_inches='tight')  # 修改为PDF
    plt.close()

    # 计算特征重要性
    feature_importance = np.abs(shap_values).mean(axis=0)
    top_four_indices = np.argsort(feature_importance)[-5:][::-1]
    top_four_features = [common_descriptors[i] for i in top_four_indices]

    # 绘制排名前四的特征取向图
    for i, feature_index in enumerate(top_four_indices):
        feature_name = common_descriptors[feature_index]
        shap.dependence_plot(
            ind=feature_index,
            shap_values=shap_values,
            features=X_train,
            feature_names=common_descriptors,
            interaction_index=None,
            show=False
        )
        plt.xlabel(feature_name)
        plt.ylabel('SHAP Value')
        plt.tight_layout()
        plt.savefig(f'shap_dependence_plot_{feature_name}.pdf', format='pdf', dpi=300, bbox_inches='tight')  # 修改为PDF
        plt.close()

    # 绘制各个样本的瀑布图
    save_dir = 'shap_waterfall_plots'
    os.makedirs(save_dir, exist_ok=True)
    for idx in range(X_train.shape[0]):
        sample = X_train[idx:idx + 1]
        shap_value = shap_values[idx]
        explanation = shap.Explanation(values=shap_value,
                                       base_values=explainer.expected_value,
                                       feature_names=common_descriptors)
        shap.plots.waterfall(explanation)
        sample_num = num_train[idx]  # 获取对应的 num 值
        plt.title(f'Sample {sample_num} SHAP Waterfall Plot')
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'sample_{sample_num}_waterfall_plot.pdf')  # 修改为PDF
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    # 训练并获取数据
    rf_model, scaler, X_train, common_descriptors, \
    y_train, y_train_pred, y_val, y_val_pred, y_test, y_test_pred, num_train = train_model()

    # 绘制性能图
    plot_performance(y_train, y_train_pred,
                     y_val, y_val_pred,
                     y_test, y_test_pred)

    # SHAP 分析
    shap_analysis(rf_model, X_train, common_descriptors, num_train)


import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.preprocessing import StandardScaler
import joblib
from rdkit.Chem.rdMolDescriptors import CalcTPSA

# 定义标准分子描述符
common_descriptors = [
    'MolWt',       # 分子量
    'NumHDonors',  # 氢键供体数
    'NumHAcceptors',# 氢键受体数
    'TPSA',        # 莫尔极性表面积
    'LabuteASA',    # 可近似表示分子表面积
    'MolLogP',     # 脂溶性
    'NumRotatableBonds',
    'NumAliphaticRings',
    'NumAromaticRings',
    'NumSaturatedRings'
]

# 计算分子描述符（仅保留标准描述符）
def calculate_descriptors(smiles, descriptor_list):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        descriptors = []
        for descriptor_name in descriptor_list:
            if descriptor_name == 'TPSA':
                descriptors.append(CalcTPSA(mol))
            else:
                descriptor_function = getattr(Descriptors, descriptor_name, None)
                if descriptor_function:
                    descriptors.append(descriptor_function(mol))
                else:
                    descriptors.append(np.nan)
        return descriptors
    return [np.nan] * len(descriptor_list)

def predict_from_smiles():
    # 加载模型和缩放器
    rf_model = joblib.load('RF_weight/random_forest_model.joblib')
    scaler = joblib.load('RF_weight/scaler.joblib')
    common_descriptors = joblib.load('RF_weight/common_descriptors.joblib')

    # 读取 CSV 文件
    data = pd.read_csv('RF_weight/test.csv')
    smiles = data['smiles']

    # 计算特征
    features = []
    for s in smiles:
        features.append(calculate_descriptors(s, common_descriptors))
    features = np.array(features)

    # 处理缺失值
    features = np.nan_to_num(features, nan=0)

    # 特征缩放
    scaled_features = scaler.transform(features)

    # 进行预测
    predictions = rf_model.predict(scaled_features)

    # 创建输出目录
    output_dir = 'RF_weight/BDBS_RF'
    os.makedirs(output_dir, exist_ok=True)

    # 保存预测结果到 CSV 文件
    output_df = pd.DataFrame({'smiles': smiles, 'predictions': predictions})
    output_path = os.path.join(output_dir, 'predictions.csv')
    output_df.to_csv(output_path, index=False)

    return predictions

if __name__ == "__main__":
    import os
    predictions = predict_from_smiles()
    print("Predictions have been saved to RF_weight/BDBS_RF/predictions.csv")

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.preprocessing import StandardScaler
import joblib
from rdkit.Chem.rdMolDescriptors import CalcTPSA

# 定义标准分子描述符
common_descriptors = [
    'MolWt',       # 分子量
    'NumHDonors',  # 氢键供体数
    'NumHAcceptors',# 氢键受体数
    'TPSA',        # 莫尔极性表面积
    'LabuteASA',    # 可近似表示分子表面积
    'MolLogP',     # 脂溶性
    'NumRotatableBonds',
    'NumAliphaticRings',
    'NumAromaticRings',
    'NumSaturatedRings'
]

# 计算分子描述符（仅保留标准描述符）
def calculate_descriptors(smiles, descriptor_list):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        descriptors = []
        for descriptor_name in descriptor_list:
            if descriptor_name == 'TPSA':
                descriptors.append(CalcTPSA(mol))
            else:
                descriptor_function = getattr(Descriptors, descriptor_name, None)
                if descriptor_function:
                    descriptors.append(descriptor_function(mol))
                else:
                    descriptors.append(np.nan)
        return descriptors
    return [np.nan] * len(descriptor_list)

def predict_from_smiles():
    # 加载模型和缩放器
    rf_model = joblib.load('RF_weight/random_forest_model.joblib')
    scaler = joblib.load('RF_weight/scaler.joblib')
    common_descriptors = joblib.load('RF_weight/common_descriptors.joblib')

    # 读取 CSV 文件
    data = pd.read_csv('RF_weight/train.csv')
    smiles = data['smiles']

    # 计算特征
    features = []
    for s in smiles:
        features.append(calculate_descriptors(s, common_descriptors))
    features = np.array(features)

    # 处理缺失值
    features = np.nan_to_num(features, nan=0)

    # 特征缩放
    scaled_features = scaler.transform(features)

    # 进行预测
    predictions = rf_model.predict(scaled_features)

    # 创建输出目录
    output_dir = 'RF_weight/MDGM_RF'
    os.makedirs(output_dir, exist_ok=True)

    # 保存预测结果到 CSV 文件
    output_df = pd.DataFrame({'smiles': smiles, 'predictions': predictions})
    output_path = os.path.join(output_dir, 'predictions.csv')
    output_df.to_csv(output_path, index=False)

    return predictions

if __name__ == "__main__":
    import os
    predictions = predict_from_smiles()
    print("Predictions have been saved to RF_weight/MDGM.csv")

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.preprocessing import StandardScaler
import joblib
from rdkit.Chem.rdMolDescriptors import CalcTPSA

# 定义标准分子描述符
common_descriptors = [
    'MolWt',       # 分子量
    'NumHDonors',  # 氢键供体数
    'NumHAcceptors',# 氢键受体数
    'TPSA',        # 莫尔极性表面积
    'LabuteASA',    # 可近似表示分子表面积
    'MolLogP',     # 脂溶性
    'NumRotatableBonds',
    'NumAliphaticRings',
    'NumAromaticRings',
    'NumSaturatedRings'
]

# 计算分子描述符（仅保留标准描述符）
def calculate_descriptors(smiles, descriptor_list):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        descriptors = []
        for descriptor_name in descriptor_list:
            if descriptor_name == 'TPSA':
                descriptors.append(CalcTPSA(mol))
            else:
                descriptor_function = getattr(Descriptors, descriptor_name, None)
                if descriptor_function:
                    descriptors.append(descriptor_function(mol))
                else:
                    descriptors.append(np.nan)
        return descriptors
    return [np.nan] * len(descriptor_list)

def predict_from_smiles():
    # 加载模型和缩放器
    rf_model = joblib.load('RF_weight/random_forest_model.joblib')
    scaler = joblib.load('RF_weight/scaler.joblib')
    common_descriptors = joblib.load('RF_weight/common_descriptors.joblib')

    # 读取 CSV 文件
    data = pd.read_csv('RF_weight/B3DB.csv')
    smiles = data['smiles']

    # 计算特征
    features = []
    for s in smiles:
        features.append(calculate_descriptors(s, common_descriptors))
    features = np.array(features)

    # 处理缺失值
    features = np.nan_to_num(features, nan=0)

    # 特征缩放
    scaled_features = scaler.transform(features)

    # 进行预测
    predictions = rf_model.predict(scaled_features)

    # 创建输出目录
    output_dir = 'RF_weight/B3DB_RF'
    os.makedirs(output_dir, exist_ok=True)

    # 保存预测结果到 CSV 文件
    output_df = pd.DataFrame({'smiles': smiles, 'predictions': predictions})
    output_path = os.path.join(output_dir, 'predictions.csv')
    output_df.to_csv(output_path, index=False)

    return predictions

if __name__ == "__main__":
    import os
    predictions = predict_from_smiles()
    print("Predictions have been saved to RF_weight/B3DB.csv")
