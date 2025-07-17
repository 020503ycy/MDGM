import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.rdMolDescriptors import CalcTPSA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt

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

# 读取数据
data = pd.read_csv('train.csv')
smiles = data.iloc[:, 1]
pmf = data.iloc[:, 2]

# 计算分子特征（直接在循环中计算，去掉原函数定义）
features = []
for s in smiles:
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        features.append([np.nan] * len(common_descriptors))
        continue
    desc_values = []
    for desc in common_descriptors:
        if desc == 'TPSA':
            desc_values.append(CalcTPSA(mol))
        else:
            desc_func = getattr(Descriptors, desc, None)
            if desc_func:
                desc_values.append(desc_func(mol))
            else:
                desc_values.append(np.nan)  # 处理不存在的描述符
    features.append(desc_values)

feature_names = common_descriptors
features_df = pd.DataFrame(features, columns=feature_names)

# 处理缺失值（删除含有整列缺失的特征）
features_df = features_df.dropna(axis=1)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    features_df, pmf, test_size=0.2, random_state=42
)

# 定义模型
models = {
    'XGBoost': xgb.XGBRegressor(),
    'LightGBM': lgb.LGBMRegressor(),
    'Random Forest': RandomForestRegressor()
}

# 训练和评估模型
results = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = np.mean((y_test - y_pred) ** 2)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = np.corrcoef(y_test, y_pred)[0, 1] ** 2
    results[model_name] = {
        'MSE': mse,
        'MAE': mae,
        'R^2': r2
    }

# 输出结果
for model_name, metrics in results.items():
    print(f'{model_name}:')
    print(f'Mean Squared Error: {metrics["MSE"]:.4f}')
    print(f'Mean Absolute Error: {metrics["MAE"]:.4f}')
    print(f'R^2 Score: {metrics["R^2"]:.4f}')
    print()

# 绘制柱状图展示验证结果
model_names = list(results.keys())
mse_values = [results[model]['MSE'] for model in model_names]
mae_values = [results[model]['MAE'] for model in model_names]
r2_values = [results[model]['R^2'] for model in model_names]

x = np.arange(len(model_names))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width, mse_values, width, label='MSE')
rects2 = ax.bar(x, mae_values, width, label='MAE')
rects3 = ax.bar(x + width, r2_values, width, label='R^2')

ax.set_ylabel('Scores')
ax.set_title('Model Performance Comparison')
ax.set_xticks(x)
ax.set_xticklabels(model_names)
ax.legend()

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(
            f'{height:.2f}',
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center', va='bottom'
        )

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

fig.tight_layout()

# 保存为 SVG 图片
plt.savefig('model_performance.svg', format='svg')
plt.show()