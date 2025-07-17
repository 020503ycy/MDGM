import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.rdMolDescriptors import CalcTPSA
import joblib
import shap
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler

# 定义标准分子描述符
common_descriptors = [
    'MolWt',       # Molecular weight
    'NumHDonors',  # Number of hydrogen bond donors
    'NumHAcceptors',# Number of hydrogen bond acceptors
    'TPSA',        # Topological polar surface area
    'LabuteASA',    # Approximate molecular surface area
    'MolLogP',     # Lipophilicity
    'NumRotatableBonds',
    'NumAliphaticRings',
    'NumAromaticRings',
    'NumSaturatedRings'
]

# 计算分子描述符
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

# 预测并生成瀑布图
def predict_and_plot_shap():
    # 加载模型和预处理工具
    model_path = 'RF_weight/random_forest_model.joblib'
    scaler_path = 'RF_weight/scaler.joblib'
    descriptors_path = 'RF_weight/common_descriptors.joblib'

    rf_model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    common_descriptors = joblib.load(descriptors_path)

    # 读取输入文件
    input_file = 'smiles.txt'
    try:
        df = pd.read_csv(input_file, delimiter='\t', encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(input_file, delimiter='\t', encoding='gbk')

    # 检查列名是否正确
    if 'smiles' not in df.columns or 'num' not in df.columns:
        raise ValueError("Input file must contain 'num' and 'smiles' columns.")

    # 准备特征
    features = []
    for smiles in df['smiles']:
        feat = calculate_descriptors(smiles, common_descriptors)
        features.append(feat)
    features = np.nan_to_num(features, nan=0)  # 处理缺失值
    scaled_features = scaler.transform(features)  # 特征缩放

    # 预测
    predictions = rf_model.predict(scaled_features)

    # 生成SHAP解释
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(scaled_features)

    # 创建输出目录
    output_dir = 'shap_waterfall_plots_pred'
    os.makedirs(output_dir, exist_ok=True)

    # 为每个样本绘制瀑布图
    for idx in range(len(df)):
        sample_num = df['num'].iloc[idx]
        shap_value = shap_values[idx]
        explanation = shap.Explanation(
            values=shap_value,
            base_values=explainer.expected_value,
            feature_names=common_descriptors
        )

        # 绘制瀑布图
        plt.figure(figsize=(8, 4))
        shap.plots.waterfall(explanation, show=False)
        plt.title(f'Sample {sample_num} SHAP Waterfall Plot')
        plt.xlabel('SHAP Value')
        plt.ylabel('Feature Name')
        plt.tight_layout()

        # 保存为PDF
        save_path = os.path.join(output_dir, f'sample_{sample_num}_waterfall.pdf')
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()

    # 保存预测结果
    df['predictions'] = predictions
    df.to_csv('rf_predict_results.csv', index=False, encoding='utf-8')
    print(f"Prediction completed. Results saved in rf_predict_results.csv.")
    print(f"Waterfall plots saved in {output_dir} folder.")

if __name__ == "__main__":
    predict_and_plot_shap()