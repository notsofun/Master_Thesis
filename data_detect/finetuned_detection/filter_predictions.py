import os
import glob
import pandas as pd

# 获取当前目录
current_dir = os.getcwd()

# 查找所有 *_predictions.csv 文件
prediction_files = glob.glob(os.path.join(current_dir, '*_predictions.csv'))

for file_path in prediction_files:
    # 读取 CSV 文件
    df = pd.read_csv(file_path)
    
    # 筛选 is_religion_related 和 is_hate_speech 都为 1 的行
    filtered_df = df[(df['is_religion_related'] == 1) & (df['is_hate_speech'] == 1)]
    
    # 只保留 text 字段
    output_df = filtered_df[['text']]
    
    # 生成输出文件名
    base_name = os.path.basename(file_path)
    output_name = base_name.replace('_predictions.csv', '_final_religious_hate.csv')
    output_path = os.path.join(current_dir, output_name)
    
    # 保存到新 CSV 文件
    output_df.to_csv(output_path, index=False)
    
    print(f"Filtered data saved to {output_path}")