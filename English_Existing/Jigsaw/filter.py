import pandas as pd
import sys

# 保证输出 UTF-8，避免 Windows PowerShell 打印乱码
sys.stdout.reconfigure(encoding='utf-8')

# 文件路径
all_data_file = 'all_data.csv'
label_file = 'identity_individual_annotations.csv'
output_file = 'christian_comments.csv'

# 读取 CSV 文件
all_data = pd.read_csv(all_data_file)
labels = pd.read_csv(label_file)

# 过滤出宗教为 Christian 的 ID
christian_ids = labels[labels['religion'] == 'christian']['id']

# 根据 ID 在 all_data 中匹配 comment text
christian_comments = all_data[all_data['id'].isin(christian_ids)][['id', 'comment_text']]

# 输出到新的 CSV
christian_comments.to_csv(output_file, index=False)

print(f"已生成文件: {output_file}, 共 {len(christian_comments)} 条记录")
