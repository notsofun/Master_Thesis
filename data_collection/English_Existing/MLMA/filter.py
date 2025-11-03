import pandas as pd
import sys

# 保证输出 UTF-8，避免 Windows PowerShell 打印乱码
sys.stdout.reconfigure(encoding='utf-8')

# 文件路径
input_file = 'en_dataset.csv'        # 原始 CSV 文件
output_file = 'christian_tweets.csv'  # 输出 CSV 文件

# 读取 CSV
df = pd.read_csv(input_file, encoding='utf-8-sig', skipinitialspace=True)
print(df.columns)  # 查看 pandas 识别的列名
print(df.head())


# 筛选 target 为 christian 的行
christian_tweets = df[df['group'].str.lower() == 'christian']

# 只保留 HITId 和 tweet 字段
christian_tweets = christian_tweets[['HITId', 'tweet']]

# 输出到新 CSV
christian_tweets.to_csv(output_file, index=False)

print(f"已生成文件: {output_file}, 共 {len(christian_tweets)} 条记录")
