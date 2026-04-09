import pandas as pd

# 1. 加载数据（确保该文件在你的当前目录下）
df = pd.read_csv(r'unsupervised_classification\RQ3\data\rq3_bias_matrix.csv')

# 2. 选择代表五个道德基础的列
# 这些列对应：伤害、公平、忠诚、权威、圣洁
cols = ['bias_harm', 'bias_fairness', 'bias_loyalty', 'bias_authority', 'bias_sanctity']

# 3. 计算皮尔逊相关系数
correlation_matrix = df[cols].corr()

# 4. 打印结果
print(correlation_matrix)

# 如果你想看平均相关性（剔除对角线的1）
import numpy as np
corr_values = correlation_matrix.values
np.fill_diagonal(corr_values, np.nan)
print(f"\n平均相关系数: {np.nanmean(corr_values):.4f}")