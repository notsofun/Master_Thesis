import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import matplotlib.cm as cm

# ================= 配置区 =================
FILE_PATH = r'unsupervised_classification\RQ1\data\rq1_topic_targets_summary.csv'
FONT_PATH = "msyh.ttc" # Windows 微软雅黑
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False
# ==========================================

def clean_target(target_str):
    """提取词汇并去除频率括号，将长句视作一个整体或进一步切分"""
    if pd.isna(target_str): return ""
    # 提取括号前的文字内容
    words = re.findall(r"([^',\[\]]+)\(\d+\)", target_str)
    # 将一个话题下的所有词/句合并为一个长字符串，用空格隔开
    return " ".join([w.strip() for w in words])

def get_top_one(target_str):
    """获取每个话题最核心的那个词作为标签"""
    match = re.search(r"([^',\[\]]+)\(\d+\)", str(target_str))
    return match.group(1).strip() if match else "Unknown"

# 1. 加载与预处理
df = pd.read_csv(FILE_PATH)
df['clean_text'] = df['Top_Targets'].apply(clean_target)
df['label'] = df['Top_Targets'].apply(get_top_one)

# 2. 向量化 (TF-IDF)
# 这里由于包含中日文字符，我们使用 char_wb 或者是默认分词（因为原始数据已经是逗号分隔的列表形态）
vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b") 
tfidf_matrix = vectorizer.fit_transform(df['clean_text'])

# 3. 降维到 2D (PCA)
pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(tfidf_matrix.toarray())

# 4. 绘图
plt.figure(figsize=(14, 10))
colors = cm.rainbow(np.linspace(0, 1, len(df)))

for i in range(len(df)):
    plt.scatter(coords[i, 0], coords[i, 1], color=colors[i], s=df.iloc[i]['Topic_Size']*2, alpha=0.6)
    # 给每个点加标签 (只显示 Top 1 关键词)
    plt.text(coords[i, 0] + 0.01, coords[i, 1] + 0.01, 
             f"{df.iloc[i]['Topic_ID']}:{df.iloc[i]['label']}", 
             fontsize=10, alpha=0.8)

plt.title("Topic 语义分布二维映射图\n(点的大小代表话题规模, 距离代表语义相似度)", fontsize=15)
plt.xlabel("维度 1 (Principal Component 1)")
plt.ylabel("维度 2 (Principal Component 2)")
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()