import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import umap
import hdbscan
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ---------------------------
# 1. 配置与模型初始化
# ---------------------------
model_name = 'intfloat/multilingual-e5-large-instruct'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)

CORPUS_PATHS = {
    'zh': 'data_detect/finetuned_detection/chinese_final_religious_hate.csv',
    'en': 'data_collection/English_Existing/merged_deduped.csv',
    'jp': 'data_detect/finetuned_detection/japanese_final_religious_hate.csv'
}

def get_embeddings(texts, instruction, batch_size=16):
    processed_texts = [f"Instruct: {instruction}\nQuery: {t}" for t in texts]
    all_embeddings = []
    
    for i in range(0, len(processed_texts), batch_size):
        batch = processed_texts[i : i + batch_size]
        inputs = tokenizer(batch, max_length=512, padding=True, truncation=True, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu().numpy())
            
    return np.vstack(all_embeddings)

# ---------------------------
# 2. 加载与预处理数据
# ---------------------------
all_dfs = []
for lang, path in CORPUS_PATHS.items():
    if os.path.exists(path):
        df = pd.read_csv(path)
        df = df[['text']].dropna().head(500) # 每种语言先取500条测试，多了可以去掉head
        df['lang'] = lang
        all_dfs.append(df)
    else:
        print(f"警告: 找不到路径 {path}")

df_total = pd.concat(all_dfs, ignore_index=True)
texts = df_total['text'].tolist()

# ---------------------------
# 3. 提取特征与降维
# ---------------------------
instruction = "Identify and cluster hate speech patterns targeting Christianity, its clergy, and belief systems."
print(f"正在为 {len(texts)} 条文本生成嵌入...")
embeddings = get_embeddings(texts, instruction)

# UMAP 降维到 2D 用于可视化
print("正在进行 UMAP 降维...")
reducer = umap.UMAP(n_neighbors=15, n_components=2, metric='cosine', random_state=42)
embeddings_2d = reducer.fit_transform(embeddings)
df_total['x'] = embeddings_2d[:, 0]
df_total['y'] = embeddings_2d[:, 1]

# ---------------------------
# 4. 执行聚类
# ---------------------------
# K-means (假设你想要分4类：3个理论分类 + 1个其他)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df_total['cluster_kmeans'] = kmeans.fit_predict(embeddings)

# HDBSCAN (自动发现聚类，-1 为噪声)
hdb = hdbscan.HDBSCAN(min_cluster_size=10, metric='euclidean', cluster_selection_method='eom')
df_total['cluster_hdbscan'] = hdb.fit_predict(embeddings_2d) # 密度聚类建议在降维后做

# ---------------------------
# 5. 可视化输出
# ---------------------------
plt.figure(figsize=(20, 6))

# 图1：按语言分布 (查看是否存在语言隔离)
plt.subplot(1, 3, 1)
sns.scatterplot(data=df_total, x='x', y='y', hue='lang', palette='Set1', s=20, alpha=0.6)
plt.title('Distribution by Language (zh/en/jp)')

# 图2：K-means 聚类结果
plt.subplot(1, 3, 2)
sns.scatterplot(data=df_total, x='x', y='y', hue='cluster_kmeans', palette='viridis', s=20, alpha=0.6)
plt.title('K-Means Clustering (n=4)')

# 图3：HDBSCAN 聚类结果
plt.subplot(1, 3, 3)
# HDBSCAN 通常会有 -1 类（噪声），用黑色标记
sns.scatterplot(data=df_total, x='x', y='y', hue='cluster_hdbscan', palette='tab20', s=20, alpha=0.6)
plt.title('HDBSCAN Clustering (Auto-detected)')

plt.tight_layout()
plt.savefig('unsupervised_classification/pics/clustering_comparison.png', dpi=300)
print("可视化图片已保存为 clustering_comparison.png")
plt.show()

# 存储结果方便后续分析
df_total.to_csv('unsupervised_classification/result/clustering_results.csv', index=False, encoding='utf-8-sig')