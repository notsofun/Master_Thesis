import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from sklearn.metrics import pairwise_distances_argmin_min

# ================= 配置区 =================
INPUT_PATH = 'data_collection/English_Existing/merged_deduped.csv'
OUTPUT_PATH = 'data_augmentation/LLM/generated_texts/sampled_english_for_translation.csv'
TEXT_COLUMN = 'text'  # 英文 CSV 中文本所在的列名

# 抽样参数
NUM_CLUSTERS = 50      # 想要涵盖多少个不同的攻击维度/角度
SAMPLES_PER_CLUSTER = 50 # 每个维度抽多少条“经典”文本
# 总计将获得 50 * 50 = 2500 条极具代表性的英文样本
# ==========================================

def sample_classic_diverse_english():
    print(f"📖 正在加载数据: {INPUT_PATH}")
    if not os.path.exists(INPUT_PATH):
        print("❌ 错误：找不到输入文件。")
        return

    df = pd.read_csv(INPUT_PATH)
    # 去重并清洗空值
    raw_texts = df[TEXT_COLUMN].dropna().unique().tolist()
    print(f"✅ 加载完成，共有 {len(raw_texts)} 条唯一文本。")

    # 1. 向量化 (使用最快的英文专用轻量模型)
    print("🚀 正在生成语义向量 (Embedding)... 这可能需要 1-3 分钟")
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    embeddings = model.encode(raw_texts, batch_size=64, show_progress_bar=True)

    # 2. 执行聚类以确保“多元性”
    print(f"划分语义簇 (Clustering into {NUM_CLUSTERS} groups)...")
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, n_init=10, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)
    centroids = kmeans.cluster_centers_

    # 3. 在每个簇内寻找“经典”样本
    print("🎯 正在提取每个簇的质心代表 (Classic Samples)...")
    sampled_indices = []
    
    for i in range(NUM_CLUSTERS):
        # 获取属于当前簇的所有样本索引
        idx_in_cluster = np.where(cluster_labels == i)[0]
        cluster_vecs = embeddings[idx_in_cluster]
        
        # 计算该簇内所有点到簇质心的距离
        # 距离越小，代表性越强（越“经典”）
        dist_to_centroid = np.linalg.norm(cluster_vecs - centroids[i], axis=1)
        
        # 选取距离最近的前 N 个
        top_k_within_cluster = np.argsort(dist_to_centroid)[:SAMPLES_PER_CLUSTER]
        actual_indices = idx_in_cluster[top_k_within_cluster]
        sampled_indices.extend(actual_indices)

    # 4. 组装结果
    sampled_df = pd.DataFrame({
        'cluster_id': [cluster_labels[i] for i in sampled_indices],
        'original_en': [raw_texts[i] for i in sampled_indices]
    })

    # 保存
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    sampled_df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
    
    print("\n" + "="*30)
    print(f"✅ 抽样完成！")
    print(f"📊 总计抽样: {len(sampled_df)} 条")
    print(f"📂 已保存至: {OUTPUT_PATH}")
    print("="*30)

if __name__ == "__main__":
    sample_classic_diverse_english()