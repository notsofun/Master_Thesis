import random
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from typing import List, Dict, Any

# ==========================================
# 2. 向量引擎 (核心：E5 / BERT)
# ==========================================

class VectorEngine:
    def __init__(self, model_name: str = 'intfloat/multilingual-e5-large'):
        print(f"正在加载 Embedding 模型: {model_name} ...")
        self.model = SentenceTransformer(model_name)
        print("模型加载完成。")

    def encode(self, texts: List[str]) -> np.ndarray:
        # e5 模型通常需要添加 "query: " 前缀来获得更好的效果
        formatted_texts = [f"query: {t}" for t in texts]
        return self.model.encode(formatted_texts, normalize_embeddings=True)

    def select_diverse_seeds(self, seed_data: List[str], n_samples: int = 3) -> List[str]:
        """
        利用 K-Means 聚类，从不同簇中选取种子，确保 prompt 里的示例差异化最大。
        """
        if len(seed_data) <= n_samples:
            return seed_data
        
        vectors = self.encode(seed_data)
        kmeans = KMeans(n_clusters=n_samples, random_state=42, n_init=10)
        kmeans.fit(vectors)
        
        selected_seeds = []
        # 从每个簇中随机选一个
        for i in range(n_samples):
            cluster_indices = np.where(kmeans.labels_ == i)[0]
            if len(cluster_indices) > 0:
                selected_idx = np.random.choice(cluster_indices)
                selected_seeds.append(seed_data[selected_idx])
        
        return selected_seeds

    def filter_generated_data(self, seed_texts: List[str], generated_texts: List[Dict], 
                              min_sim: float = 0.60, max_sim: float = 0.95) -> List[Dict]:
        """
        回环验证：计算生成的文本向量与原始种子质心的距离。
        过滤掉太远(跑题)和太近(抄袭)的数据。
        """
        if not generated_texts:
            return []

        seed_vectors = self.encode(seed_texts)
        centroid = np.mean(seed_vectors, axis=0).reshape(1, -1)
        
        texts_only = [item['text'] for item in generated_texts]
        gen_vectors = self.encode(texts_only)
        
        # 计算余弦相似度
        similarities = cosine_similarity(gen_vectors, centroid).flatten()
        
        valid_data = []
        for i, sim in enumerate(similarities):
            item = generated_texts[i]
            item['similarity_score'] = float(sim)
            
            if min_sim <= sim <= max_sim:
                valid_data.append(item)
            else:
                print(f"过滤数据: (Sim: {sim:.4f}) - {item['text'][:30]}...")
                
        return valid_data