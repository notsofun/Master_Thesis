import pandas as pd
import numpy as np
import os
import sys
import time
import umap
from bertopic import BERTopic
from bertopic.representation import OpenAI
import openai
from scipy.spatial.distance import cdist
from dotenv import load_dotenv

load_dotenv()
import matplotlib.pyplot as plt
import matplotlib

# 检查你的系统中有哪些中文字体。
# Windows 通常用 'SimHei' (黑体) 或 'Microsoft YaHei' (微软雅黑)
# Mac 通常用 'Arial Unicode MS' 或 'Heiti TC'
# Linux (Ubuntu) 通常用 'Noto Sans CJK JP' 或 'WenQuanYi Micro Hei'

matplotlib.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体为黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

# 设置路径和日志 (假设你已有的 scripts.set_logger 可用)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from scripts.set_logger import setup_logging
    logger, _ = setup_logging(name='cluster_naming_openai')
except:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("OpenAI_Naming")

def load_and_rename_openai_pipeline(model_path, csv_path, embeddings_path, output_path, api_key):
    # --- 1. 加载并清洗数据 ---
    logger.info("正在加载数据与向量...")
    df = pd.read_csv(csv_path)
    all_embeddings = np.load(embeddings_path)
    
    # 核心：确保索引对齐，剔除非数字的 topic (比如误读的表头)
    df['temp_idx'] = range(len(df))
    df['topic'] = pd.to_numeric(df['topic'], errors='coerce')
    clean_df = df.dropna(subset=['topic', 'text']).copy()
    
    valid_indices = clean_df['temp_idx'].values
    embeddings = all_embeddings[valid_indices]
    texts = clean_df['text'].tolist()
    topics = clean_df['topic'].astype(int).tolist()
    
    # --- 2. 加载模型 ---
    logger.info("正在加载 BERTopic 模型...")
    topic_model = BERTopic.load(model_path, embedding_model=None)

    # --- 3. 【核心补全】手动计算代表性文档 (带物理截断版本) ---
    logger.info("正在手动计算所有话题的代表性文档并进行物理截断...")
    representative_docs = {}
    unique_topics = sorted([t for t in set(topics) if t != -1])
    
    for topic in unique_topics:
        indices = [i for i, t in enumerate(topics) if t == topic]
        if not indices: continue
        
        topic_vecs = embeddings[indices]
        centroid = np.mean(topic_vecs, axis=0).reshape(1, -1)
        
        distances = cdist(centroid, topic_vecs, metric='cosine')[0]
        top_3_local_idx = np.argsort(distances)[:3]
        
        # 【关键修改】：这里对 text 进行 [:500] 截断
        # 3篇文档 * 500字 = 1500字左右，加上关键词，绝对不会超过 OpenAI 的限制
        representative_docs[topic] = [str(texts[indices[i]])[:500] for i in top_3_local_idx]
    
    topic_model.representative_docs_ = representative_docs
    
    # 强行注入模型，这样 update_topics 才会处理所有话题
    topic_model.representative_docs_ = representative_docs

    # --- 4. 配置 OpenAI 表示模型 ---
    logger.info("配置 OpenAI 命名模型...")
    client = openai.OpenAI(api_key=api_key)
    
    prompt = """
    I have a topic that contains the following documents:
    [DOCUMENTS]
    The topic is described by the following keywords: [KEYWORDS]
    Based on the information above, extract a short, concise Chinese topic label.
    Format: topic: <label>
    """
    
    representation_model = OpenAI(
        client, 
        model="gpt-4o-mini", # 或 gpt-4o-mini 省钱
        chat=True, 
        prompt=prompt,
        nr_docs=3,        # 限制文档数，省钱！
        delay_in_seconds=1 # 避免触发速率限制
    )

    # --- 5. 执行更新 ---
    logger.info(f"开始调用 OpenAI 命名 {len(unique_topics)} 个话题...")
    topic_model.update_topics(
        texts, 
        topics=topics, 
        representation_model=representation_model
    )

    # 将生成的 OpenAI 标签设置为官方标签
    info = topic_model.get_topic_info()
    if "OpenAI" in info.columns:
        # 清洗标签格式
        new_labels = info["OpenAI"].apply(lambda x: x[0].replace("topic:", "").strip() if isinstance(x, list) else str(x)).tolist()
        topic_model.set_topic_labels(new_labels)
        logger.info("话题标签已更新。")

    # --- 6. 2D 降维与可视化 (解决 NaN 问题) ---
    logger.info("正在进行 2D 降维并生成 DataMap...")
    try:
        # 必须手动转为 2D，否则 datamapplot 会因为维度对不上而报 NaN
        reducer = umap.UMAP(n_components=2, random_state=42)
        embeddings_2d = reducer.fit_transform(embeddings)
        
        # 排除噪声话题进行绘图
        viz_topics = [t for t in sorted(list(set(topics))) if t >= 0][:30]
        
        fig = topic_model.visualize_document_datamap(
            texts,
            reduced_embeddings=embeddings_2d,
            topics=viz_topics,
            custom_labels=True
        )
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, bbox_inches='tight', dpi=300)
        logger.info(f"可视化已保存至: {output_path}")
    except Exception as e:
        logger.error(f"可视化失败: {e}")

    # --- 7. 保存最终模型 ---
    topic_model.save(model_path, serialization="safetensors")
    logger.info("所有流程已完成。")

if __name__ == "__main__":
    # 填入你的参数
    MY_API_KEY = os.environ.get("OPENAI_API_KEY")
    BASE_DIR = r"unsupervised_classification\topic_modeling_results\sixth"
    
    load_and_rename_openai_pipeline(
        model_path=os.path.join(BASE_DIR, "models", "bertopic_model"),
        csv_path=os.path.join(BASE_DIR, "data", "document_topic_mapping.csv"),
        embeddings_path=os.path.join(BASE_DIR, "models", "embeddings.npy"),
        output_path=os.path.join(BASE_DIR, "visualizations", "datamap_refined.png"),
        api_key=MY_API_KEY
    )