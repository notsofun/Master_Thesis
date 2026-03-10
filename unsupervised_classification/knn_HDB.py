# python unsupervised_classification/knn_HDB.py
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import umap
import hdbscan
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
import seaborn as sns
import os, sys, re
import logging
from tqdm import tqdm  # 引入进度条
import jieba
from janome.tokenizer import Tokenizer as JanomeTokenizer
from nltk.tokenize import word_tokenize
import nltk

# 初始化分词器
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

jt = JanomeTokenizer()

# Get the directory where the script is located (unsupervised_classification)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory (Master_Thesis)
project_root = os.path.dirname(current_dir)

# Add the project root to sys.path
if project_root not in sys.path:
    sys.path.append(project_root)

from scripts.set_logger import setup_logging

# ---------------------------
# 0. 日志与路径管理配置
# ---------------------------
warnings.filterwarnings("ignore", category=UserWarning) 
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logger, _ = setup_logging()

# 确保输出目录存在
os.makedirs('unsupervised_classification/pics', exist_ok=True)
os.makedirs('unsupervised_classification/result', exist_ok=True)


# --- 新增：手动加载支持中日文的字体 ---
font_path = '/root/autodl-tmp/Master_Thesis/SourceHanSansSC-Regular.otf'
if os.path.exists(font_path):
    my_font = font_manager.FontProperties(fname=font_path)
    # 设置全局字体（可选，但有时对某些组件无效）
    plt.rcParams['font.family'] = my_font.get_name()
    plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题
    logger.info(f"已加载字体: {font_path}")
else:
    logger.warning("未找到中文字体文件，图片可能显示乱码。")
    

# ---------------------------
# 1. 配置与模型初始化
# ---------------------------
model_name = 'intfloat/multilingual-e5-large-instruct'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"正在使用设备: {device}")

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    logger.info(f"模型 {model_name} 加载成功。")
except Exception as e:
    logger.error(f"模型加载失败: {e}")
    raise

CORPUS_PATHS = {
    'zh': 'data_detect/finetuned_detection/chinese_final_religious_hate.csv',
    'en': 'data_collection/English_Existing/merged_deduped.csv',
    'jp': 'data_detect/finetuned_detection/japanese_final_religious_hate.csv'
}


def get_embeddings(texts, instruction, batch_size=16):
    processed_texts = [f"Instruct: {instruction}\nQuery: {t}" for t in texts]
    all_embeddings = []
    
    # 使用 tqdm 显示推理进度
    logger.info("开始生成语义嵌入 (Embedding)...")
    for i in tqdm(range(0, len(processed_texts), batch_size), desc="Inferencing"):
        batch = processed_texts[i : i + batch_size]
        inputs = tokenizer(batch, max_length=512, padding=True, truncation=True, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            # 提取 Mean Pooling 并归一化
            embeddings = outputs.last_hidden_state.mean(dim=1)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu().numpy())
            
    return np.vstack(all_embeddings)

def basic_tokenizer(text, lang):
    text = str(text).lower()
    if lang == 'zh':
        return " ".join(jieba.cut(text))
    if lang == 'jp':
        return " ".join([token.surface for token in jt.tokenize(text)])
    if lang == 'en':
        tokens = word_tokenize(text)
        return " ".join(tokens)
    return text

def get_keywords(df, cluster_col, n_words=5):
    """提取每个簇的关键词标签"""
    cluster_labels = {}
    unique_clusters = [c for c in df[cluster_col].unique() if c != -1] # 忽略噪声点-1
    
    for cluster in unique_clusters:
        texts_in_cluster = df[df[cluster_col] == cluster]['tokenized_text']
        if len(texts_in_cluster) == 0: continue
        
        # 使用 TF-IDF 找到该簇区别于其他簇的特征词
        vectorizer = TfidfVectorizer(max_features=100)
        tfidf_matrix = vectorizer.fit_transform(texts_in_cluster)
        words = vectorizer.get_feature_names_out()
        sums = tfidf_matrix.sum(axis=0).A1
        top_indices = sums.argsort()[-n_words:][::-1]
        cluster_labels[cluster] = "\n".join([words[i] for i in top_indices])
    
    return cluster_labels
# ---------------------------
# 2. 加载与预处理数据
# ---------------------------
all_dfs = []
for lang, path in CORPUS_PATHS.items():
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            df = df[['text']].dropna().head(100)
            df['lang'] = lang
            all_dfs.append(df)
            logger.info(f"成功加载 {lang} 语料: {len(df)} 条记录")
        except Exception as e:
            logger.warning(f"读取 {path} 时出错: {e}")
    else:
        logger.error(f"找不到路径: {path}")

if not all_dfs:
    logger.critical("未加载任何数据，程序退出。")
    exit()

df_total = pd.concat(all_dfs, ignore_index=True)
texts = df_total['text'].tolist()
logger.info(f"总计处理文本量: {len(texts)}")

# ---------------------------
# 3. 语义处理与聚类
# ---------------------------
# 预分词用于关键词提取
logger.info("执行多语言预分词...")
df_total['tokenized_text'] = df_total.apply(lambda row: basic_tokenizer(row['text'], row['lang']), axis=1)

instruction = "Identify and cluster hate speech patterns targeting Christianity, its clergy, and belief systems."
embeddings = get_embeddings(df_total['text'].tolist(), instruction)

logger.info("UMAP 降维...")
reducer = umap.UMAP(n_neighbors=15, n_components=2, metric='cosine', random_state=42)
embeddings_2d = reducer.fit_transform(embeddings)
df_total['x'], df_total['y'] = embeddings_2d[:, 0], embeddings_2d[:, 1]

# K-Means
logger.info("执行 K-Means...")
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df_total['cluster_kmeans'] = kmeans.fit_predict(embeddings)
keywords_kmeans = get_keywords(df_total, 'cluster_kmeans')

# HDBSCAN
logger.info("执行 HDBSCAN...")
hdb = hdbscan.HDBSCAN(min_cluster_size=10, metric='euclidean')
df_total['cluster_hdbscan'] = hdb.fit_predict(embeddings_2d)
keywords_hdbscan = get_keywords(df_total, 'cluster_hdbscan')

# ---------------------------
# 5. 可视化输出 (保留原版样式)
# ---------------------------
logger.info("正在生成对比图并标注关键词...")
plt.figure(figsize=(22, 7))
lang_palette = {'zh': '#e74c3c', 'en': '#3498db', 'jp': '#2ecc71'} # 红、蓝、绿

def plot_with_keywords(subplot_idx, title, cluster_col, keywords_dict):
    plt.subplot(1, 3, subplot_idx)
    sns.scatterplot(data=df_total, x='x', y='y', hue='lang', palette=lang_palette, s=15, alpha=0.4)
    
    for cluster, tags in keywords_dict.items():
        center_x = df_total[df_total[cluster_col] == cluster]['x'].mean()
        center_y = df_total[df_total[cluster_col] == cluster]['y'].mean()
        
        if not np.isnan(center_x):
            plt.text(
                center_x, center_y, tags,
                fontsize=8, 
                fontproperties=my_font, # <-- 这里是关键：显式指定加载的字体
                fontweight='bold', 
                color='black',
                ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.3')
            )
    plt.title(title, fontproperties=my_font) # 标题也加上

# 图1：原始语言分布（不标词）
plt.subplot(1, 3, 1)
sns.scatterplot(data=df_total, x='x', y='y', hue='lang', palette=lang_palette, s=15, alpha=0.6)
plt.title('Language Distribution (Ground Truth)')

# 图2：K-means 结果与词标签
plot_with_keywords(2, 'K-Means Clusters & Keywords', 'cluster_kmeans', keywords_kmeans)

# 图3：HDBSCAN 结果与词标签
plot_with_keywords(3, 'HDBSCAN Clusters & Keywords', 'cluster_hdbscan', keywords_hdbscan)

plt.tight_layout()
pic_path = 'unsupervised_classification/pics/clustering_keywords_comparison.png'
plt.savefig(pic_path, dpi=300)
logger.info(f"可视化图片已保存至: {pic_path}")

# 存储结果
df_total.to_csv('unsupervised_classification/result/clustering_results.csv', index=False, encoding='utf-8-sig')
logger.info("Process Completed.")
plt.show()