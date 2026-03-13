# python unsupervised_classification/bertopic_hate.py
import os
import logging
import warnings
import pandas as pd
import numpy as np
from datetime import datetime

# NLP & Topic Modeling
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
import jieba
from janome.tokenizer import Tokenizer as JanomeTokenizer
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. 严格的日志与警告管理
# ==========================================
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*TBB threading layer.*")

for lib in ['numba', 'matplotlib', 'hdbscan', 'umap', 'transformers', 'urllib3', 'bertopic']:
    logging.getLogger(lib).setLevel(logging.ERROR)

def setup_logger():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    log_dir = os.path.join(current_dir, 'logs')
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_filename = datetime.now().strftime("topic_pipeline_%Y%m%d_%H%M.log")
    
    logger = logging.getLogger("TopicPipeline")
    logger.setLevel(logging.INFO)
    
    # 避免重复绑定 handler
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        
        # 控制台输出
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        # 文件输出
        fh = logging.FileHandler(log_filename, encoding='utf-8')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
    return logger

logger = setup_logger()

# ==========================================
# 2. 全局配置与路径管理
# ==========================================
# 数据文件路径配置
DATA_PATHS = {
    'zh': 'data_detect/finetuned_detection/chinese_final_religious_hate.csv',
    'en': 'data_collection/English_Existing/merged_deduped.csv',
    'jp': 'data_detect/finetuned_detection/japanese_final_religious_hate.csv'
}

# 输出目录配置
OUTPUT_DIR = 'unsupervised_classification/topic_modeling_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)
for sub_dir in ['data', 'models', 'visualizations']:
    os.makedirs(os.path.join(OUTPUT_DIR, sub_dir), exist_ok=True)

# 确保 nltk 资源存在
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# ==========================================
# 3. 核心功能函数
# ==========================================
def load_and_sample_data(selected_langs):
    """根据选择的语言加载并采样数据"""
    logger.info(f"正在加载数据集，目标语言: {selected_langs}")
    df_list = []
    
    is_multi_lang = len(selected_langs) > 1
    
    for lang in selected_langs:
        if lang not in DATA_PATHS or not os.path.exists(DATA_PATHS[lang]):
            logger.error(f"找不到 {lang} 的数据文件: {DATA_PATHS.get(lang)}")
            continue
            
        df = pd.read_csv(DATA_PATHS[lang]).dropna(subset=['text'])
        df['lang'] = lang
        
        if is_multi_lang:
            # 多语言模式：随机采样 1000 条
            sample_size = min(1000, len(df))
            df = df.sample(n=sample_size, random_state=42)
            logger.info(f"[{lang}] 采样了 {sample_size} 条数据 (多语言模式).")
        else:
            # 单语言模式：保留全量数据
            logger.info(f"[{lang}] 加载了全部 {len(df)} 条数据 (单语言模式).")
            
        df_list.append(df)
        
    if not df_list:
        raise ValueError("没有成功加载任何数据。")
        
    final_df = pd.concat(df_list, ignore_index=True)
    logger.info(f"数据集构建完成，总计 {len(final_df)} 条数据。")
    return final_df

def pre_tokenize(texts, langs):
    """中日英文预分词，以兼容 BERTopic 的 CountVectorizer"""
    logger.info("正在执行多语言预分词处理...")
    jt = JanomeTokenizer()
    tokenized_texts = []
    
    for text, lang in zip(texts, langs):
        text = str(text).lower()
        if lang == 'zh':
            tokenized = " ".join(jieba.cut(text))
        elif lang == 'jp':
            tokenized = " ".join([token.surface for token in jt.tokenize(text)])
        else:
            tokenized = text # 英文可直接依赖 BERTopic 默认分词
        tokenized_texts.append(tokenized)
        
    return tokenized_texts

def load_stopwords(path):
    """加载停用词文件，返回集合"""
    if not path or not os.path.exists(path):
        return set()
    with open(path, 'r', encoding='utf-8') as f:
        return set([w.strip() for w in f if w.strip()])

def get_multilingual_stopwords():
    """整合中日英停用词与领域停用词，从文件加载中日停用词以保持一致"""
    en_stops = set(nltk.corpus.stopwords.words('english'))
    zh_stops = load_stopwords("data_preanalysis/Chinese/merged_stopwords.txt")
    jp_stops = load_stopwords("data_preanalysis/Japanese/stopwords-ja.txt")
    domain_stops = {
        # 中文
        '圣经', '耶稣', '上帝', '基督教', '基督', '修女', '神父', '教会', '教徒', '信仰', '宗教', '福音',
        # 日文
        'キリスト', '教会', '神', '聖書', 'クリスチャン', 'イエス', '教団', '宗教', 
        # 英文
        'christian', 'christians', 'church', 'jesus', 'bible', 'religion', 'god', 'pastor', 'faith'
    }
    
    return list(en_stops | zh_stops | jp_stops | domain_stops)


def visualize_umap_hdbscan(embeddings, langs, labels, outpath, title="UMAP + HDBSCAN"):
    """给定 embeddings + HDBSCAN labels，绘制 UMAP 投影并按集群+语言着色/标记"""
    # 如果输入已经是 2D（例如已预计算的 UMAP），则直接使用。
    if embeddings.shape[1] > 2:
        reducer = umap.UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine', random_state=42)
        embs_2d = reducer.fit_transform(embeddings)
    else:
        embs_2d = embeddings

    df_plot = pd.DataFrame({
        "x": embs_2d[:, 0],
        "y": embs_2d[:, 1],
        "lang": langs,
        "cluster": labels,
    })
    df_plot["cluster"] = df_plot["cluster"].astype(str)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=df_plot,
        x="x",
        y="y",
        hue="cluster",
        style="lang",
        palette="tab10",
        s=50,
        alpha=0.85,
        edgecolor="none",
    )
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def check_clustering_quality(df, model):
    logger.info("--- 聚类质量自我检查报告 ---")
    
    # 1. 噪声比例统计
    total_count = len(df)
    outlier_count = len(df[df['topic'] == -1])
    outlier_ratio = (outlier_count / total_count) * 100
    logger.info(f"噪声点占比: {outlier_ratio:.2f}% (建议保持在 10%-40% 之间)")
    
    if outlier_ratio > 50:
        logger.warning("警告：噪声占比过高！建议增加 UMAP n_neighbors 或减小 HDBSCAN min_cluster_size。")

    # 2. 语言分布一致性 (核心指标：你的三语是否真的合体了)
    lang_dist = df.groupby(['topic', 'lang']).size().unstack(fill_value=0)
    # 过滤掉噪声簇 -1，看有效话题
    valid_topics = lang_dist.drop(index=-1, errors='ignore')
    
    logger.info("\n前 5 个话题的语言构成 (行=话题, 列=语种计数):")
    logger.info("\n" + valid_topics.head(5).to_string())
    
    # 检查是否有“纯语种话题”
    pure_topics = valid_topics[(valid_topics > 0).sum(axis=1) == 1]
    if len(pure_topics) > 0:
        logger.info(f"发现 {len(pure_topics)} 个纯语种话题，跨语言对齐仍有提升空间。")
    else:
        logger.info("恭喜！所有话题均包含多种语言，跨语言语义对齐效果良好。")

    # 保存这份统计到 CSV
    lang_dist.to_csv(os.path.join(OUTPUT_DIR, 'data/cluster_quality_check.csv'))

# ==========================================
# 4. 组装并执行 BERTopic 管线
# ==========================================
def run_topic_modeling_pipeline(langs_to_analyze):
    # 1. 准备数据
    df = load_and_sample_data(langs_to_analyze)
    docs = df['text'].tolist()
    langs = df['lang'].tolist()
    
    # 执行预分词（避免 BERTopic 吞掉中日文）
    tokenized_docs = pre_tokenize(docs, langs)
    df['tokenized_text'] = tokenized_docs
    df.to_csv(os.path.join(OUTPUT_DIR, 'data/processed_corpus.csv'), index=False, encoding='utf-8-sig')
    logger.info("已保存预处理后的语料库。")

    # 2. Embedding 抽取
    logger.info("加载基座模型: intfloat/multilingual-e5-large-instruct")
    embedding_model = SentenceTransformer('intfloat/multilingual-e5-large-instruct')
    
    # 加入诱导指令以强化语义对齐
    # instruction = "Extract universal semantic patterns of religious hate speech, focusing on dehumanization, hypocrisy, or intellectual questioning."
    # prompts = [f"Instruct: {instruction}\nQuery: {text}" for text in docs]
    
    logger.info("正在生成 Embeddings (此过程可能较慢)...")
    embeddings = embedding_model.encode(tokenized_docs, show_progress_bar=True)
    np.save(os.path.join(OUTPUT_DIR, 'models/embeddings.npy'), embeddings)

    # 2.5. 生成 UMAP + HDBSCAN 可视化 (在拟合 BERTopic 之前)
    logger.info("生成 UMAP + HDBSCAN 可视化 (拟合前)...")
    umap_vis = umap.UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine', random_state=42)
    embs_2d = umap_vis.fit_transform(embeddings)
    hdbscan_vis = hdbscan.HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
    hdb_labels = hdbscan_vis.fit_predict(embs_2d)
    visualize_umap_hdbscan(
        embs_2d,
        langs,
        hdb_labels,
        os.path.join(OUTPUT_DIR, 'visualizations/umap_hdbscan_pre_bertopic.png'),
        title="UMAP + HDBSCAN (Pre-BERTopic)"
    )

    # 3. 降维模型 (UMAP)
    logger.info("配置 UMAP 模型...")
    umap_model = umap.UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)

    # 4. 聚类模型 (HDBSCAN)
    logger.info("配置 HDBSCAN 模型...")
    hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
    # logger.info(f"我们发现了{len(set(hdbscan_model.labels))}个簇")

    # 5. 向量化器 (CountVectorizer)
    logger.info("配置 CountVectorizer (加载自定义停用词)...")
    vectorizer_model = CountVectorizer(
        stop_words=get_multilingual_stopwords(),
        token_pattern=r"(?u)\b\w+\b" # 允许单字通过，交由停用词表过滤
    )

    # 6. 初始化并训练 BERTopic
    logger.info("初始化并拟合 BERTopic 模型...")
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        language="multilingual",
        calculate_probabilities=False,
        verbose=True
    )
    
    # 注意：传给 BERTopic 提取特征的是 tokenized_docs，但计算向量用的是原始 embeddings
    topics, probs = topic_model.fit_transform(tokenized_docs, embeddings)

    # ==========================================
    # 5. 结果保存与可视化导出
    # ==========================================
    logger.info("执行完毕，正在导出分析结果...")
    
    # A. 提取并保存 Topic 概览
    topic_info = topic_model.get_topic_info()
    topic_info.to_csv(os.path.join(OUTPUT_DIR, 'data/topic_info.csv'), index=False, encoding='utf-8-sig')
    
    # B. 保存文本与 Topic 的映射关系
    df['topic'] = topics
    df.to_csv(os.path.join(OUTPUT_DIR, 'data/document_topic_mapping.csv'), index=False, encoding='utf-8-sig')
    
    # C. 交互式可视化导出 (存为 HTML 以保留交互性)
    logger.info("正在生成可视化 HTML 图表...")
    try:
        # 词频条形图
        fig_barchart = topic_model.visualize_barchart(top_n_topics=10)
        fig_barchart.write_html(os.path.join(OUTPUT_DIR, 'visualizations/topic_barchart.html'))
        
        # 聚类散点图 (2D 映射)
        fig_docs = topic_model.visualize_documents(tokenized_docs, embeddings=embeddings, hide_document_hover=False)
        fig_docs.write_html(os.path.join(OUTPUT_DIR, 'visualizations/document_clusters.html'))
        
        # 主题距离热力图 (Hierarchical view)
        fig_heatmap = topic_model.visualize_heatmap()
        fig_heatmap.write_html(os.path.join(OUTPUT_DIR, 'visualizations/topic_heatmap.html'))
    except Exception as e:
        logger.error(f"可视化生成失败: {e}")

    # D. 序列化保存模型 (方便日后直接加载)
    # 注意：保存大模型可能会占用较多空间，这里我们保存轻量级的安全张量模型
    topic_model.save(os.path.join(OUTPUT_DIR, 'models/bertopic_model'), serialization="safetensors", save_ctfidf=True)
    
    logger.info(f"所有步骤已完成！结果保存在 {OUTPUT_DIR} 目录下。")
    return topic_model, df

# ==========================================
# 6. 启动器
# ==========================================
if __name__ == "__main__":
    # 在这里选择你要分析的语言。
    # 如果想跑单语言，就改成 ['en'] 或 ['zh']
    TARGET_LANGS = ['zh', 'en', 'jp'] 
    
    try:
        model, result_df = run_topic_modeling_pipeline(TARGET_LANGS)
        check_clustering_quality(result_df, model)
        logger.info("\n=== Top 5 Topics ===")
        logger.info(model.get_topic_info().head(6)) # head(6) 是因为第0行通常是 -1 (噪声)
    except Exception as e:
        logger.exception("Pipeline 运行过程中发生致命错误")