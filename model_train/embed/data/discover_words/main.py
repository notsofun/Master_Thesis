# python model_train/embed/data/discover_words/main.py
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util
import torch
import jieba
import re
from tqdm import tqdm
from janome.tokenizer import Tokenizer as JanomeTokenizer
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

# ================= 配置区域 =================
# 1. 种子词典路径 (包含: 概念含义,中文隐喻/黑话,英文等价表达,日文等价表达,文化背景解释)
SEED_CSV_PATH = 'model_train/embed/data/seed_words.csv'

# 2. 三语语料库路径 (假设每个 CSV 只有一列 'text')
CORPUS_PATHS = {
    'zh': 'data_detect/finetuned_detection/chinese_final_religious_hate.csv',
    'en': 'data_collection/English_Existing/merged_deduped.csv',
    'jp': 'data_detect/finetuned_detection/japanese_final_religious_hate.csv'
}

# 3. 模型选择
MODEL_NAME = 'intfloat/multilingual-e5-large-instruct'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 4. 参数设置
NEAREST_NEIGHBORS = 10  # 每个种子词找多少个近邻
# ===========================================

print(f"正在加载模型: {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME, device=DEVICE)

jt = JanomeTokenizer()

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

def extract_candidates(file_path, lang, top_k=500):
    print(f"正在分析 {lang} 语料库关键词...")
    df = pd.read_csv(file_path)
    
    # 1. 清洗：去掉URL和特殊噪音
    def clean_text(t):
        t = re.sub(r'http\S+|www\S+|https\S+', '', str(t))
        # 过滤掉一些乱码或过短的干扰
        return t

    processed_text = df['text'].apply(clean_text).apply(lambda x: basic_tokenizer(x, lang))
    
    # 2. 配置停用词
    # 只有英文和部分语言有内置停用词表，中文通常需要自定义
    stop_words_list = None
    if lang == 'en':
        stop_words_list = 'english' # 使用 sklearn 内置的英文停用词表
    
    # 3. 配置 TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=top_k,
        # token_pattern 需要覆盖中英日。对于已经分好词（有空格）的文本，
        # 这个正则主要起到了“过滤掉纯标点符号”的作用
        token_pattern=r"(?u)\b\w+\b", 
        stop_words=stop_words_list,
        max_df=0.85,  # 降低阈值，如果一个词在85%的文档都出现，那它太泛滥了
        min_df=3,     # 提高门槛，至少在3篇文档出现，过滤掉偶发性的拼写错误
        ngram_range=(1, 2), # 额外福利：允许提取“双词短语”（如 Snake oil），而不仅仅是单字
        use_idf=True,
        smooth_idf=True
    )
    
    try:
        tfidf_matrix = vectorizer.fit_transform(processed_text)
        # 获取特征名（候选词）
        candidates = vectorizer.get_feature_names_out()
        return list(candidates)
    except ValueError as e:
        print(f"警告：{lang} 语料库有效词不足。错误：{e}")
        return []

def find_neighbors(lang, seeds, corpus_candidates):
    """步骤 2: 词向量探测 (Nearest Neighbors)"""
    print(f"正在为 {lang} 种子词探测近邻...")
    
    # E5 模型要求前缀
    seed_queries = [f"query: {s}" for s in seeds]
    candidate_passages = [f"passage: {c}" for c in corpus_candidates]
    
    # 计算 Embedding
    seed_embeddings = model.encode(seed_queries, convert_to_tensor=True)
    candidate_embeddings = model.encode(candidate_passages, convert_to_tensor=True)
    
    # 计算余弦相似度矩阵 (Seeds x Candidates)
    cos_scores = util.cos_sim(seed_embeddings, candidate_embeddings)
    
    results = []
    for i, seed in enumerate(seeds):
        # 找出相似度最高的前 K 个候选词
        top_results = torch.topk(cos_scores[i], k=min(NEAREST_NEIGHBORS, len(candidate_embeddings)))
        
        for score, idx in zip(top_results.values, top_results.indices):
            results.append({
                'seed_word': seed,
                'candidate_word': corpus_candidates[idx],
                'similarity': score.item(),
                'language': lang
            })
    return results

def main():
    # 1. 加载种子词
    seed_df = pd.read_csv(SEED_CSV_PATH)
    
    all_discovery_results = []

    for lang in ['zh', 'en', 'jp']:
        # 获取该语种的种子列
        col_map = {'zh': '中文隐喻/黑话', 'en': '英文等价表达', 'jp': '日文等价表达'}
        current_seeds = seed_df[col_map[lang]].dropna().unique().tolist()
        
        # 提取语料候选词
        candidates = extract_candidates(CORPUS_PATHS[lang], lang)
        
        # 挖掘近邻
        neighbors = find_neighbors(lang, current_seeds, candidates)
        all_discovery_results.extend(neighbors)

    # 3. 输出结果
    output_df = pd.DataFrame(all_discovery_results)
    # 按相似度排序
    output_df = output_df.sort_values(by='similarity', ascending=False)
    
    output_df.to_csv('model_train/embed/data/discovered_candidates.csv', index=False)
    print("\n[完成] 挖掘出的候选词已保存至 'discovered_candidates.csv'")
    print("建议人工检查相似度在 0.7 - 0.9 之间的词汇，这些通常是极佳的隐喻对齐点。")

if __name__ == "__main__":
    main()