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
TOP_K_KEYWORDS = 1000  # 每种语言从语料中提取多少个高频词作为候选
NEAREST_NEIGHBORS = 15  # 每个种子词找多少个近邻
# ===========================================

print(f"正在加载模型: {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME, device=DEVICE)

jt = JanomeTokenizer()

def basic_tokenizer(text, lang):
    if lang == 'zh':
        return " ".join(jieba.cut(text))
    if lang == 'jp':
        # 使用 Janome 进行日语分词，比直接切单字效果好得多
        return " ".join([token.surface for token in jt.tokenize(text)])
    return text.lower()

def extract_candidates(file_path, lang):
    print(f"正在分析 {lang} 语料库关键词...")
    df = pd.read_csv(file_path)
    
    # 增加一个简单的清洗，去掉 URL 和标点，防止干扰 TF-IDF
    def clean_text(t):
        t = re.sub(r'http\S+|www\S+|https\S+', '', str(t), flags=re.MULTILINE)
        return t

    processed_text = df['text'].apply(clean_text).apply(lambda x: basic_tokenizer(x, lang))
    
    # 关键修改点：token_pattern 允许单字词，并增加 max_df 防止常见词霸榜
    vectorizer = TfidfVectorizer(
        max_features=TOP_K_KEYWORDS,
        token_pattern=r"(?u)\b\w+\b",  # 这里的 \w+ 允许匹配长度为 1 的词
        stop_words=None, 
        max_df=0.9,  # 过滤掉在 90% 文档里都出现的太常见的词
        min_df=2     # 至少在 2 篇文章里出现过
    )
    
    try:
        vectorizer.fit(processed_text)
    except ValueError as e:
        print(f"警告：{lang} 语料库处理失败，可能是有效词太少。错误信息：{e}")
        return []
        
    candidates = vectorizer.get_feature_names_out()
    return candidates

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
    
    output_df.to_csv('discovered_candidates.csv', index=False)
    print("\n[完成] 挖掘出的候选词已保存至 'discovered_candidates.csv'")
    print("建议人工检查相似度在 0.7 - 0.9 之间的词汇，这些通常是极佳的隐喻对齐点。")

if __name__ == "__main__":
    main()