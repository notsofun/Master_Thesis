# cooccurrence_keywords_chinese.py
import os
import re
import math
import json
import jieba
import random
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx
from gensim.models.phrases import Phrases, Phraser

# ---------- 用户可配置项 ----------
CSV_PATH = "../../data_collection/Tieba/all_search_posts.csv"   # <-- 修改为你的 CSV 路径
TEXT_COL = "main_content"
MIN_FREQ = 5             # 词/短语至少出现多少次（可调）
WINDOW_SIZE = 5          # 共现窗口
TOP_N = 300              # 输出前 N 个候选关键词
CUSTOM_DICT = "custom_dict.txt"  # 可选：把你的种子词写进去，每行一个词，jieba 加载
STOPWORDS_PATH = "merged_stopwords.txt"  # 标准停用词（需准备）
# ------------------------------------

# 若无 pkuseg 环境，使用 jieba；你也可以替换分词器
def load_texts(csv_path, text_col):
    df = pd.read_csv(csv_path)
    texts = df[text_col].fillna("").astype(str).tolist()
    return texts, df

def clean_text(s):
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'http[s]?://\S+', ' ', s)
    s = re.sub(r'[^\u4e00-\u9fffA-Za-z0-9\s\-]', ' ', s)
    s = re.sub(r'\-{2,}', '-', s)
    return s.strip()

def load_stopwords(path):
    if not os.path.exists(path):
        return set()
    with open(path, 'r', encoding='utf-8') as f:
        return set([w.strip() for w in f if w.strip()])

# 可将你给的那些词放进 custom_dict.txt（每行一个）
def setup_jieba(custom_dict=None):
    if custom_dict and os.path.exists(custom_dict):
        jieba.load_userdict(custom_dict)

def tokenize(texts, stopwords):
    tokenized = []
    for s in texts:
        s2 = clean_text(s)
        segs = [w for w in jieba.lcut(s2) if w.strip() and w not in stopwords]
        tokenized.append(segs)
    return tokenized

# 用 gensim Phrases 自动抽取常见短语（二/三元组）
def detect_phrases(tokenized, min_count=5, threshold=10.0):
    phrases = Phrases(tokenized, min_count=min_count, threshold=threshold, delimiter='_')
    ph = Phraser(phrases)
    tokenized_ph = [ph[s] for s in tokenized]
    # 将 b'_' 转换成 '_' 的字符串
    tokenized_ph = [[t if isinstance(t, str) else t.decode('utf-8') for t in sent] for sent in tokenized_ph]

    return tokenized_ph, ph

def build_vocab(tokenized):
    freq = Counter()
    doc_freq = Counter()
    for sent in tokenized:
        freq.update(sent)
        doc_freq.update(set(sent))
    return freq, doc_freq

def build_cooccurrence(tokenized, window=5):
    pair_counts = Counter()
    total_windows = 0
    for sent in tokenized:
        n = len(sent)
        for i in range(n):
            wi = sent[i]
            j0 = i+1
            j1 = min(n, i+window)
            for j in range(j0, j1):
                wj = sent[j]
                if wi == wj: continue
                key = tuple(sorted([wi, wj]))
                pair_counts[key] += 1
    return pair_counts

def pmi_score(pair_counts, freq, total_tokens):
    pmi = {}
    for pair, co in pair_counts.items():
        w1, w2 = pair
        p_w1 = freq[w1] / total_tokens
        p_w2 = freq[w2] / total_tokens
        p_w1w2 = co / total_tokens  # approximate
        if p_w1w2 == 0 or p_w1 == 0 or p_w2 == 0:
            continue
        score = math.log(p_w1w2 / (p_w1 * p_w2) + 1e-12)
        pmi[pair] = score
    return pmi

def centrality_scores_from_graph(nodes, edges):
    G = nx.Graph()
    G.add_nodes_from(nodes)
    for (a,b), w in edges.items():
        G.add_edge(a,b,weight=w)
    deg = nx.degree_centrality(G)
    bet = nx.betweenness_centrality(G, weight='weight')
    eig = nx.pagerank(G, weight='weight')
    return deg, bet, eig, G

def compose_candidate_table(freq, doc_freq, pair_counts, pmi_scores, tfidf_scores, deg, bet, eig, tokenized_texts, top_n=300, min_freq=5):
    rows = []
    total_tokens = sum(freq.values())
    # 单词 / 短语候选（频数过滤）
    candidates = [w for w,c in freq.items() if c >= min_freq]
    # 向量化的 TF-IDF 可能包含多字短语（如果用 custom tokenizer）
    for w in candidates:
        row = {
            "term": w,
            "freq": freq[w],
            "doc_freq": doc_freq[w],
            "pmi_max": 0.0,
            "pmi_partner": "",
            "tfidf": tfidf_scores.get(w, 0.0),
            "degree": deg.get(w, 0.0),
            "betweenness": bet.get(w, 0.0),
            "eigenvector": eig.get(w, 0.0),
            "examples": []
        }
        # pmi_max 与 partner
        related = [(pair,score) for pair,score in pmi_scores.items() if w in pair]
        if related:
            pair, score = max(related, key=lambda x: x[1])
            row["pmi_max"] = score
            row["pmi_partner"] = pair[0] if pair[1]==w else pair[1]
        rows.append(row)
    df = pd.DataFrame(rows)
    # 归一化并计算综合分
    for col in ["freq","pmi_max","tfidf","degree","betweenness","eigenvector"]:
        if df[col].max() > 0:
            df[col+"_norm"] = df[col] / (df[col].max())
        else:
            df[col+"_norm"] = 0.0
    # 综合分：按权重组合（可调）
    df["score"] = (0.35*df["pmi_max_norm"] + 0.3*df["tfidf_norm"] + 0.2*df["degree_norm"] + 0.15*df["freq_norm"])
    # 提取上下文例句（最多 3 条）
    for i, row in df.iterrows():
        t = row["term"]
        exs = []
        for sent in tokenized_texts:
            if t in sent:
                exs.append("".join(sent))
            if len(exs) >= 3:
                break
        df.at[i,"examples"] = " || ".join(exs)
    df = df.sort_values("score", ascending=False).head(top_n)
    return df

def compute_tfidf_inverse(tokenized_texts):
    # 把 tokenized_texts 转回空格分隔字符串供 TfidfVectorizer 使用
    docs = [" ".join(s) for s in tokenized_texts]
    vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b', max_features=20000)
    X = vectorizer.fit_transform(docs)
    # 取出词表和 idf
    idf = dict(zip(vectorizer.get_feature_names_out(), vectorizer.idf_))
    return idf

def main():
    # 1. 加载
    texts, df_raw = load_texts(CSV_PATH, TEXT_COL)
    stopwords = load_stopwords(STOPWORDS_PATH)
    setup_jieba(CUSTOM_DICT)

    # 2. 分词
    tokenized = tokenize(texts, stopwords)

    # 3. 发现短语（可保留或替换 tokenized）
    tokenized_ph, ph = detect_phrases(tokenized, min_count=3, threshold=5.0)

    # 4. 词表与共现
    freq, doc_freq = build_vocab(tokenized_ph)
    pair_counts = build_cooccurrence(tokenized_ph, window=WINDOW_SIZE)
    total_tokens = sum(freq.values())

    # 5. PMI，TF-IDF
    pmi_scores = pmi_score(pair_counts, freq, total_tokens)
    tfidf_idf = compute_tfidf_inverse(tokenized_ph)

    # 6. 构建图并计算中心性
    # 只用 top M 共现作边以保持图不太稠密
    top_pairs = {p:c for p,c in pair_counts.items() if c >= 2}
    deg, bet, eig, G = centrality_scores_from_graph(list(freq.keys()), top_pairs)

    # 7. 候选表
    df_candidates = compose_candidate_table(freq, doc_freq, pair_counts, pmi_scores, tfidf_idf, deg, bet, eig, tokenized_ph, top_n=TOP_N, min_freq=MIN_FREQ)

    # 8. 输出
    os.makedirs("output", exist_ok=True)
    df_candidates.to_csv("output/keywords_candidates.csv", index=False, encoding='utf-8-sig')
    nx.write_gexf(G, "output/cooccurrence_network.gexf")
    print("Saved: output/keywords_candidates.csv and output/cooccurrence_network.gexf")

if __name__ == "__main__":
    main()
