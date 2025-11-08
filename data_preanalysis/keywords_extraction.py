#!/usr/bin/env python3
# cooccurrence_keywords.py
# Python 3.8+
import os
import re
import math
import argparse
from collections import Counter
from nltk.tokenize import TweetTokenizer
import pandas as pd

tweet_tok = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)
try:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    SKLEARN_STOPWORDS = set(ENGLISH_STOP_WORDS)
except Exception:
    SKLEARN_STOPWORDS = set()  # fallback, still OK

# -----------------------
# 参数与配置（可调整）
# -----------------------
WINDOW_SIZE = 4           # 滑动窗口大小（token数）用于窗口共现
MIN_TERM_FREQ = 5         # 最低频次阈值（词/短语）用于过滤噪声
MAX_NGRAM = 3             # ngram 最大长度(1=单词, 2=二短语, 3=三短语)
TOP_K_TERMS = 300         # 导出多少 top 单词/短语供人工查看
TARGET_TERMS = ["christian", "christians", "church", "jesus", "bible", "priest"]  # 可自定义关注词

# -----------------------
# 文本处理与分词函数
# -----------------------
URL_RE = re.compile(r'https?://\S+|www\.\S+')
EMAIL_RE = re.compile(r'\S+@\S+')
NONWORD_RE = re.compile(r"[^A-Za-z0-9'\- ]+")  # 保留字母、数字、撇号和连字符
MULTI_SPACE = re.compile(r'\s+')

def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.replace('\u200b', ' ')
    s = URL_RE.sub(' ', s)
    s = EMAIL_RE.sub(' ', s)
    s = s.replace('\n', ' ')
    s = s.lower()
    s = NONWORD_RE.sub(' ', s)
    s = MULTI_SPACE.sub(' ', s).strip()
    return s

WORD_TOKEN_RE = re.compile(r"[A-Za-z0-9'\-]+")

def tokenize(text: str):
    return tweet_tok.tokenize(text)

# -----------------------
# 共现与 PMI 计算
# -----------------------
def window_cooccurrence(docs_tokens, window_size=WINDOW_SIZE):
    term_count = Counter()
    pair_count = Counter()
    total_windows = 0

    for tokens in docs_tokens:
        n = len(tokens)
        # increment term counts
        term_count.update(tokens)
        # sliding window
        for i in range(n):
            window_tokens = tokens[i : i + window_size]
            if len(window_tokens) < 2:
                continue
            total_windows += 1
            # count all unordered pairs within this window (unique pairs per window)
            unique = set(window_tokens)
            for a in unique:
                for b in unique:
                    if a < b:
                        pair_count[(a, b)] += 1
    return term_count, pair_count, total_windows

def pmi(pair_count, term_count, total_windows, eps=1e-9):
    """Compute PMI (symmetric) for each pair; return dict of pair -> PMI"""
    pmi_dict = {}
    for (a, b), co in pair_count.items():
        pa = term_count[a] / total_windows
        pb = term_count[b] / total_windows
        pab = co / total_windows
        # PMI = log2(pab / (pa*pb))
        if pa * pb <= 0 or pab <= 0:
            continue
        pmi_val = math.log2((pab + eps) / (pa * pb + eps))
        pmi_dict[(a, b)] = pmi_val
    return pmi_dict

def npmi(pair_count, term_count, total_windows, eps=1e-9):
    npmi_dict = {}
    for (a, b), co in pair_count.items():
        pa = term_count[a] / total_windows
        pb = term_count[b] / total_windows
        pab = co / total_windows
        if pa * pb <= 0 or pab <= 0:
            continue
        pmi_val = math.log2((pab + eps) / (pa * pb + eps))
        npmi_val = pmi_val / (-math.log2(pab + eps))
        npmi_dict[(a, b)] = npmi_val
    return npmi_dict


def t_score(pair_count, term_count, total_windows):
    t_dict = {}
    for (a, b), co in pair_count.items():
        expected = (term_count[a] * term_count[b]) / total_windows
        if co > 0:
            t_dict[(a, b)] = (co - expected) / math.sqrt(co)
    return t_dict

def jaccard(pair_count, term_count):
    jaccard_dict = {}
    for (a, b), co in pair_count.items():
        denom = term_count[a] + term_count[b] - co
        if denom > 0:
            jaccard_dict[(a, b)] = co / denom
    return jaccard_dict


def log_likelihood(pair_count, term_count, total_windows, eps=1e-9):
    llr_dict = {}
    for (a, b), co in pair_count.items():
        k11 = co
        k12 = term_count[a] - co
        k21 = term_count[b] - co
        k22 = total_windows - (k11 + k12 + k21)
        # expected counts
        row_sum = [k11 + k12, k21 + k22]
        col_sum = [k11 + k21, k12 + k22]
        total = total_windows
        def safe_log(x): return math.log(x + eps)
        E = [[r * c / total for c in col_sum] for r in row_sum]
        G2 = 0
        for i, row in enumerate([k11, k12]):
            for j, obs in enumerate([k21, k22]):
                exp = E[i][j]
                if exp > 0 and obs > 0:
                    G2 += 2 * obs * safe_log(obs / exp)
        llr_dict[(a, b)] = G2
    return llr_dict


# -----------------------
# n-gram 统计（基于文档级频率）
# -----------------------
def build_ngrams(tokens_list, max_n=MAX_NGRAM):
    ngram_counter = Counter()
    for tokens in tokens_list:
        L = len(tokens)
        for n in range(1, max_n + 1):
            for i in range(0, L - n + 1):
                ngram = tuple(tokens[i:i+n])
                # naive filter: 不能全为停用词与数字
                if all(tok in SKLEARN_STOPWORDS or tok.isdigit() for tok in ngram):
                    continue
                ngram_counter[' '.join(ngram)] += 1
    return ngram_counter

# -----------------------
# 主流程
# -----------------------
def analyze(input_file: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    print("[INFO] 开始分析，输出目录：", output_dir)

    # 读取 CSV
    print("[INFO] 读取 CSV 文件：", input_file)
    df = None
    tried = []
    for enc in ["utf-8", "utf-8-sig", "latin1"]:
        for sep in [",", "\t", ";"]:
            try:
                tmp = pd.read_csv(input_file, sep=sep, encoding=enc, engine='python', on_bad_lines='skip')
                if tmp.shape[1] >= 2:
                    df = tmp
                    tried.append((enc, sep))
                    break
            except Exception:
                continue
        if df is not None:
            break
    if df is None:
        raise RuntimeError(f"无法读取文件 {input_file}（尝试了：{tried}）")
    print(f"[INFO] 成功读取 {len(df)} 条原始数据（尝试过编码和分隔符：{tried}）")

    # 取文本列
    cols = df.columns.tolist()
    text_col = cols[1]
    df = df[[cols[0], text_col]].rename(columns={cols[0]: 'post_id', text_col: 'text'})
    df['text'] = df['text'].astype(str).fillna('').str.strip()
    df = df[df['text'] != '']
    print(f"[INFO] 有效文本条数：{len(df)}")

    # 清洗与分词
    print("[INFO] 文本清洗中...")
    docs_clean = df['text'].map(clean_text).tolist()
    print("[INFO] 分词中...")
    docs_tokens = [tokenize(t) for t in docs_clean]

    # n-gram 统计
    print(f"[INFO] 统计 n-gram（最大长度 {MAX_NGRAM}）...")
    ngram_counter = build_ngrams(docs_tokens, max_n=MAX_NGRAM)
    ngram_counter = Counter({k:v for k,v in ngram_counter.items() if v >= MIN_TERM_FREQ})
    print(f"[INFO] n-gram 统计完成，候选 n-gram 数量：{len(ngram_counter)}")

    # token-level 共现
    print(f"[INFO] 计算 token-level 窗口共现（窗口大小 {WINDOW_SIZE}）...")
    token_docs = [[tok for tok in toks if not (tok in SKLEARN_STOPWORDS or tok.isdigit())] for toks in docs_tokens]
    term_count, pair_count, total_windows = window_cooccurrence(token_docs, window_size=WINDOW_SIZE)
    term_count = Counter({k:v for k,v in term_count.items() if v >= MIN_TERM_FREQ})
    print(f"[INFO] token-level 统计完成，term 数量：{len(term_count)}, pair 数量：{len(pair_count)}")

    # PMI / NPMI / t-score / jaccard / LLR
    print("[INFO] 计算 PMI...")
    pmi_dict = pmi(pair_count, term_count, total_windows)
    print("[INFO] 计算 NPMI...")
    npmi_dict = npmi(pair_count, term_count, total_windows)
    print("[INFO] 计算 t-score...")
    t_dict = t_score(pair_count, term_count, total_windows)
    print("[INFO] 计算 Jaccard...")
    jaccard_dict = jaccard(pair_count, term_count)
    print("[INFO] 计算 LLR...")
    llr_dict = log_likelihood(pair_count, term_count, total_windows)

    # 输出 top 单词 / ngram
    print(f"[INFO] 输出 top {TOP_K_TERMS} 单词...")
    top_terms = term_count.most_common(TOP_K_TERMS)
    pd.DataFrame(top_terms, columns=['term', 'freq']).to_csv(os.path.join(output_dir, 'top_terms_freq.csv'), index=False)
    print(f"[INFO] 输出 top {TOP_K_TERMS} n-gram...")
    top_ngrams = ngram_counter.most_common(TOP_K_TERMS)
    pd.DataFrame(top_ngrams, columns=['ngram', 'freq']).to_csv(os.path.join(output_dir, 'top_ngrams_freq.csv'), index=False)

    # 输出 top pair
    print("[INFO] 输出 top 共现对（按次数和 PMI）...")
    top_pairs_by_count = sorted(pair_count.items(), key=lambda x: x[1], reverse=True)[:1000]
    pd.DataFrame([(a,b,c) for (a,b),c in top_pairs_by_count], columns=['a','b','co_count']).to_csv(os.path.join(output_dir,'top_pairs_count.csv'), index=False)
    top_pairs_by_pmi = sorted(pmi_dict.items(), key=lambda x: x[1], reverse=True)[:1000]
    pd.DataFrame([(a,b,c) for (a,b),c in top_pairs_by_pmi], columns=['a','b','pmi']).to_csv(os.path.join(output_dir,'top_pairs_pmi.csv'), index=False)

    # 与目标词关联
    print("[INFO] 输出与目标词强关联词...")
    assoc_rows = []
    target_lower = [t.lower() for t in TARGET_TERMS]
    for (a,b),c in pair_count.items():
        if a in target_lower or b in target_lower:
            assoc_rows.append((a,b,c, pmi_dict.get((a,b), None)))
    if assoc_rows:
        pd.DataFrame(assoc_rows, columns=['a','b','co_count','pmi']).sort_values(['co_count','pmi'], ascending=[False, False]).to_csv(os.path.join(output_dir,'assoc_with_targets.csv'), index=False)

    # 各指标 CSV
    print("[INFO] 输出各指标 CSV...")
    def dict_to_df(d, name):
        df = pd.DataFrame([(a,b,v) for (a,b),v in d.items()], columns=['a','b',name])
        df.sort_values(by=name, ascending=False, inplace=True)
        df.to_csv(os.path.join(output_dir, f'top_pairs_{name}.csv'), index=False)

    for d,name in [(pmi_dict,'pmi'), (npmi_dict,'npmi'), (t_dict,'t'), (jaccard_dict,'jaccard'), (llr_dict,'llr')]:
        dict_to_df(d, name)
    print("[INFO] 指标 CSV 输出完成")

    # 综合候选关键词
    print("[INFO] 生成综合候选关键词...")
    # -----------------------
    # 构建词 -> 对索引，加速指标计算
    # -----------------------
    from collections import defaultdict

    # 候选关键词（unigram + ngram + 与目标词强关联）
    candidate_keywords = set([t for t,_ in top_terms]) | set([g for g,_ in top_ngrams])
    for (a,b),co_count in pair_count.items():
        if a in target_lower or b in target_lower:
            candidate_keywords.add(a)
            candidate_keywords.add(b)

    # 建立词 -> pair 索引
    kw2pairs = defaultdict(list)
    for (a,b) in pair_count:
        kw2pairs[a].append((a,b))
        kw2pairs[b].append((a,b))

    # -----------------------
    # 计算综合评分
    # -----------------------
    max_term_freq = max(term_count.values()) if term_count else 1
    candidate_scores = {}

    for kw in candidate_keywords:
        freq_score = term_count.get(kw, 0) / max_term_freq

        pairs = kw2pairs.get(kw, [])
        count_pairs = max(1, len(pairs))  # 避免除零

        # 平均指标
        assoc_score = sum(pmi_dict.get(p,0) for p in pairs) / count_pairs
        t_score_val = sum(t_dict.get(p,0) for p in pairs) / count_pairs
        llr_val = sum(llr_dict.get(p,0) for p in pairs) / count_pairs

        # 综合评分，可调权重
        candidate_scores[kw] = 0.4*freq_score + 0.3*assoc_score + 0.15*t_score_val + 0.15*llr_val

    # 排序取 top K
    sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
    top_candidates = [kw for kw,_ in sorted_candidates[:TOP_K_TERMS]]

    # 导出候选关键词
    pd.DataFrame(top_candidates, columns=['candidate']).to_csv(
        os.path.join(output_dir,'candidate_keywords.txt'), index=False
    )

def load_pair_dicts_from_csv(output_dir):
    """
    从之前输出的 CSV 文件读取指标，返回 dicts。
    """
    dicts = {}
    for name in ['pmi','npmi','t','jaccard','llr']:
        path = os.path.join(output_dir, f'top_pairs_{name}.csv')
        if not os.path.exists(path):
            print(f"[WARN] 文件不存在: {path}, 将返回空字典")
            dicts[name] = {}
            continue
        print(f"[INFO] 读取指标文件: {path}")
        df = pd.read_csv(path)
        dicts[name] = { (row['a'], row['b']): row[name] for _, row in df.iterrows() }
        print(f"[INFO] {name} 指标加载完成，共 {len(dicts[name])} 条记录")
    return dicts

def load_term_pair_counts(output_dir):
    # term_count
    term_count_path = os.path.join(output_dir, 'top_terms_freq.csv')
    print(f"[INFO] 读取 term_count 文件: {term_count_path}")
    term_count_df = pd.read_csv(term_count_path)
    term_count = { row['term']: row['freq'] for _, row in term_count_df.iterrows() }
    print(f"[INFO] term_count 加载完成，共 {len(term_count)} 个词")

    # pair_count
    pair_count_path = os.path.join(output_dir, 'top_pairs_count.csv')
    print(f"[INFO] 读取 pair_count 文件: {pair_count_path}")
    pair_count_df = pd.read_csv(pair_count_path)
    pair_count = { (row['a'], row['b']): row['co_count'] for _, row in pair_count_df.iterrows() }
    print(f"[INFO] pair_count 加载完成，共 {len(pair_count)} 对词对")

    return term_count, pair_count

def extract_candidate_keywords_with_log(term_count, pair_count, pmi_dict, t_dict, llr_dict, target_terms, output_dir, top_k=300):
    print("[INFO] 开始计算候选关键词综合评分...")

    candidate_keywords = set(term_count.keys())
    candidate_keywords.update({a for (a,b) in pair_count if a in target_terms})
    candidate_keywords.update({b for (a,b) in pair_count if b in target_terms})
    print(f"[INFO] 候选关键词总数: {len(candidate_keywords)}")

    max_term_freq = max(term_count.values()) if term_count else 1
    candidate_scores = {}
    for i, kw in enumerate(candidate_keywords, 1):
        if i % 100 == 0:
            print(f"[INFO] 已处理 {i}/{len(candidate_keywords)} 个候选词...")
        # 词频归一化
        freq_score = term_count.get(kw, 0) / max_term_freq

        # 与目标词平均 PMI
        assoc_pairs = [(a,b) for (a,b) in pair_count if kw in (a,b)]
        assoc_score = sum(pmi_dict.get((a,b),0) for (a,b) in assoc_pairs) / max(1,len(assoc_pairs))

        # 平均 t-score
        t_score_val = sum(t_dict.get((a,b),0) for (a,b) in assoc_pairs) / max(1,len(assoc_pairs))

        # 平均 LLR
        llr_val = sum(llr_dict.get((a,b),0) for (a,b) in assoc_pairs) / max(1,len(assoc_pairs))

        # 综合评分（可调整权重）
        candidate_scores[kw] = 0.4*freq_score + 0.3*assoc_score + 0.15*t_score_val + 0.15*llr_val

    print("[INFO] 综合评分计算完成，开始排序...")
    sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
    top_candidates = [kw for kw,_ in sorted_candidates[:top_k]]

    # 导出
    pd.DataFrame(top_candidates, columns=['candidate']).to_csv('candidate_keywords.txt', index=False)
    print(f"[INFO] 候选关键词提取完成，输出到 {'candidate_keywords.txt'}")


# -----------------------
# CLI
# -----------------------
if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Co-occurrence based keyword/phrase extraction")
    # parser.add_argument("--input", "-i", default="merged_deduped.csv", help="输入 CSV 文件（第二列为文本）")
    # parser.add_argument("--out", "-o", default="cooc_output", help="结果输出目录")
    # parser.add_argument("--min_freq", type=int, default=MIN_TERM_FREQ, help="最小频次阈值")
    # parser.add_argument("--window", type=int, default=WINDOW_SIZE, help="滑动窗口大小")
    # args = parser.parse_args()

    # MIN_TERM_FREQ = args.min_freq
    # WINDOW_SIZE = args.window

    # analyze(args.input, args.out)
    output_dir = "cooc_output"
    print("[INFO] 开始从 CSV 加载指标和计数...")
    term_count, pair_count = load_term_pair_counts(output_dir)
    dicts = load_pair_dicts_from_csv(output_dir)
    pmi_dict = dicts['pmi']
    t_dict = dicts['t']
    llr_dict = dicts['llr']
    print("[INFO] 指标加载完成，开始提取候选关键词...")

    extract_candidate_keywords_with_log(
        term_count=term_count,
        pair_count=pair_count,
        pmi_dict=pmi_dict,
        t_dict=t_dict,
        llr_dict=llr_dict,
        target_terms=[t.lower() for t in TARGET_TERMS],
        output_dir=output_dir,
        top_k=TOP_K_TERMS
    )
    print("[INFO] 候选关键词提取流程全部完成")
