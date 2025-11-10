#!/usr/bin/env python3
# cooccurrence_keywords.py
# Python 3.8+
import os
import re,datetime
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
TARGET_TERMS = [
	# Theological core
	"christian", "christians", "christ", "church", "jesus", "bible", "god", "holy",
	"catholic", "catholics", "pope", "teach", "teaching", "teachings", "belief", "believe", "true",
	# Institutional / state-related
	"authority", "state", "institutional", "separation", "change", "history", "right",
	
	# Cultural / emotional
	"love", "think", "women", "white", "world", "people", "time", "way", "needs", "like"
]

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

    # 读取 CSV，尽量容错：尝试多种编码与分隔符
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

    # 只取第二列（文本列）与可选的 post_id 列（第一列）
    cols = df.columns.tolist()
    text_col = cols[1]  # 按约定“第二列是文本”
    df = df[[cols[0], text_col]].rename(columns={cols[0]: 'post_id', text_col: 'text'})
    df['text'] = df['text'].astype(str).fillna('').str.strip()
    df = df[df['text'] != '']
    print(f"读取 {len(df)} 条有效文本（从 {input_file}）")

    # 清洗与分词
    docs_clean = df['text'].map(clean_text).tolist()
    docs_tokens = [tokenize(t) for t in docs_clean]

    # 统计 n-gram（文档级）
    ngram_counter = build_ngrams(docs_tokens, max_n=MAX_NGRAM)
    # 过滤低频
    ngram_counter = Counter({k:v for k,v in ngram_counter.items() if v >= MIN_TERM_FREQ})

    # token-level (单词) 以及窗口共现
    token_docs = [[tok for tok in toks if not (tok in SKLEARN_STOPWORDS or tok.isdigit())] for toks in docs_tokens]
    term_count, pair_count, total_windows = window_cooccurrence(token_docs, window_size=WINDOW_SIZE)

    # 过滤低频 term
    term_count = Counter({k:v for k,v in term_count.items() if v >= MIN_TERM_FREQ})

    # PMI
    pmi_dict = pmi(pair_count, term_count, total_windows)
    npmi_dict = npmi(pair_count, term_count, total_windows)
    t_dict = t_score(pair_count, term_count, total_windows)
    jaccard_dict = jaccard(pair_count, term_count)
    llr_dict = log_likelihood(pair_count, term_count, total_windows)

    # 输出：top 单词、top ngrams
    top_terms = term_count.most_common(TOP_K_TERMS)
    pd.DataFrame(top_terms, columns=['term', 'freq']).to_csv(os.path.join(output_dir, 'top_terms_freq.csv'), index=False)

    top_ngrams = ngram_counter.most_common(TOP_K_TERMS)
    pd.DataFrame(top_ngrams, columns=['ngram', 'freq']).to_csv(os.path.join(output_dir, 'top_ngrams_freq.csv'), index=False)

    # 输出：top 共现对（按共现次数与 PMI 排序）
    top_pairs_by_count = sorted(pair_count.items(), key=lambda x: x[1], reverse=True)[:1000]
    pd.DataFrame([(a,b,c) for (a,b),c in top_pairs_by_count], columns=['a','b','co_count']).to_csv(os.path.join(output_dir,'top_pairs_count.csv'), index=False)

    top_pairs_by_pmi = sorted(pmi_dict.items(), key=lambda x: x[1], reverse=True)[:1000]
    pd.DataFrame([(a,b,c) for (a,b),c in top_pairs_by_pmi], columns=['a','b','pmi']).to_csv(os.path.join(output_dir,'top_pairs_pmi.csv'), index=False)


    def dict_to_df(d, name):
        df = pd.DataFrame([(a,b,v) for (a,b),v in d.items()], columns=['a','b',name])
        df.sort_values(by=name, ascending=False, inplace=True)
        df.to_csv(os.path.join(output_dir, f'top_pairs_{name}.csv'), index=False)

    dict_to_df(pmi_dict, 'pmi')
    dict_to_df(npmi_dict, 'npmi')
    dict_to_df(t_dict, 't')
    dict_to_df(jaccard_dict, 'jaccard')
    dict_to_df(llr_dict, 'llr')

    print("分析完成，结果输出到目录：", 'candidate_keywords')

    print("开始抽取候选词")

    # 与目标词强关联（按共现数 + PMI + T-score）
    assoc_rows = []
    target_lower = [t.lower() for t in TARGET_TERMS]
    for (a,b), co_count in pair_count.items():
        if a in target_lower or b in target_lower:
            assoc_rows.append((a, b, co_count,
                            pmi_dict.get((a,b), None),
                            t_dict.get((a,b), None)))  # 新增 T-score

    id_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if assoc_rows:
        df_assoc = pd.DataFrame(assoc_rows, columns=['a','b','co_count','pmi','t_score'])
        df_assoc.sort_values(['co_count','pmi'], ascending=[False, False])\
                .to_csv(os.path.join('candidate_keywords', f'candidate_keywords_pmi_{id_str}.txt'), index=False)
        print("已保存PMI关键词")
        df_assoc1 = pd.DataFrame(assoc_rows, columns=['a','b','co_count','pmi','t_score'])
        df_assoc1.sort_values(['co_count','t_score'], ascending=[False, False])\
                .to_csv(os.path.join('candidate_keywords', f'candidate_keywords_tscore_{id_str}.txt'), index=False)
        print("已保存T-Score关键词")

# -----------------------
# CLI
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Co-occurrence based keyword/phrase extraction")
    parser.add_argument("--input", "-i", default="merged_deduped.csv", help="输入 CSV 文件（第二列为文本）")
    parser.add_argument("--out", "-o", default="cooc_output", help="结果输出目录")
    parser.add_argument("--min_freq", type=int, default=MIN_TERM_FREQ, help="最小频次阈值")
    parser.add_argument("--window", type=int, default=WINDOW_SIZE, help="滑动窗口大小")
    args = parser.parse_args()

    MIN_TERM_FREQ = args.min_freq
    WINDOW_SIZE = args.window

    analyze(args.input, args.out)
