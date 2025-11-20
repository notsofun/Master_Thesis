# find_high_pmi_tscore.py
import os, json
import re
import math
import jieba
import pandas as pd
from collections import Counter, defaultdict
from tqdm import tqdm

# ------------------ 配置 ------------------
CSV_PATH = "../../data_collection/Tieba/all_search_posts.csv"  # <- 改为你的文件路径
TEXT_COL = "main_content"
CUSTOM_DICT = "custom_dict.txt"     # 可选，自定义词典（每行一个词）
STOPWORDS_PATH = "../../data_preanalysis/Chinese/merged_stopwords.txt"  # 可选停用词
OUTPUT_DIR = "output"
MIN_CO = 3          # 最小共现文档数阈值（避免虚高PMI，建议 3-5）
TOPK_PER_CORE = 50  # 每个 core 输出 topK

# 读取核心词列表
with open("../../data_collection/fianl_keywors.json", "r", encoding="utf-8") as f:
    kw_json = json.load(f)

core_terms = kw_json["Chinese"]   # 例如 ["基督徒", "基督", "教会", ...]

# ------------------------------------------

def clean_text(s):
    if not isinstance(s, str):
        return ""
    s = re.sub(r'http[s]?://\S+', ' ', s)
    s = re.sub(r'[^\u4e00-\u9fffA-Za-z0-9\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def load_stopwords(path):
    if not os.path.exists(path):
        return set()
    with open(path, 'r', encoding='utf-8') as f:
        return set([w.strip() for w in f if w.strip()])

def setup_jieba(custom_dict=None):
    if custom_dict and os.path.exists(custom_dict):
        jieba.load_userdict(custom_dict)

def tokenize_texts(texts, stopwords=None, show_progress=True):
    stopwords = stopwords or set()
    tokenized = []
    iterable = texts
    if show_progress:
        iterable = tqdm(texts, desc="tokenize")
    for s in iterable:
        s2 = clean_text(s)
        if not s2:
            tokenized.append([])
            continue
        segs = [w for w in jieba.lcut(s2) if w.strip() and w not in stopwords]
        tokenized.append(segs)
    return tokenized

def build_doc_freqs(tokenized_texts):
    N_docs = len(tokenized_texts)
    doc_freq = Counter()
    for sent in tokenized_texts:
        unique = set(sent)
        doc_freq.update(unique)
    return doc_freq, N_docs

def build_cooccurrence_doclevel(tokenized_texts, core_set):
    """
    返回：co_doc_count: dict[(core, term)] -> count
    """
    co_doc = Counter()
    for sent in tokenized_texts:
        unique = set(sent)
        cores_in_doc = unique & core_set
        if not cores_in_doc:
            continue
        for core in cores_in_doc:
            for term in unique:
                if term == core:
                    continue
                co_doc[(core, term)] += 1
    return co_doc

def compute_pmi_tscore(co_doc, doc_freq, N_docs):
    results = []
    for (core, term), co in co_doc.items():
        df_core = doc_freq.get(core, 0)
        df_term = doc_freq.get(term, 0)
        if co <= 0 or df_core == 0 or df_term == 0:
            continue
        # PMI (document-level)
        # p(core,term) = co / N_docs
        # p(core) = df_core / N_docs, p(term) = df_term / N_docs
        # PMI = log( p(core,term) / (p(core) p(term)) ) = log(N_docs * co / (df_core * df_term))
        pmi = math.log((N_docs * co) / (df_core * df_term) + 1e-12)
        # t-score: (O - E) / sqrt(O), E = df_core * df_term / N_docs
        E = (df_core * df_term) / N_docs
        tscore = (co - E) / math.sqrt(co) if co > 0 else 0.0
        results.append({
            "core": core,
            "term": term,
            "co_doc_count": co,
            "df_core": df_core,
            "df_term": df_term,
            "pmi": pmi,
            "tscore": tscore,
            "E": E
        })
    return results

def extract_examples_for_pairs(df_texts, tokenized_texts, pairs, max_examples=3):
    # pairs: set of tuples (core, term)
    examples = {p: [] for p in pairs}
    for raw, tokens in zip(df_texts, tokenized_texts):
        uniq = set(tokens)
        for p in list(pairs):
            core, term = p
            if core in uniq and term in uniq:
                examples[p].append(raw.strip())
                if len(examples[p]) >= max_examples:
                    pairs.discard(p)  # optional: once we have enough, stop collecting for this pair
        if not pairs:
            break
    return examples

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Loading CSV:", CSV_PATH)
    df = pd.read_csv(CSV_PATH, dtype=str)
    if TEXT_COL not in df.columns:
        raise ValueError(f"CSV missing column {TEXT_COL}")
    texts = df[TEXT_COL].fillna("").astype(str).tolist()

    # jieba setup
    setup_jieba(CUSTOM_DICT)
    stopwords = load_stopwords(STOPWORDS_PATH)
    tokenized = tokenize_texts(texts, stopwords=stopwords)

    # doc freq
    doc_freq, N_docs = build_doc_freqs(tokenized)
    print(f"Documents: {N_docs}, unique terms in doc_freq: {len(doc_freq)}")

    core_set = set([c for c in core_terms if c in doc_freq])
    missing = [c for c in core_terms if c not in core_set]
    if missing:
        print("Warning: some core terms not found in corpus:", missing)
    if not core_set:
        raise ValueError("No core terms found in corpus; check your CORE_TERMS or tokenization.")

    co_doc = build_cooccurrence_doclevel(tokenized, core_set)
    print(f"Found {len(co_doc)} core-term co-occurrence pairs (before filtering)")

    # filter by min co
    co_doc_filtered = Counter({k:v for k,v in co_doc.items() if v >= MIN_CO})
    print(f"{len(co_doc_filtered)} pairs remain after MIN_CO={MIN_CO} filter")

    stats = compute_pmi_tscore(co_doc_filtered, doc_freq, N_docs)
    df_stats = pd.DataFrame(stats)
    if df_stats.empty:
        print("No pairs after filtering—consider lowering MIN_CO or checking tokenization.")
        return

    # add composite score (normalize pmi and tscore to combine)
    # stable normalization: rank-based or min-max within each core; here we do min-max globally
    for col in ("pmi","tscore"):
        vmax = df_stats[col].max()
        vmin = df_stats[col].min()
        if vmax > vmin:
            df_stats[f"{col}_norm"] = (df_stats[col] - vmin) / (vmax - vmin)
        else:
            df_stats[f"{col}_norm"] = 0.0
    # composite: weighted sum (可调整权重)
    df_stats["score"] = 0.5 * df_stats["pmi_norm"] + 0.5 * df_stats["tscore_norm"]
    df_stats = df_stats.sort_values(["core","score"], ascending=[True, False])

    # extract examples for top pairs across all cores (to save time, we collect for topK per core)
    top_pairs = []
    for core, g in df_stats.groupby("core"):
        top = g.head(TOPK_PER_CORE)
        for _, r in top.iterrows():
            top_pairs.append((r["core"], r["term"]))
    top_pairs_set = set(top_pairs)
    examples_map = extract_examples_for_pairs(df[TEXT_COL].fillna("").astype(str).tolist(), tokenized, set(top_pairs_set), max_examples=3)

    # ---- NEW: keep top 10 terms per core ----
    df_stats = (
        df_stats.sort_values(["core", "score"], ascending=[True, False])
        .groupby("core")
        .head(10)
        .reset_index(drop=True)
    )

    # attach examples
    def get_examples(row):
        key = (row["core"], row["term"])
        exs = examples_map.get(key, [])
        return " || ".join(exs)
    df_stats["examples"] = df_stats.apply(get_examples, axis=1)


    out_path = os.path.join(OUTPUT_DIR, "cooc_stats_by_core.csv")
    df_stats.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("Saved:", out_path)

    # # also save per-core top files (PMI and tscore)
    # for core, g in df_stats.groupby("core"):
    #     g.sort_values("pmi", ascending=False).head(TOPK_PER_CORE).to_csv(os.path.join(OUTPUT_DIR, f"{core}_top_pmi.csv"), index=False, encoding="utf-8-sig")
    #     g.sort_values("tscore", ascending=False).head(TOPK_PER_CORE).to_csv(os.path.join(OUTPUT_DIR, f"{core}_top_tscore.csv"), index=False, encoding="utf-8-sig")
    # print("Per-core top files saved.")

if __name__ == "__main__":
    main()
