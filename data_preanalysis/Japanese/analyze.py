#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Japanese co-occurrence analysis (document-level PMI & t-score)

Usage: edit constants below to point to your CSV and run.

This script mirrors the structure of the Chinese `analyze.py` in this repo
and uses `janome` for tokenization. It produces `output/cooc_stats_by_core.csv`.
"""
import os
import re
import math
import json
from collections import Counter
from tqdm import tqdm
import pandas as pd

try:
    from janome.tokenizer import Tokenizer
except Exception:
    raise ImportError("Please install janome: pip install janome")

# ------------------ CONFIG ------------------
# default CSV: use the 5ch file included in the workspace (change as needed)
CSV_PATH = "../../data_collection/5ch/20251123_140301_5ch_posts.csv"
TEXT_COL = "text"
CUSTOM_DICT = None  # janome doesn't support a simple userdict load like jieba, keep None
STOPWORDS_PATH = "stopwords-ja.txt"  # optional path to a newline-separated Japanese stopwords file
OUTPUT_DIR = "output"
MIN_CO = 3
TOPK_PER_CORE = 50
# 读取核心词列表
with open("../../data_collection/fianl_keywors.json", "r", encoding="utf-8") as f:
    kw_json = json.load(f)
CORE_TERMS = kw_json["Japanese"]

# --------------------------------------------

_RE_URL = re.compile(r'http[s]?://\S+')
# keep Hiragana, Katakana, Kanji, ASCII letters/digits and prolongation mark
_RE_NONJP = re.compile(r'[^\u3040-\u30ff\u4e00-\u9fff\uff10-\uff19A-Za-z0-9ー]')

def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = _RE_URL.sub(' ', s)
    s = _RE_NONJP.sub(' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def load_stopwords(path):
    if not path or not os.path.exists(path):
        return set()
    with open(path, 'r', encoding='utf-8') as f:
        return set([w.strip() for w in f if w.strip()])

def setup_tokenizer():
    # janome Tokenizer is sufficient for many cases; user can replace with MeCab
    return Tokenizer()

def tokenize_doc_janome(tokenizer, text, stopwords=None):
    """Return list of single-token base forms for the document.

    We intentionally only return single tokens (no n-grams or joined compounds)
    so co-occurrence terms are single tokens as requested.
    """
    stopwords = stopwords or set()
    s = clean_text(text)
    if not s:
        return []
    tokens = []
    for token in tokenizer.tokenize(s):
        base = token.base_form if token.base_form != '*' else token.surface
        part = token.part_of_speech.split(',')[0]
        # filter out symbols and punctuation
        if part == '記号':
            continue
        # normalize ascii parts
        if re.match(r'^[A-Za-z0-9]+$', base):
            base = base.lower()
        if base in stopwords:
            continue
        tokens.append(base)
    return tokens

def tokenize_texts(texts, stopwords=None, show_progress=True):
    tokenizer = setup_tokenizer()
    tokenized = []
    iterable = texts
    if show_progress:
        iterable = tqdm(texts, desc="tokenize")
    for s in iterable:
        tokenized.append(tokenize_doc_janome(tokenizer, s, stopwords=stopwords))
    return tokenized

def build_doc_freqs(tokenized_texts):
    N_docs = len(tokenized_texts)
    doc_freq = Counter()
    for sent in tokenized_texts:
        unique = set(sent)
        doc_freq.update(unique)
    return doc_freq, N_docs

def build_cooccurrence_doclevel(tokenized_texts, core_set):
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
        pmi = math.log((N_docs * co) / (df_core * df_term) + 1e-12)
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

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Loading CSV:", CSV_PATH)
    df = pd.read_csv(CSV_PATH, dtype=str)
    if TEXT_COL not in df.columns:
        raise ValueError(f"CSV missing column {TEXT_COL}")
    texts = df[TEXT_COL].fillna("").astype(str).tolist()

    stopwords = load_stopwords(STOPWORDS_PATH)
    tokenized = tokenize_texts(texts, stopwords=stopwords)

    doc_freq, N_docs = build_doc_freqs(tokenized)
    print(f"Documents: {N_docs}, unique terms in doc_freq: {len(doc_freq)}")

    core_set = set([c for c in CORE_TERMS if c in doc_freq])
    missing = [c for c in CORE_TERMS if c not in core_set]
    if missing:
        print("Warning: some core terms not found in corpus:", missing)
    if not core_set:
        raise ValueError("No core terms found in corpus; check your CORE_TERMS or tokenization.")

    co_doc = build_cooccurrence_doclevel(tokenized, core_set)
    print(f"Found {len(co_doc)} core-term co-occurrence pairs (before filtering)")

    co_doc_filtered = Counter({k:v for k,v in co_doc.items() if v >= MIN_CO})
    print(f"{len(co_doc_filtered)} pairs remain after MIN_CO={MIN_CO} filter")

    stats = compute_pmi_tscore(co_doc_filtered, doc_freq, N_docs)
    df_stats = pd.DataFrame(stats)
    if df_stats.empty:
        print("No pairs after filtering—consider lowering MIN_CO or checking tokenization.")
        return

    for col in ("pmi","tscore"):
        vmax = df_stats[col].max()
        vmin = df_stats[col].min()
        if vmax > vmin:
            df_stats[f"{col}_norm"] = (df_stats[col] - vmin) / (vmax - vmin)
        else:
            df_stats[f"{col}_norm"] = 0.0
    df_stats["score"] = 0.5 * df_stats["pmi_norm"] + 0.5 * df_stats["tscore_norm"]
    df_stats = df_stats.sort_values(["core","score"], ascending=[True, False])

    # keep top results per core (no examples column)
    df_stats = (
        df_stats.sort_values(["core", "score"], ascending=[True, False])
        .groupby("core")
        .head(10)
        .reset_index(drop=True)
    )

    out_path = os.path.join(OUTPUT_DIR, "cooc_stats_by_core.csv")
    df_stats.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("Saved:", out_path)

if __name__ == "__main__":
    main()

