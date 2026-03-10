#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final keyword analysis script for Chinese, English, and Japanese texts
Combines analysis, term frequency, and co-occurrence patterns
Based on respective language-specific analyze scripts
"""

import os
import re
import sys
import math
import json
from collections import Counter, defaultdict
from tqdm import tqdm
import pandas as pd
import numpy as np

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(os.path.dirname(current_dir), '..')
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scripts.set_logger import setup_logging

# 多语言分词依赖
import jieba
from janome.tokenizer import Tokenizer as JanomeTokenizer
from nltk.tokenize import TweetTokenizer
import nltk

# 下载必要的NLTK资源
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# ==================== 配置 ====================
CORPUS_PATHS = {
    'zh': '../../data_detect/finetuned_detection/chinese_final_religious_hate.csv',
    'en': '../../data_collection/English_Existing/merged_deduped.csv',
    'jp': '../../data_detect/finetuned_detection/japanese_final_religious_hate.csv'
}

# 停用词文件路径
STOPWORDS_PATHS = {
    'zh': '../../data_preanalysis/Chinese/merged_stopwords.txt',
    'en': None,  # 使用sklearn内置的英文停用词
    'jp': '../../data_preanalysis/Japanese/stopwords-ja.txt'  # 可选，若无则为空
}

OUTPUT_DIR = './output'
MIN_TERM_FREQ = 5        # 最小词频
MIN_CO = 3               # 最小共现频次
TOPK_TERMS = 300         # 输出Top K词汇
TOPK_PAIRS_PER_CORE = 50 # 每个核心词的Top K共现词
# ============================================

# 初始化日志
logger, _ = setup_logging()

# 初始化多语言分词工具
jt = JanomeTokenizer()
tweet_tok = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)

# 导入英文停用词
try:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    EN_STOPWORDS = set(ENGLISH_STOP_WORDS)
except:
    EN_STOPWORDS = set()

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================== 停用词加载函数 ====================
def load_stopwords(lang):
    """加载指定语言的停用词"""
    if lang == 'en':
        return EN_STOPWORDS
    
    stopwords_path = STOPWORDS_PATHS.get(lang)
    if not stopwords_path or stopwords_path.startswith('NOT_FOUND'):
        logger.warning(f"{lang} 未配置停用词文件，将不过滤停用词")
        return set()
    
    abs_path = os.path.join(current_dir, stopwords_path)
    if not os.path.exists(abs_path):
        logger.warning(f"找不到 {lang} 停用词文件: {abs_path}，将不过滤停用词")
        return set()
    
    try:
        with open(abs_path, 'r', encoding='utf-8') as f:
            stopwords = set([line.strip() for line in f if line.strip()])
        logger.info(f"成功加载 {lang} 停用词，共 {len(stopwords)} 个")
        return stopwords
    except Exception as e:
        logger.error(f"加载 {lang} 停用词失败: {e}")
        return set()


# ================== 文本清洗 ====================
def clean_text_zh(s):
    """中文文本清洗"""
    if not isinstance(s, str):
        return ""
    s = re.sub(r'http[s]?://\S+', ' ', s)
    s = re.sub(r'[^\u4e00-\u9fffA-Za-z0-9\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def clean_text_jp(s):
    """日文文本清洗"""
    if not isinstance(s, str):
        return ""
    s = re.sub(r'http[s]?://\S+', ' ', s)
    # 保留平假名、片假名、汉字、ASCII和日本延长音符
    s = re.sub(r'[^\u3040-\u30ff\u4e00-\u9fff\uff10-\uff19A-Za-z0-9ー]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def clean_text_en(s):
    """英文文本清洗"""
    if not isinstance(s, str):
        return ""
    s = s.replace('\u200b', ' ')
    s = re.sub(r'https?://\S+|www\.\S+', ' ', s)
    s = re.sub(r'\S+@\S+', ' ', s)
    s = s.replace('\n', ' ')
    s = s.lower()
    s = re.sub(r"[^A-Za-z0-9'\- ]+", ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

# ================== 分词函数 ====================
def tokenize_zh(text, stopwords=None):
    """中文分词 - jieba"""
    stopwords = stopwords or set()
    s = clean_text_zh(text)
    if not s:
        return []
    return [w for w in jieba.lcut(s) if w.strip() and w not in stopwords]

def tokenize_jp(text, stopwords=None):
    """日文分词 - janome"""
    stopwords = stopwords or set()
    s = clean_text_jp(text)
    if not s:
        return []
    tokens = []
    for token in jt.tokenize(s):
        base = token.base_form if token.base_form != '*' else token.surface
        part = token.part_of_speech.split(',')[0]
        if part == '記号':
            continue
        if re.match(r'^[A-Za-z0-9]+$', base):
            base = base.lower()
        if base not in stopwords:
            tokens.append(base)
    return tokens

def tokenize_en(text, stopwords=None):
    """英文分词 - TweetTokenizer"""
    stopwords = stopwords or set()
    s = clean_text_en(text)
    if not s:
        return []
    tokens = tweet_tok.tokenize(s)
    return [t for t in tokens if t not in stopwords and not t.isdigit()]

# ================== 分词驱动函数 ====================
def tokenize_texts(texts, lang, stopwords=None):
    """统一的分词接口"""
    tokenizers = {
        'zh': tokenize_zh,
        'en': tokenize_en,
        'jp': tokenize_jp
    }
    tokenizer = tokenizers.get(lang)
    if not tokenizer:
        raise ValueError(f"不支持的语言: {lang}")
    
    tokenized = []
    for text in tqdm(texts, desc=f"Tokenize {lang}", total=len(texts)):
        tokenized.append(tokenizer(text, stopwords))
    return tokenized

# ================== 统计函数 ====================
def build_doc_freqs(tokenized_texts):
    """计算文档频次"""
    N_docs = len(tokenized_texts)
    doc_freq = Counter()
    for sent in tokenized_texts:
        unique = set(sent)
        doc_freq.update(unique)
    return doc_freq, N_docs

def build_term_freqs(tokenized_texts):
    """计算词频"""
    term_freq = Counter()
    for sent in tokenized_texts:
        term_freq.update(sent)
    return term_freq

def build_cooccurrence_doclevel(tokenized_texts):
    """文档级共现统计"""
    co_doc = Counter()
    for sent in tokenized_texts:
        unique = set(sent)
        if len(unique) < 2:
            continue
        # 计算所有词对的共现
        terms_list = list(unique)
        for i in range(len(terms_list)):
            for j in range(i + 1, len(terms_list)):
                a, b = terms_list[i], terms_list[j]
                # 使用排序确保结果一致性
                pair = tuple(sorted([a, b]))
                co_doc[pair] += 1
    return co_doc

def compute_pmi_tscore(co_doc, doc_freq, N_docs):
    """计算PMI和T-score"""
    results = []
    for (a, b), co in co_doc.items():
        df_a = doc_freq.get(a, 0)
        df_b = doc_freq.get(b, 0)
        if co <= 0 or df_a == 0 or df_b == 0:
            continue
        
        # PMI (document-level)
        pmi = math.log((N_docs * co) / (df_a * df_b) + 1e-12)
        
        # t-score
        E = (df_a * df_b) / N_docs
        tscore = (co - E) / math.sqrt(co) if co > 0 else 0.0
        
        results.append({
            'term_a': a,
            'term_b': b,
            'co_doc_count': co,
            'df_a': df_a,
            'df_b': df_b,
            'pmi': pmi,
            'tscore': tscore,
            'E': E
        })
    return results

# ================== 主分析函数 ====================
def analyze_language(lang, csv_path, stopwords=None, text_col='text'):
    """对单一语言的文本进行分析"""
    logger.info(f"开始分析 {lang} 语言，路径: {csv_path}")
    
    # 加载CSV
    if not os.path.exists(csv_path):
        logger.error(f"找不到文件: {csv_path}")
        return None
    
    try:
        df = pd.read_csv(csv_path, dtype=str)
    except Exception as e:
        logger.error(f"读取 {csv_path} 失败: {e}")
        return None
    
    if text_col not in df.columns:
        # 尝试第二列
        text_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
        logger.warning(f"使用列 {text_col}")
    
    texts = df[text_col].fillna("").astype(str).tolist()
    logger.info(f"成功加载 {len(texts)} 条文本")
    
    # 分词（传入停用词）
    tokenized = tokenize_texts(texts, lang, stopwords=stopwords)
    
    # 计算频次
    doc_freq, N_docs = build_doc_freqs(tokenized)
    term_freq = build_term_freqs(tokenized)
    
    logger.info(f"文档数: {N_docs}, 唯一词汇数: {len(doc_freq)}")
    
    # 过滤低频词
    doc_freq_filtered = Counter({k: v for k, v in doc_freq.items() if v >= MIN_TERM_FREQ})
    term_freq_filtered = Counter({k: v for k, v in term_freq.items() if k in doc_freq_filtered})
    
    logger.info(f"过滤后词汇数: {len(doc_freq_filtered)}")
    
    # 共现分析
    co_doc = build_cooccurrence_doclevel(tokenized)
    co_doc_filtered = Counter({k: v for k, v in co_doc.items() if v >= MIN_CO})
    logger.info(f"共现对数: {len(co_doc_filtered)}")
    
    # 计算PMI和T-score
    stats = compute_pmi_tscore(co_doc_filtered, doc_freq_filtered, N_docs)
    
    return {
        'lang': lang,
        'term_freq': term_freq_filtered,
        'doc_freq': doc_freq_filtered,
        'cooccurrence_stats': stats,
        'N_docs': N_docs,
        'texts': texts,
        'tokenized': tokenized
    }

# ================== 输出函数 ====================
def save_results(lang, results):
    """保存分析结果"""
    if results is None:
        return
    
    lang_dir = os.path.join(OUTPUT_DIR, lang)
    os.makedirs(lang_dir, exist_ok=True)
    
    # 1. 词频统计
    term_freq = results['term_freq']
    top_terms = term_freq.most_common(TOPK_TERMS)
    df_terms = pd.DataFrame(top_terms, columns=['term', 'freq'])
    df_terms.to_csv(os.path.join(lang_dir, f'{lang}_top_terms.csv'), index=False, encoding='utf-8-sig')
    logger.info(f"已保存 {lang} 词频统计到 {lang_dir}/{lang}_top_terms.csv")
    
    # 2. 共现分析
    stats = results['cooccurrence_stats']
    if stats:
        df_cooc = pd.DataFrame(stats)
        
        # 归一化分数
        for col in ['pmi', 'tscore']:
            vmax = df_cooc[col].max()
            vmin = df_cooc[col].min()
            if vmax > vmin:
                df_cooc[f'{col}_norm'] = (df_cooc[col] - vmin) / (vmax - vmin)
            else:
                df_cooc[f'{col}_norm'] = 0.0
        
        # 综合评分
        df_cooc['score'] = 0.5 * df_cooc['pmi_norm'] + 0.5 * df_cooc['tscore_norm']
        
        # 按PMI排序输出
        df_pmi = df_cooc.sort_values('pmi', ascending=False)[['term_a', 'term_b', 'co_doc_count', 'pmi', 'tscore', 'score']]
        df_pmi.to_csv(os.path.join(lang_dir, f'{lang}_cooccurrence_pmi.csv'), index=False, encoding='utf-8-sig')
        logger.info(f"已保存 {lang} 共现分析到 {lang_dir}/{lang}_cooccurrence_pmi.csv")
        
        # 按T-score排序输出
        df_tscore = df_cooc.sort_values('tscore', ascending=False)[['term_a', 'term_b', 'co_doc_count', 'pmi', 'tscore', 'score']]
        df_tscore.to_csv(os.path.join(lang_dir, f'{lang}_cooccurrence_tscore.csv'), index=False, encoding='utf-8-sig')
        logger.info(f"已保存 {lang} T-score排序到 {lang_dir}/{lang}_cooccurrence_tscore.csv")
    
    # 3. 文档统计摘要
    summary = {
        'language': lang,
        'document_count': results['N_docs'],
        'unique_terms': len(results['term_freq']),
        'unique_doc_freq_terms': len(results['doc_freq']),
        'cooccurrence_pairs': len(stats) if stats else 0
    }
    
    with open(os.path.join(lang_dir, f'{lang}_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"已保存 {lang} 摘要到 {lang_dir}/{lang}_summary.json")

def main():
    logger.info("开始最终关键词分析")
    logger.info(f"输出目录: {OUTPUT_DIR}")
    
    # 处理三种语言
    all_results = {}
    for lang, csv_path in CORPUS_PATHS.items():
        # 加载停用词
        stopwords = load_stopwords(lang)
        
        # 转换为相对路径
        abs_path = os.path.join(current_dir, csv_path)
        results = analyze_language(lang, abs_path, stopwords=stopwords)
        if results:
            all_results[lang] = results
            save_results(lang, results)
    
    logger.info("=== 最终关键词分析完成 ===")
    logger.info(f"结果已保存到: {os.path.abspath(OUTPUT_DIR)}")
    
    # 输出摘要
    logger.info("\n=== 分析摘要 ===")
    for lang, results in all_results.items():
        top_5_terms = results['term_freq'].most_common(5)
        logger.info(f"{lang.upper()} Top 5 词汇: {[t[0] for t in top_5_terms]}")

if __name__ == "__main__":
    main()
