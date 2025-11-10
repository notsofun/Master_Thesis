import pandas as pd
import numpy as np
import os

# === 假设你已经有两个 DataFrame：pmi_df 和 tscore_df ===
# 它们至少包含 ['a', 'b', 'pmi', 't_score'] 四列
# 并且你已经有 TARGET_TERMS 列表

pmi_df = pd.read_csv(r'candidate_keywords\candidate_keywords_pmi_15.txt')
tscore_df = pd.read_csv(r'candidate_keywords\candidate_keywords_tscore_76.txt')
TARGET_TERMS = ["christian", "christians", "church", "jesus", "bible", "priest"]

# 提取共现的词（去掉 target 自身）
def extract_terms(df, top_n=50):
    words = []
    for _, row in df.head(top_n).iterrows():
        for term in (row["a"], row["b"]):
            if term not in TARGET_TERMS:
                words.append(term)
    return set(words)

# --- ① 核心共现扩展（交集）---
core_terms = extract_terms(pmi_df, 50) & extract_terms(tscore_df, 50)

# --- ② 双轨版本 ---
latent_terms = extract_terms(pmi_df, 50) - extract_terms(tscore_df, 50)
stable_terms = extract_terms(tscore_df, 50)

# --- ③ 加权融合版本 ---
# 归一化两个指标
merged_df = pd.merge(pmi_df, tscore_df, on=['a','b'], suffixes=('_pmi','_ts'))
merged_df["pmi_norm"] = (merged_df["pmi_pmi"] - merged_df["pmi_pmi"].min()) / (merged_df["pmi_pmi"].max() - merged_df["pmi_pmi"].min())
merged_df["ts_norm"] = (merged_df["t_score_ts"] - merged_df["t_score_ts"].min()) / (merged_df["t_score_ts"].max() - merged_df["t_score_ts"].min())

alpha = 0.6  # PMI 权重（可调整）
merged_df["fused_score"] = alpha * merged_df["pmi_norm"] + (1 - alpha) * merged_df["ts_norm"]

fused_terms = extract_terms(merged_df.sort_values("fused_score", ascending=False), 50)

# === 保存结果 ===
os.makedirs("candidate_keywords", exist_ok=True)
pd.DataFrame({"term": sorted(core_terms)}).to_csv("candidate_keywords/targets_core.csv", index=False)
pd.DataFrame({"term": sorted(latent_terms)}).to_csv("candidate_keywords/targets_latent.csv", index=False)
pd.DataFrame({"term": sorted(fused_terms)}).to_csv("candidate_keywords/targets_fused.csv", index=False)

# === 打印统计 ===
print(f"Core terms (intersection): {len(core_terms)}")
print(f"Stable (T-score only): {len(stable_terms)} | Latent (PMI only): {len(latent_terms)}")
#这个latent为0，说明就是没有差值
print(f"Fused version total: {len(fused_terms)}")
