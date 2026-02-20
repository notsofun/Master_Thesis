import torch
import gc, os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sentence_transformers import SentenceTransformer, util

os.environ["SENTENCE_TRANSFORMERS_HOME"] = "/root/autodl-tmp/hf_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/root/autodl-tmp/hf_cache"

# 强制使用 GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {device}")

# 加载生成的 100+ 条数据
try:
    with open('testsets/cn_ja_testset.json', 'r', encoding='utf-8') as f:
        test_suites = json.load(f)
except FileNotFoundError:
    print("请先运行上面的 generator 脚本生成 JSON 文件！")
    exit()

MODELS = [
    "BAAI/bge-m3",
    "jinaai/jina-embeddings-v3",
    "intfloat/multilingual-e5-large-instruct",
    "sentence-transformers/LaBSE",
    "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "HIT-TMG/KaLM-embedding-multilingual-mini-v1",
    "shibing624/text2vec-base-multilingual",
    "Alibaba-NLP/gte-multilingual-base",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "intfloat/multilingual-e5-base",
    "Snowflake/snowflake-arctic-embed-m-v1.5"
]

def run_bench():
    # 存储所有单独的样本得分，而不是平均值，以便做统计检验
    raw_data_records = [] 
    
    for m_id in MODELS:
        print(f"Evaluating {m_id} ...")
        try:
            model = SentenceTransformer(m_id, device=device, trust_remote_code=True)
            
            for cat, pairs in test_suites.items():
                # 批量编码以加速 (Batch Processing)
                batch_en = [p['en'] for p in pairs]
                batch_zh = [p['zh'] for p in pairs]
                batch_ja = [p['ja'] for p in pairs]
                
                # 编码
                emb_en = model.encode(batch_en, convert_to_tensor=True, normalize_embeddings=True)
                emb_zh = model.encode(batch_zh, convert_to_tensor=True, normalize_embeddings=True)
                emb_ja = model.encode(batch_ja, convert_to_tensor=True, normalize_embeddings=True)
                
                # 计算余弦相似度 (Pairwise)
                # distinct cos_sim for each pair
                scores_zh = torch.sum(emb_en * emb_zh, dim=1).cpu().numpy()
                scores_ja = torch.sum(emb_en * emb_ja, dim=1).cpu().numpy()
                
                # 记录每一个样本的得分
                for i in range(len(pairs)):
                    avg_score = (scores_zh[i] + scores_ja[i]) / 2
                    raw_data_records.append({
                        "Model": m_id.split('/')[-1],
                        "Category": cat,
                        "Score": float(avg_score)
                    })
            
            # 清理显存
            del model
            gc.collect()
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error: {e}")

    # ==========================================
    # 数据分析与统计检验
    # ==========================================
    df = pd.DataFrame(raw_data_records)
    
    # 1. 计算 T-test (Standard vs Slur_Alignment)
    print("\n" + "="*80)
    print("显著性检验 (T-test: Standard vs Slur)")
    print("H0: 两者均值无差异 (P > 0.05) | H1: 仇恨语义导致显著下降 (P < 0.05)")
    print("="*80)
    
    stats_results = []
    for m in df['Model'].unique():
        subset = df[df['Model'] == m]
        std_scores = subset[subset['Category'] == 'Standard_Baseline']['Score']
        slur_scores = subset[subset['Category'] == 'Slur_Alignment']['Score']
        
        # 独立双样本 T 检验
        t_stat, p_val = stats.ttest_ind(std_scores, slur_scores, equal_var=False)
        
        gap = std_scores.mean() - slur_scores.mean()
        drop_rate = (gap / std_scores.mean()) * 100
        
        stats_results.append({
            "Model": m,
            "Mean_Std": std_scores.mean(),
            "Mean_Slur": slur_scores.mean(),
            "Gap": gap,
            "Drop_Rate(%)": drop_rate,
            "P-Value": p_val,
            "Significant?": "YES (***)" if p_val < 0.001 else ("YES (*)" if p_val < 0.05 else "NO")
        })
    
    stats_df = pd.DataFrame(stats_results).sort_values("Gap", ascending=False)
    print(stats_df.to_string(index=False))
    stats_df.to_csv("statistical_analysis.csv", index=False)

    # ==========================================
    # 可视化 (Boxplot) - 论文级图表
    # ==========================================
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")
    
    # 定义绘图顺序
    order = ["Standard_Baseline", "Slur_Alignment", "Hate_Complex", "Random_Control"]
    
    # 绘制箱线图：展示中位数、四分位距和异常值
    ax = sns.boxplot(x="Model", y="Score", hue="Category", data=df, 
                     hue_order=order, palette="Set2", showfliers=False)
    
    # 叠加散点图 (Strip plot) 展示真实分布，适合 N=100 左右的数据
    sns.stripplot(x="Model", y="Score", hue="Category", data=df, 
                  hue_order=order, dodge=True, alpha=0.4, color=".3", legend=False)

    plt.title("Cross-Lingual Semantic Alignment Stability: Standard vs. Hate Speech", fontsize=14)
    plt.ylabel("Cosine Similarity (EN <-> ZH/JA)", fontsize=12)
    plt.xlabel("Embedding Model", fontsize=12)
    plt.xticks(rotation=15)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()
    
    # 保存图片
    plt.savefig("alignment_boxplot.png", dpi=300)
    print("\n图表已保存为: alignment_boxplot.png")
    print("原始数据已保存为: statistical_analysis.csv")

if __name__ == "__main__":
    run_bench()