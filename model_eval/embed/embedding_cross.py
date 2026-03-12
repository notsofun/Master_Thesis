# python model_eval/embed/embedding_cross.py
import argparse
import pathlib
import sys, os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.neighbors import NearestNeighbors
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import json
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# -------- helpers -------------------------------------------------

def load_testset(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def flatten_suite(test_suites):
    data_list = []
    for category, items in test_suites.items():
        for item in items:
            for lang, text in item.items():
                data_list.append({
                    "text": text,
                    "lang": lang,
                    "category": category,
                    "group_id": id(item),
                })
    return pd.DataFrame(data_list)


def compute_alignment(df):
    alignment_scores = []
    for gid in df['group_id'].unique():
        group = df[df['group_id'] == gid]
        vectors = np.stack(group['embedding'].values)
        cos_sim = util.cos_sim(vectors, vectors)
        tri_mask = np.triu(np.ones(cos_sim.shape), k=1).astype(bool)
        alignment_scores.append(cos_sim[tri_mask].mean().item())
    return np.mean(alignment_scores), alignment_scores


def compute_retrieval(embs, labels, topk=5):
    nbrs = NearestNeighbors(n_neighbors=topk + 1, metric='cosine').fit(embs)
    distances, indices = nbrs.kneighbors(embs)
    precisions = []
    for i, true_cat in enumerate(labels):
        neigh_cats = [labels[j] for j in indices[i, 1:]]
        correct = sum(1 for c in neigh_cats if c == true_cat)
        precisions.append(correct / topk)
    return np.mean(precisions)


# -------- evaluation -------------------------------------------------

def evaluate_model(model_name, df):
    print(f"正在评估模型: {model_name}")
    
    # --- 修改部分：自动检测设备 ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = 'cpu'
    print(f"使用设备: {device}")
    
    # 在初始化时显式传入 device
    model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
    # ----------------------------

    texts = df['text'].tolist()
    
    # encode 阶段也可以确保
    embeddings = model.encode(texts, show_progress_bar=True, device=device)

    df = df.copy()
    df['embedding'] = list(embeddings)

    avg_align, all_align = compute_alignment(df)
    sil = silhouette_score(embeddings, df['category'])
    db = davies_bouldin_score(embeddings, df['category'])
    ch = calinski_harabasz_score(embeddings, df['category'])
    prec5 = compute_retrieval(embeddings, df['category'].tolist(), topk=5)

    return {
        "Model": model_name,
        "Alignment": avg_align,
        "Silhouette": sil,
        "DB": db,
        "CH": ch,
        "Prec@5": prec5,
        "Embeddings": embeddings,
        "AlignScores": all_align,
        "DataFrame": df,
    }


# -------- plotting -------------------------------------------------

def plot_umap(embs, df, ax, title):
    reducer = umap.UMAP(n_neighbors=5, min_dist=0.3, random_state=42)
    embs_2d = reducer.fit_transform(embs)
    sns.scatterplot(
        x=embs_2d[:, 0],
        y=embs_2d[:, 1],
        hue=df['category'],
        style=df['lang'],
        ax=ax,
        s=100,
    )
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


def plot_alignment_box(results_df, outdir):
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Model', y='Alignment', data=results_df)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(outdir / 'alignment_boxplot.png', dpi=300)
    plt.close()


# -------- main -------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="跨语言 embedding 评估")
    parser.add_argument(
        "--testset",
        type=pathlib.Path,
        default="model_eval/embed/testsets/cn_ja_testset.json",
    )
    parser.add_argument(
        "--outdir",
        type=pathlib.Path,
        default=pathlib.Path("model_eval/embed/BAAI_results"),
    )
    args = parser.parse_args()

    if not args.testset.exists():
        print(f"测试集文件不存在: {args.testset}")
        sys.exit(1)
    args.outdir.mkdir(parents=True, exist_ok=True)

    test_suites = load_testset(str(args.testset))
    df_eval = flatten_suite(test_suites)

    models_to_test = [
        # "/root/autodl-tmp/models/AI-ModelScope/multilingual-e5-large-instruct",
        "/root/autodl-tmp/models/sentence-transformers/LaBSE",
        # "/root/autodl-tmp/models/AI-ModelScope/gte-Qwen2-1___5B-instruct",
        "/root/autodl-tmp/models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "/root/autodl-tmp/models/HIT-TMG/KaLM-embedding-multilingual-mini-v1",
        # "/root/autodl-tmp/models/shibing624/text2vec-base-multilingual",
        # "/root/autodl-tmp/models/AI-ModelScope/gte-multilingual-base",
        "/root/autodl-tmp/models/sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "/root/autodl-tmp/models/AI-ModelScope/multilingual-e5-base",
        "/root/autodl-tmp/models/AI-ModelScope/snowflake-arctic-embed-m-v1.5",
    ]

    retry_models = [
        "shibing624/text2vec-base-multilingual",
        "Alibaba-NLP/gte-multilingual-base",
        # "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        "intfloat/multilingual-e5-large-instruct",
        # "BAAI/bge-m3",
    ]

    BAI_Model = ["BAAI/bge-m3"]
        
    results = []
    n = len(BAI_Model)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5)) if n > 1 else (None, [None])
    for i, m in enumerate(BAI_Model):
        r = evaluate_model(m, df_eval)
        results.append(r)
        if n > 1:
            plot_umap(
                r['Embeddings'],
                r['DataFrame'],
                axes[i],
                f"{m.split('/')[-1]}\nAlign: {r['Alignment']:.3f} | Sep: {r['Silhouette']:.3f}",
            )
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    if n > 1:
        plt.tight_layout()
        plt.savefig(args.outdir / "umap_grid.png", dpi=300)
        plt.close()

    df_results = pd.DataFrame(results).drop(
        columns=['Embeddings', 'AlignScores', 'DataFrame']
    )
    df_results.to_csv(args.outdir / "cross_statistical_analysis.csv", index=False)
    plot_alignment_box(df_results, args.outdir)

    print("\n--- 最终量化评估报告 ---")
    print(df_results)


if __name__ == "__main__":
    main()
