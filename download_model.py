import os
from modelscope.hub.snapshot_download import snapshot_download

# 定义模型映射（HF ID -> ModelScope ID）
# 绝大部分模型在 ModelScope 上需要加前缀或微调路径
model_mapping = {
    "intfloat/multilingual-e5-large-instruct": "AI-ModelScope/multilingual-e5-large-instruct",
    "sentence-transformers/LaBSE": "sentence-transformers/LaBSE",
    "Alibaba-NLP/gte-Qwen2-1.5B-instruct": "AI-ModelScope/gte-Qwen2-1.5B-instruct",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "HIT-TMG/KaLM-embedding-multilingual-mini-v1": "HIT-TMG/KaLM-embedding-multilingual-mini-v1",
    "shibing624/text2vec-base-multilingual": "shibing624/text2vec-base-multilingual",
    "Alibaba-NLP/gte-multilingual-base": "AI-ModelScope/gte-multilingual-base",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "intfloat/multilingual-e5-base": "AI-ModelScope/multilingual-e5-base",
    "Snowflake/snowflake-arctic-embed-m-v1.5": "AI-ModelScope/snowflake-arctic-embed-m-v1.5"
}

save_root = "/root/autodl-tmp/models"

for hf_id, ms_id in model_mapping.items():
    print(f"--- 正在下载: {hf_id} (ModelScope ID: {ms_id}) ---")
    try:
        # 下载模型到指定目录
        download_path = snapshot_download(ms_id, cache_dir=save_root)
        print(f"成功保存至: {download_path}")
    except Exception as e:
        print(f"下载失败 {hf_id}: {e}")

print("\n所有模型准备就绪！")