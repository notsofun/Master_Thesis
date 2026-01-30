import torch
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# ==========================================
# 1. GPU 检查逻辑 (核心：确保 GPU 被用上)
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"当前使用的计算设备: {device.upper()}")

# ==========================================
# 2. 严格语义对齐的宗教仇恨测试对
# ==========================================
test_suites = {
    "Standard_Baseline": [
        {"en": "Religious faith provides spiritual guidance.", 
         "zh": "宗教信仰提供精神指引。", 
         "ja": "宗教的な信仰は精神的な指針を与えます。"}
    ],
    "Slur_Alignment": [
        # 测试重点：Slur (蔑称) 跨语言映射：christcuck (EN) == 基督狗 (ZH) == キリスト厨 (JA)
        {"en": "That christcuck is just a brainwashed puppet.", 
         "zh": "那个基督狗不过是个被洗脑的傀儡。", 
         "ja": "あのキリスト厨はただの洗脳されたパペットだ。"}
    ],
    "Metaphor_Hate": [
        # 测试重点：攻击性隐喻：Oppression Tool (EN) == 压迫工具 (ZH) == 存在否定の道具 (JA)
        {"en": "Christianity is a tool for racial oppression and denying existence.", 
         "zh": "基督教实质已经成为种族压迫和否定生存的工具。", 
         "ja": "キリスト教は人種抑圧と存在否定のために使われる道具だ。"}
    ],
    "Deception_Hate": [
        # 测试重点：教义抹黑：Fairy tale (EN) == 诡辩 (ZH) == 誤魔化す作り話 (JA)
        {"en": "The doctrine is a fairy tale to cover up truths.", 
         "zh": "教义不过是为了掩盖真相而编造的诡辩。", 
         "ja": "教義は不都合な真実を誤魔化すための作り話だ。"}
    ]
}

# ==========================================
# 3. 20 个开源模型列表
# ==========================================
OPEN_SOURCE_MODELS = [
    "BAAI/bge-m3",
    "jinaai/jina-embeddings-v3",
    "Alibaba-NLP/gte-Qwen2-7B-instruct",
    "Qwen/Qwen3-Embedding-8B",
    "intfloat/multilingual-e5-large-instruct",
    "nvidia/NV-Embed-v2",
    "sentence-transformers/LaBSE",
    "Alibaba-NLP/gte-multilingual-base",
    "HIT-TMG/KaLM-embedding-multilingual-mini-v1",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "shibing624/text2vec-base-multilingual",
    "nvidia/llama-embed-nemotron-8b",
    "Snowflake/snowflake-arctic-embed-m-v1.5",
    "ibm-granite/granite-embedding-278m-multilingual",
    "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2",
    "intfloat/multilingual-e5-base",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "BAAI/bge-multilingual-gemma2",
    "sentence-transformers/static-similarity-mrl-multilingual-v1",
    "intfloat/multilingual-e5-small"
]

def run_bench():
    results = []
    
    for m_id in OPEN_SOURCE_MODELS:
        print(f"\n正在加载模型: {m_id}")
        try:
            # 加载并显式搬运到 GPU
            model = SentenceTransformer(m_id, device=device, trust_remote_code=True)
            
            for cat, pairs in test_suites.items():
                for p in pairs:
                    # 获取向量 (计算在 GPU 上完成)
                    v_en = model.encode(p['en'], convert_to_tensor=True)
                    v_zh = model.encode(p['zh'], convert_to_tensor=True)
                    v_ja = model.encode(p['ja'], convert_to_tensor=True)
                    
                    # 计算跨语言相似度
                    sim_zh = util.cos_sim(v_en, v_zh).item()
                    sim_ja = util.cos_sim(v_en, v_ja).item()
                    
                    results.append({
                        "Model": m_id.split('/')[-1],
                        "Category": cat,
                        "Align_Score": (sim_zh + sim_ja) / 2
                    })
            
            # 手动释放显存，防止模型堆叠导致 OOM
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"跳过模型 {m_id}，错误: {e}")

    # 生成报表
    df = pd.DataFrame(results)
    pivot = df.pivot_table(index="Model", columns="Category", values="Align_Score")
    
    # 核心指标：对齐衰减 (Standard 减去 Slur)
    if "Standard_Baseline" in pivot.columns and "Slur_Alignment" in pivot.columns:
        pivot['Gap (Sensitivity)'] = pivot['Standard_Baseline'] - pivot['Slur_Alignment']
    
    print("\n评估结果汇总:")
    print(pivot.sort_values(by="Gap (Sensitivity)", ascending=False))
    pivot.to_csv("benchmark_results.csv")

if __name__ == "__main__":
    run_bench()