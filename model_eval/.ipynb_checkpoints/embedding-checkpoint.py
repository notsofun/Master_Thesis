import torch
import gc
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# 强制使用 GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

test_suites = {
    "Standard_Baseline": [
        {"en": "Religious faith provides spiritual guidance.", 
         "zh": "宗教信仰提供精神指引。", 
         "ja": "宗教的な信仰は精神的な指針を与えます。"}
    ],
    "Slur_Alignment": [
        {"en": "That christcuck is just a brainwashed puppet.", 
         "zh": "那个基督狗不过是个被洗脑的傀儡。", 
         "ja": "あのキリスト厨はただの洗脳されたパペットだ。"}
    ],
    "Metaphor_Hate": [
        {"en": "Christianity is a tool for racial oppression and denying existence.", 
         "zh": "基督教实质已经成为种族压迫和否定生存的工具。", 
         "ja": "キリスト教は人種抑圧と存在否定のために使われる道具だ。"}
    ],
    "Deception_Hate": [
        {"en": "The doctrine is a fairy tale to cover up truths.", 
         "zh": "教义不过是为了掩盖真相而编造的诡辩。", 
         "ja": "教義は不都合な真実を誤魔化すための作り話だ。"}
    ]
}

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
    results = []
    for m_id in MODELS:
        print(f"Loading {m_id} to GPU...")
        try:
            # 1. 加载模型到 GPU
            model = SentenceTransformer(m_id, device=device, trust_remote_code=True)
            
            for cat, pairs in test_suites.items():
                for p in pairs:
                    # 2. 编码 (确保在 GPU 上)
                    v = model.encode([p['en'], p['zh'], p['ja']], convert_to_tensor=True)
                    sim_zh = util.cos_sim(v[0], v[1]).item()
                    sim_ja = util.cos_sim(v[0], v[2]).item()
                    
                    results.append({"Model": m_id, "Category": cat, "Avg_Sim": (sim_zh + sim_ja)/2})
            
            # 3. 严格显存管理
            del model
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            
        except Exception as e:
            print(f"Error loading {m_id}: {e}")

    # --- 修正后的数据处理逻辑 ---
    df = pd.DataFrame(results).pivot(index="Model", columns="Category", values="Avg_Sim")
    
    # 定义仇恨类字段
    hate_cols = ['Slur_Alignment', 'Metaphor_Hate', 'Deception_Hate']
    # 确保所有列都在 df 中，防止报错
    actual_hate_cols = [c for c in hate_cols if c in df.columns]
    
    # 计算均值：反映整体仇恨言论的跨语言对齐得分
    df['Hate_Mean'] = df[actual_hate_cols].mean(axis=1)
    
    # 计算核心 Gap: Standard 基准与仇恨类均值的差距
    # Gap 为正且越大，说明模型处理仇恨词汇时对齐越失败
    df['Gap'] = df['Standard_Baseline'] - df['Hate_Mean']
    
    # 计算相对于基准的跌幅 (%)
    df['Drop_Rate (%)'] = (df['Gap'] / df['Standard_Baseline']) * 100
    
    print("\n" + "="*80)
    print("宗教仇恨跨语言语义对齐评估汇总 (Sorted by Gap)")
    print("="*80)
    print(df.sort_values("Gap", ascending=False)[['Standard_Baseline', 'Hate_Mean', 'Gap', 'Drop_Rate (%)']])
    print("="*80)
    
    # 导出结果
    df.to_csv("hate_speech_alignment_report.csv")

if __name__ == "__main__":
    run_bench()