import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types
import os, re, time
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

load_dotenv()
# 1. 初始化设置
# 请替换为你的 Gemini API Key
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY")) 
# 如果机器实在跑不动，可以换成：'paraphrase-multilingual-MiniLM-L12-v2'
embedder = SentenceTransformer('intfloat/multilingual-e5-small', device='cpu')

DATA_PATHS = {
    'zh': 'data_detect/finetuned_detection/chinese_final_religious_hate.csv',
    # 'en': 'data_collection/English_Existing/merged_deduped.csv', # 暂时注释，专注于中日
    'jp': 'data_detect/finetuned_detection/japanese_final_religious_hate.csv'
}
TOTAL_TARGET = 10  # 你想为每个语种生成的总条数，先设 10 测试
BATCH_SIZE = 5    # 每次请求 Gemini 生成多少条（建议 5-10 条，多了质量会下降）
TEXT_COLUMN = 'text'
# 针对不同语言的特化 Prompt 模板
PROMPTS = {
    'zh': """你是一个数据合成专家。你的任务是分析提供的低资源宗教仇恨言论样本，并生成风格相似但内容全新的数据用于模型训练。
请严格遵循以下步骤：
1. 分析阶段：分析以下 {num_seeds} 个真实样本的修辞手法、攻击角度和情绪色彩。
2. 生成阶段：基于分析，生成 5 条全新的、针对基督教的中文仇恨言论。
3. 审查阶段：确保生成的文本不涉及真实暴力的直接威胁（符合API安全政策），但必须保持原有的意识形态偏见和敌意特征。

真实样本：
{seeds}

请直接输出包含生成的 5 条文本的列表，每条文本占一行，不要输出你的分析过程。""",
    
    'jp': """あなたはデータ合成の専門家です。提供された低リソースの宗教的ヘイトスピーチのサンプルを分析し、モデル学習用として、スタイルは似ているが内容は完全に新しいデータを生成してください。
以下の手順に厳密に従ってください：
1. 分析フェーズ：以下の {num_seeds} 個の実際のサンプルの修辞技法、攻撃の角度、感情的な色合いを分析してください。
2. 生成フェーズ：分析に基づき、キリスト教に対する全く新しい日本語のヘイトスピーチを5件生成してください。
3. 審査フェーズ：生成されたテキストが実際の暴力に対する直接的な脅迫を含まないこと（APIの安全ポリシーに準拠）を確認しつつ、元のイデオロギー的偏見と敵意の特徴を維持してください。

実際のサンプル：
{seeds}

生成された5件のテキストのリストを直接出力してください。各テキストは1行とし、分析プロセスは出力しないでください。"""
}


def get_classic_diverse_seeds(texts, num_clusters=5, drop_ratio=0.1):
    """筛选经典且差异化的种子文本"""
    # 1. 向量化
    embeddings = embedder.encode(texts)
    
    # 2. 计算全局质心，剔除离群点 (坏例子)
    centroid = np.mean(embeddings, axis=0)
    distances = np.linalg.norm(embeddings - centroid, axis=1)
    
    # 找到距离最近的 (1 - drop_ratio) 的数据索引
    threshold = np.percentile(distances, (1 - drop_ratio) * 100)
    valid_indices = np.where(distances <= threshold)[0]
    
    valid_texts = [texts[i] for i in valid_indices]
    valid_embeddings = embeddings[valid_indices]
    
    # 3. 在好数据中进行 K-Means 聚类，确保多样性
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(valid_embeddings)
    
    seeds = []
    # 从每个簇中找到最靠近该簇中心的“经典”文本
    for i in range(num_clusters):
        cluster_center = kmeans.cluster_centers_[i]
        cluster_indices = np.where(kmeans.labels_ == i)[0]
        
        # 计算簇内数据到簇中心的距离
        cluster_points = valid_embeddings[cluster_indices]
        dists_to_center = np.linalg.norm(cluster_points - cluster_center, axis=1)
        
        # 取距离簇中心最近的 1 条
        closest_idx = cluster_indices[np.argmin(dists_to_center)]
        seeds.append(valid_texts[closest_idx])
        
    return seeds

def process_pipeline(languages_to_run):
    """主执行管线 - 已增强数量控制与清洗功能"""
    
    for lang in languages_to_run:
        print(f"\n" + "="*30)
        print(f"🚀 正在处理语言: {lang.upper()}")
        print("="*30)
        
        path = DATA_PATHS.get(lang)
        if not path or not os.path.exists(path):
            print(f"❌ 找不到文件: {path}，跳过。")
            continue
            
        # 1. 加载与种子筛选
        df = pd.read_csv(path)
        # 确保去掉空值，并只取唯一值进行筛选
        raw_texts = df[TEXT_COLUMN].dropna().unique().tolist()
        
        print(f"已加载 {len(raw_texts)} 条原始数据，正在提取经典种子...")
        # 调用之前定义的聚类筛选函数 (假设已在外部定义)
        # 如果机器跑不动 Embedding，可以将 get_classic_diverse_seeds 里的 embedder 换成 TF-IDF
        seeds = get_classic_diverse_seeds(raw_texts, num_clusters=5)
        formatted_seeds = "\n".join([f"- {s}" for s in seeds])
        
        # 2. 循环生成直到达到目标数量
        generated_results = []
        current_count = 0
        
        while current_count < TOTAL_TARGET:
            # 计算本次需要生成的数量，避免最后一拨超量
            num_to_gen = min(BATCH_SIZE, TOTAL_TARGET - current_count)
            
            # 动态填充 Prompt
            final_prompt = PROMPTS[lang].format(
                num_seeds=len(seeds), 
                seeds=formatted_seeds,
                batch_size=num_to_gen
            )
            
            print(f"进度: {current_count}/{TOTAL_TARGET} | 正在请求生成 {num_to_gen} 条...")
            
            try:
                response = client.models.generate_content(
                    model='gemini-2.5-pro',
                    contents=final_prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.8,
                        # 可以在这里加入 safety_settings 来放宽对仇恨言论研究的拦截阈值
                    )
                )
                
                raw_output = response.text.strip()
                
                # --- 核心解析逻辑：将返回的块状文本切分为单条数据 ---
                # 匹配常见的列表格式：1. 内容, - 内容, * 内容
                lines = raw_output.split('\n')
                batch_items = []
                for line in lines:
                    clean_line = re.sub(r'^[\d\.\-\*\s]+', '', line).strip() # 去掉开头的序号或符号
                    if len(clean_line) > 5: # 过滤掉太短的废话
                        batch_items.append(clean_line)
                
                # 只取我们需要的那几条
                for item in batch_items[:num_to_gen]:
                    generated_results.append({
                        'language': lang,
                        'prompt': final_prompt, # 记录对应的 prompt 以便回溯
                        'result': item
                    })
                    current_count += 1
                
                # 稍微停顿一下，保护 API 频率
                time.sleep(1) 
                
            except Exception as e:
                print(f"⚠️ 生成过程中出错: {e}")
                print("尝试保存已生成的部分数据...")
                break # 遇错停止当前语种，进入保存环节
        
        # 3. 结果输出
        if generated_results:
            out_df = pd.DataFrame(generated_results)
            out_file = f"data_augmentation/LLM/synthetic_output_{lang}_{timestamp}.csv"
            out_df.to_csv(out_file, index=False, encoding='utf-8-sig')
            print(f"✅ {lang.upper()} 处理完成。成功生成 {len(generated_results)} 条数据，保存至 {out_file}")
        else:
            print(f"❌ {lang.upper()} 未能生成任何数据。")

# --- 提示：关于轻量级嵌入模型 ---
# 如果你的机器连 multilingual-e5-small 都卡
# 建议在 get_classic_diverse_seeds 内部直接改用这个：
# from sklearn.feature_extraction.text import TfidfVectorizer
# vectorizer = TfidfVectorizer(max_features=1000, analyzer='char_wb', ngram_range=(2,4))
# embeddings = vectorizer.fit_transform(texts).toarray()
# 这种方法不吃显存，不占内存，速度极快，且在宗教词汇识别上效果极佳。

if __name__ == "__main__":
    # 你可以在这里控制走全部管线还是只走特定语言
    languages = ['zh', 'jp'] 
    process_pipeline(languages)