import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from dotenv import load_dotenv
from google import genai
import os, re, logging, asyncio
from google_api import APIRequester
from tqdm.asyncio import tqdm as atqdm
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# 设置日志
log_dir = 'data_augmentation/LLM/logs'
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, f'llm_augmentation_{timestamp}.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

load_dotenv()
# 1. 初始化设置
# 请替换为你的 Gemini API Key
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY")) 
# embedder 在需要时初始化


DATA_PATHS = {
    'zh': 'data_detect/finetuned_detection/chinese_final_religious_hate.csv',
    'en': 'data_augmentation/LLM/generated_texts/sampled_25_per_cluster.csv', # 暂时注释，专注于中日
    'jp': 'data_detect/finetuned_detection/japanese_final_religious_hate.csv'
}
TOTAL_TARGET = 10  # 你想为每个语种生成的总条数，先设 10 测试
BATCH_SIZE = 5    # 每次请求 Gemini 生成多少条（建议 5-10 条，多了质量会下降）
TEXT_COLUMN = 'text'
OUTPUT_DIR = 'data_augmentation/LLM'
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

生成された5件のテキストのリストを直接出力してください。各テキストは1行とし、分析プロセスは出力しないでください。""",
    'en' : """
### Role
You are an expert in cross-cultural linguistics and online subcultures, specializing in religious discourse and its hostile expressions in both Western and {target_lang} contexts.

### Task
Your mission is to perform a "Cultural Alignment" of the following English religious hate speech into {target_lang}. Do NOT perform a literal translation. Instead, perform a **Dynamic Equivalence** translation that feels like it was originally written by a native speaker of {target_lang}.

### Source Text (English):
"{text}"

### Core Instructions:
1. **Identify the Attack Logic:** Analyze the underlying grievance in the source (e.g., financial exploitation, hypocrisy, erosion of traditional values, or logical absurdity).
2. **Cultural Transposition:** - If the text mentions US-specific concepts (e.g., "Mega-churches", "Tax-exempt status", "Bible Belt"), replace them with equivalent {target_lang} negative tropes (e.g., ZH: "传销式教会/洗脑班", JP: "霊感商法/マインドコントロール/新興宗教の搾取").
3. **Linguistic Texture:** - Use the slang, idioms, and sentence structures common in {target_lang} social media (e.g., Weibo/Douban for ZH, 2channel/X for JP).
   - Maintain the original's emotional intensity—if the source is crude, the output must be crude; if it is pseudo-intellectual, the output must be as well.
4. **Anti-AI Bias:** Avoid "polite" or "clinical" language. The result must sound like a human-written comment from a hostile online forum.

### Output Format:
Directly provide the localized text in {target_lang} without any introductory remarks or explanations.
"""
}


def save_result_realtime(result, output_file):
    """实时保存单条结果到CSV文件"""
    result_df = pd.DataFrame([result])
    
    if os.path.exists(output_file):
        # 文件存在，追加模式
        result_df.to_csv(output_file, mode='a', header=False, index=False, encoding='utf-8-sig')
    else:
        # 文件不存在，创建新文件
        result_df.to_csv(output_file, mode='w', header=True, index=False, encoding='utf-8-sig')
    
    logging.debug(f"Result saved to {output_file}")

def get_classic_diverse_seeds(texts, num_clusters=5, drop_ratio=0.1):
    """筛选经典且差异化的种子文本"""
    from sentence_transformers import SentenceTransformer
    
    # 初始化embedder
    embedder = SentenceTransformer('intfloat/multilingual-e5-small', device='cpu')
    
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

async def process_pipeline_async(languages_to_run):
    """主执行管线 - 支持不同语言的异步并发处理"""
    requester = APIRequester(client)
    
    for lang in languages_to_run:
        print(f"\n" + "="*50)
        print(f"🚀 正在处理语言: {lang.upper()}")
        print("="*50)
        logging.info(f"Starting processing for language: {lang.upper()}")
        
        try:
            if lang == 'en':
                await process_en_async(requester)
            else:
                await process_other_async(lang, requester)
        except Exception as e:
            logging.error(f"Error processing {lang}: {e}")
            print(f"❌ 处理{lang.upper()}时出错: {e}")
    
    # 输出最终统计
    stats = requester.concurrency_manager.get_stats()
    print(f"\n" + "="*50)
    print(f"📊 最终统计信息:")
    print(f"  成功请求数: {stats['success_count']}")
    print(f"  失败请求数: {stats['error_count']}")
    print(f"  限流次数: {stats['rate_limit_hits']}")
    print(f"  成功率: {stats['success_rate']*100:.2f}%")
    print(f"  最终并发数: {stats['current_concurrency']}")
    print(f"  最终背压倍数: {stats['backoff_multiplier']:.2f}x")
    print("="*50)
    
    logging.info(f"Pipeline completed. Stats: {stats}")

async def process_en_async(requester):
    """处理英文生成 - 异步并发版本，支持实时保存和进度显示"""
    logging.info("Starting EN processing (async)")
    
    path = DATA_PATHS.get('en')
    if not path or not os.path.exists(path):
        logging.error(f"File not found: {path}")
        print(f"❌ 找不到文件: {path}")
        return
        
    df = pd.read_csv(path)
    texts = df[TEXT_COLUMN].dropna().unique().tolist()
    
    progress_file = os.path.join(OUTPUT_DIR, 'progress_en.txt')
    last_index = 0
    result_count = 0
    
    if os.path.exists(progress_file):
        with open(progress_file) as f:
            last_index = int(f.read().strip())
        logging.info(f"Resuming from index {last_index}")
        print(f"从{last_index}条开始生成")
    
    # 准备所有请求任务
    all_requests = []
    for i, text in enumerate(texts[last_index:], start=last_index):
        for la in ['chinese', 'japanese']:
            final_prompt = PROMPTS['en'].format(text=text, target_lang=la)
            all_requests.append({
                'index': i,
                'language': la,
                'prompt': final_prompt,
                'text': text
            })
    
    out_file = f"{OUTPUT_DIR}/generated_texts/synthetic_output_en_1.csv"
    total_requests = len(all_requests)
    
    # 分批处理（每次批处理后保存进度）
    batch_size = 10
    with tqdm(total=total_requests, desc="EN处理进度", unit="请求") as pbar:
        for batch_start in range(0, len(all_requests), batch_size):
            batch_end = min(batch_start + batch_size, len(all_requests))
            batch = all_requests[batch_start:batch_end]
            
            prompts = [req['prompt'] for req in batch]
            task_names = [f"EN-{req['index']}-{req['language']}" for req in batch]
            
            logging.info(f"Processing EN batch: {batch_start}-{batch_end}/{total_requests}")
            
            responses = await requester.batch_request(prompts, task_names)
            
            # 实时处理和保存结果 - 只在成功时更新
            batch_success_count = 0
            for req, response in zip(batch, responses):
                if isinstance(response, Exception):
                    logging.error(f"Failed to process {req['index']}-{req['language']}: {response}")
                    # ❌ 失败不更新进度条，不更新result_count
                else:
                    result = {
                        'prompt': req['prompt'],
                        'model_name': 'gemini-2.5-pro',
                        'response': response,
                        'language': req['language'],
                    }
                    # 实时保存结果
                    save_result_realtime(result, out_file)
                    result_count += 1
                    batch_success_count += 1
                    # ✅ 只在成功时更新进度条
                    pbar.update(1)
            
            # ✅ 只在本批次有成功结果时才更新进度文件
            if batch_success_count > 0:
                with open(progress_file, 'w') as f:
                    f.write(str(batch_end // 2 + last_index))
                logging.info(f"EN batch progress saved: {batch_end // 2 + last_index} ({batch_success_count} successes)")
            
            await asyncio.sleep(0.5)
    
    if result_count > 0:
        logging.info(f"EN processing completed. Saved {result_count} results to {out_file}")
        print(f"\n✅ EN 处理完成。成功生成 {result_count} 条数据，保存至 {out_file}")
    else:
        print("\n❌ EN 未能生成任何数据。")
        logging.warning("EN: No successful results generated")

async def process_other_async(lang, requester):
    """处理中文和日文 - 异步并发版本，支持实时保存和进度显示"""
    logging.info(f"Starting {lang.upper()} processing (async)")
    
    path = DATA_PATHS.get(lang)
    if not path or not os.path.exists(path):
        logging.error(f"File not found: {path}")
        print(f"❌ 找不到文件: {path}，跳过。")
        return
        
    df = pd.read_csv(path)
    raw_texts = df[TEXT_COLUMN].dropna().unique().tolist()
    
    print(f"已加载 {len(raw_texts)} 条原始数据，正在提取经典种子...")
    seeds = get_classic_diverse_seeds(raw_texts, num_clusters=5)
    formatted_seeds = "\n".join([f"- {s}" for s in seeds])
    
    progress_file = os.path.join(OUTPUT_DIR, f'progress_{lang}.txt')
    current_count = 0
    result_count = 0
    
    if os.path.exists(progress_file):
        with open(progress_file) as f:
            current_count = int(f.read().strip())
        logging.info(f"Resuming {lang} from count {current_count}")
    
    # 准备所有请求任务
    all_prompts = []
    temp_count = current_count
    while temp_count < TOTAL_TARGET:
        num_to_gen = min(BATCH_SIZE, TOTAL_TARGET - temp_count)
        
        final_prompt = PROMPTS[lang].format(
            num_seeds=len(seeds), 
            seeds=formatted_seeds,
            batch_size=num_to_gen
        )
        all_prompts.append((final_prompt, temp_count, num_to_gen))
        temp_count += num_to_gen
    
    # 并发处理所有请求
    prompts_only = [p[0] for p in all_prompts]
    task_names = [f"{lang.upper()}-{i}" for i in range(len(all_prompts))]
    
    out_file = f"{OUTPUT_DIR}/synthetic_output_{lang}_{timestamp}.csv"
    total_items = TOTAL_TARGET - current_count
    
    print(f"准备为{lang.upper()}生成{total_items}条数据，将并发发送{len(all_prompts)}个请求...")
    logging.info(f"{lang}: Preparing {len(all_prompts)} concurrent requests for {total_items} items")
    
    responses = await requester.batch_request(prompts_only, task_names)
    
    # 处理所有响应并实时保存
    with tqdm(total=total_items, desc=f"{lang.upper()}处理进度", unit="条") as pbar:
        for (final_prompt, start_count, num_to_gen), response in zip(all_prompts, responses):
            if isinstance(response, Exception):
                logging.error(f"Failed to generate {start_count}-{start_count+num_to_gen}: {response}")
                # ❌ 失败不更新进度条和计数器
                continue
            
            raw_output = response
            lines = raw_output.split('\n')
            batch_items = []
            for line in lines:
                clean_line = re.sub(r'^[\d\.\-\*\s]+', '', line).strip()
                if len(clean_line) > 5:
                    batch_items.append(clean_line)
            
            # 只取需要的数量，并实时保存
            batch_success_count = 0
            for item in batch_items[:num_to_gen]:
                result = {
                    'language': lang,
                    'prompt': final_prompt,
                    'result': item
                }
                # 实时保存结果
                save_result_realtime(result, out_file)
                result_count += 1
                current_count += 1
                batch_success_count += 1
                # ✅ 只在成功时更新进度条
                pbar.update(1)
            
            # ✅ 只在本批次有成功结果时定期保存进度
            if batch_success_count > 0 and current_count % BATCH_SIZE == 0:
                with open(progress_file, 'w') as f:
                    f.write(str(current_count))
                logging.info(f"{lang}: Saved progress {current_count}/{TOTAL_TARGET} ({batch_success_count} successes in batch)")
    
    # ✅ 最终保存进度 - 只在有成功结果时保存
    if result_count > 0:
        with open(progress_file, 'w') as f:
            f.write(str(current_count))
        logging.info(f"Saved {lang} results to {out_file}. Total success: {result_count}")
        print(f"\n✅ {lang.upper()} 处理完成。成功生成 {result_count} 条数据，保存至 {out_file}")
    else:
        print(f"\n❌ {lang.upper()} 未能生成任何数据。")
        logging.warning(f"{lang}: No successful results generated")

# --- 提示：关于轻量级嵌入模型 ---
# 如果你的机器连 multilingual-e5-small 都卡
# 建议在 get_classic_diverse_seeds 内部直接改用这个：
# from sklearn.feature_extraction.text import TfidfVectorizer
# vectorizer = TfidfVectorizer(max_features=1000, analyzer='char_wb', ngram_range=(2,4))
# embeddings = vectorizer.fit_transform(texts).toarray()
# 这种方法不吃显存，不占内存，速度极快，且在宗教词汇识别上效果极佳。

if __name__ == "__main__":
    # 你可以在这里控制走全部管线还是只走特定语言
    languages = ['en']  # 可改为 ['en', 'zh', 'jp'] 处理多个语言
    asyncio.run(process_pipeline_async(languages))