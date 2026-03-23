# python data_augmentation/LLM/redundancy_check.py
import os
import re
from tqdm import tqdm
import time
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from google import genai
from dotenv import load_dotenv
from google.genai import types
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
import torch
from collections import deque
import logging

# --- 环境配置 ---
os.environ["TOKENIZERS_PARALLELISM"] = "false" # 禁用分词器并行告警
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# --- 日志配置 ---
log_dir = "data_augmentation/LLM/logs"
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# 文件处理器（保存日志到文件）
log_file = os.path.join(log_dir, f"redundancy_check_{timestamp}.log")
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setLevel(logging.DEBUG)

# 控制台处理器（同时输出到控制台）
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# 日志格式
formatter = logging.Formatter(
    '[%(asctime)s] %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

# --- 核心配置 ---
load_dotenv()
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY")) 

# 自动检测GPU，如果可用就用GPU，否则用CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"向量模型运行设备: {device.upper()}")

# 本地轻量级模型，用于监控同质化
embedder = SentenceTransformer('intfloat/multilingual-e5-small', device=device)

# --- API限流控制器 ---
class RateLimiter:
    """确保不超过Google API的RPM限制（150/分钟）"""
    def __init__(self, rpm_limit=150):
        self.rpm_limit = rpm_limit
        self.request_times = deque()  # 存储最近60秒的请求时间
        self.min_interval = 60.0 / rpm_limit  # 请求之间的最小间隔（秒）
    
    def wait_if_needed(self):
        """检查是否需要等待以遵守RPM限制"""
        now = time.time()
        # 移除60秒外的旧请求记录
        while self.request_times and self.request_times[0] < now - 60:
            self.request_times.popleft()
        
        # 如果最近60秒已有150+请求，则等待
        if len(self.request_times) >= self.rpm_limit:
            sleep_time = 60 - (now - self.request_times[0]) + 0.5
            if sleep_time > 0:
                time.sleep(sleep_time)
                now = time.time()
                self.request_times.popleft()
        
        self.request_times.append(now)

rate_limiter = RateLimiter(rpm_limit=150)

DATA_PATHS = {
    'zh': 'data_detect/finetuned_detection/chinese_final_religious_hate.csv',
    'jp': 'data_detect/finetuned_detection/japanese_final_religious_hate.csv'
}

# 多视角生成：确保语义多样性
ANGLES = {
    'zh': ["传统文化冲突", "财务欺诈质疑", "社会生活骚扰", "教义逻辑矛盾", "历史罪恶纠葛"],
    'jp': ["日本の伝統破壊", "金銭的搾取", "マインドコントロール", "歴史的対立", "偽善者的な振る舞い"]
}

PROMPTS = {
    'zh': """你是一个数据合成专家。你的任务是分析提供的低资源宗教仇恨言论样本，并生成风格相似但内容全新的数据用于模型训练。
请严格遵循以下步骤：
1. 分析阶段：分析以下真实样本的修辞手法、攻击角度和情绪色彩。
2. 生成阶段：基于分析，生成{batch_size}条全新的、针对基督教的中文仇恨言论。
3. 审查阶段：确保生成的文本不涉及真实暴力的直接威胁（符合API安全政策），但必须保持原有的意识形态偏见和敌意特征。
注意：请从【{angle}】这个具体切入点进行创作，避免重复通用词汇，多使用生动的隐喻。

真实样本：
{formatted_seeds}

请直接输出包含生成的 5 条文本的列表，每条文本占一行，不要输出你的分析过程。""",
    
    'jp': """あなたはデータ合成の専門家です。提供された低リソースの宗教的ヘイトスピーチのサンプルを分析し、モデル学習用として、スタイルは似ているが内容は完全に新しいデータを生成してください。
以下の手順に厳密に従ってください：
1. 分析フェーズ：以下の実際のサンプルの修辞技法、攻撃の角度、感情的な色合いを分析してください。
2. 生成フェーズ：分析に基づき、キリスト教に対する全く新しい日本語のヘイトスピーチを{batch_size}件生成してください。
3. 審査フェーズ：生成されたテキストが実際の暴力に対する直接的な脅迫を含まないこと（APIの安全ポリシーに準拠）を確認しつつ、元のイデオロギー的偏見と敵意の特徴を維持してください。
注意事項：
「{angle}」とのポイントから生成し、一般的な語彙を繰り返せずに、もっと典型的なメタファーを使ってください。

実際のサンプル：
{formatted_seeds}

生成された5件のテキストのリストを直接出力してください。各テキストは1行とし、分析プロセスは出力しないでください。"""
}

# 实验节点
MILESTONES = [500, 1000, 2000, 4000]
STOP_THRESHOLD = 0.30  # 同质化率超过30%则停止

# --- 辅助函数 ---

def clean_text(text):
    """移除模型生成的列表编号、符号等噪声"""
    text = re.sub(r'^[\d\.\-\*\s]+', '', text)
    return text.strip()

def calculate_homogeneity(new_texts, existing_texts, existing_embs=None):
    """
    计算冗余率：新生成的文本中有多少比例与现有库高度相似 (>0.85)
    existing_embs: 可选的预计算的旧文本embeddings，用于缓存
    """
    if not existing_texts:
        return 0.0, None  # 返回冗余率和新文本embeddings
    
    # 只需要新文本的embeddings
    new_embs = embedder.encode(new_texts, show_progress_bar=False)
    
    # 如果没有提供旧embeddings，则计算
    if existing_embs is None:
        existing_embs = embedder.encode(existing_texts, show_progress_bar=False)
    
    # 计算新文本与旧文本库的相似度矩阵
    sim_matrix = cosine_similarity(new_embs, existing_embs)
    
    # 找到每条新文本在旧库中的最大相似度
    max_sims = np.max(sim_matrix, axis=1)
    
    # 统计相似度 > 0.85 的比例
    redundant_count = np.sum(max_sims > 0.85)
    h_rate = redundant_count / len(new_texts)
    
    return h_rate, new_embs

# --- 并行API调用函数 ---

def call_api_once(prompt):
    """单次API调用（用于并行执行）"""
    try:
        # 等待直到符合RPM限制
        rate_limiter.wait_if_needed()
        
        response = client.models.generate_content(
            model='gemini-2.5-pro',
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.9)
        )
        raw_lines = response.text.strip().split('\n')
        new_batch = [clean_text(l) for l in raw_lines if len(clean_text(l)) > 10]
        return new_batch, None
    except Exception as e:
        return [], str(e)

def generate_batch_parallel(prompts, max_workers=None):
    """
    并行发送多个API请求，智能选择worker数
    不需要指定max_workers，会根据prompt数量自动选择（通常3-4个）
    """
    if max_workers is None:
        # 根据prompt数量智能选择，但不超过4个（避免过度并行）
        max_workers = min(len(prompts), 4)
    
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(call_api_once, p) for p in prompts]
        for future in futures:
            batch, error = future.result()
            if error:
                logger.warning(f"API错误: {error}")
            results.extend(batch)
    return results

def run_smart_generation(lang):
    logger.info(f"开始处理 [{lang.upper()}] 管线...")
    
    # 1. 准备种子 (保持你原有的逻辑)
    df_raw = pd.read_csv(DATA_PATHS[lang])
    raw_texts = df_raw['text'].dropna().unique().tolist()
    seeds = raw_texts[:5] 
    formatted_seeds = "\n".join([f"- {s}" for s in seeds])

    generated_pool = [] 
    history_log = []
    output_file = f"data_augmentation/LLM/cumulative_synthetic_{lang}_{timestamp}.csv"
    
    # 缓存已有的embeddings（避免重复计算）
    cached_embeddings = None
    
    if os.path.exists(output_file):
        existing_df = pd.read_csv(output_file)
        generated_pool = existing_df['result'].tolist()
        logger.info(f"检测到已有数据，从第 {len(generated_pool)} 条开始累计...")
        # 预计算已有数据的embeddings
        if generated_pool:
            cached_embeddings = embedder.encode(generated_pool, show_progress_bar=False)

    # 2. 初始化 tqdm 进度条
    target_max = MILESTONES[-1]
    pbar = tqdm(total=target_max, desc=f"Lang: {lang}", unit="条", initial=len(generated_pool))

    # 3. 开始生成循环（使用并行API调用）
    parallel_requests = 4  # 每轮发送4个并行请求（速率限制器会确保不超过150 RPM）
    
    while len(generated_pool) < target_max:
        current_idx = len(generated_pool)
        
        # 准备多个prompt，进行并行调用
        prompts = []
        angles = []
        for _ in range(parallel_requests):
            angle = np.random.choice(ANGLES[lang])
            angles.append(angle)
            batch_size = 10
            
            prompt = PROMPTS[lang].format(
                formatted_seeds=formatted_seeds,
                angle=angle,
                batch_size=batch_size
            )
            prompts.append(prompt)
        
        # 并行发送请求（速率限制器会在API调用时进行控制）
        new_items = generate_batch_parallel(prompts)
        
        if not new_items:
            logger.error("API调用失败，停止生成")
            break
        
        actual_new_count = len(new_items)

        # 3. 里程碑检查与同质化监控
        h_rate_display = "N/A"
        if current_idx > 0 and current_idx % 100 == 0:
            h_rate, new_embs = calculate_homogeneity(new_items, generated_pool, cached_embeddings)
            h_rate_display = f"{h_rate:.1%}"
            
            if h_rate > STOP_THRESHOLD:
                logger.warning(f"冗余率过高({h_rate_display})，提前停止")
                break
        
        # 更新数据池
        generated_pool.extend(new_items)
        
        # 更新缓存（如果做了相似度检查，就有新的embeddings了）
        if current_idx > 0 and current_idx % 100 == 0:
            # 重新计算或者延迟缓存清除（保守方案）
            cached_embeddings = None  # 清除缓存，让下次重新计算（保证准确）
        
        # --- 更新进度条 ---
        pbar.update(actual_new_count)
        current_angle = angles[0] if angles else "N/A"
        pbar.set_postfix({
            "视角": current_angle,
            "冗余率": h_rate_display,
            "总计": len(generated_pool)
        })

        # 保存逻辑
        pd.DataFrame({'result': generated_pool}).to_csv(output_file, index=False, encoding='utf-8-sig')
        
        # 里程碑记录（优化：采样计算，而非完整矩阵）
        for m in MILESTONES:
            if len(generated_pool) >= m and not any(log['milestone'] == m for log in history_log):
                # 采样策略：随机选1000条计算相似度（而非全部4000条）
                sample_size = min(1000, len(generated_pool))
                sample_indices = np.random.choice(len(generated_pool), sample_size, replace=False)
                sample_texts = [generated_pool[i] for i in sample_indices]
                
                sample_embeddings = embedder.encode(sample_texts, show_progress_bar=False)
                avg_sim = np.mean(cosine_similarity(sample_embeddings))
                history_log.append({'milestone': m, 'avg_similarity': avg_sim})
                logger.info(f"达到里程碑 {m}！采样相似度: {avg_sim:.4f} (样本大小: {sample_size})")

    pbar.close()
    logger.info(f"{lang.upper()} 任务结束。最终生成数量: {len(generated_pool)}")
    return history_log

# 执行
if __name__ == "__main__":
    logger.info("="*60)
    logger.info("开始数据合成任务")
    logger.info(f"日志文件: {log_file}")
    logger.info("="*60)
    
    results = {}
    for lang in ['zh', 'jp']:
        results[lang] = run_smart_generation(lang)
    
    logger.info("="*60)
    logger.info("所有任务完成！")
    logger.info("="*60)