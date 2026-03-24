# 数据合成脚本 (本地版本)
import os
import re
import time
import logging
import torch
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from datetime import datetime
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types

# --- 环境配置 ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()

# --- 路径配置 ---
OUTPUT_BASE = "data_augmentation/LLM"
LOG_DIR = os.path.join(OUTPUT_BASE, "logs")
TEXT_DIR = os.path.join(OUTPUT_BASE, "generated_texts")
os.makedirs(LOG_DIR, exist_ok=True)

# --- 日志配置 ---
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if logger.hasHandlers():
    logger.handlers.clear()

log_file = os.path.join(LOG_DIR, f"redundancy_check_{timestamp}.log")
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)-8s | %(message)s'))
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(message)s'))

logger.addHandler(file_handler)
logger.addHandler(console_handler)

# --- 设备检测与模型初始化 ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"向量模型运行设备: {device.upper()}")
embed_model = SentenceTransformer('intfloat/multilingual-e5-small', device=device)

# --- API 秘钥配置 ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.warning("未找到 GEMINI_API_KEY，请设置环境变量")

client = genai.Client(api_key=GEMINI_API_KEY)

# --- API 限流控制器 ---
class RateLimiter:
    """确保不超过 Google API 的 RPM 限制"""
    def __init__(self, rpm_limit=150):
        self.rpm_limit = rpm_limit
        self.request_times = deque()
    
    def wait_if_needed(self):
        now = time.time()
        while self.request_times and self.request_times[0] < now - 60:
            self.request_times.popleft()
        
        if len(self.request_times) >= self.rpm_limit:
            sleep_time = 60 - (now - self.request_times[0]) + 0.5
            if sleep_time > 0:
                time.sleep(sleep_time)
                now = time.time()
                self.request_times.popleft()
        
        self.request_times.append(now)

rate_limiter = RateLimiter(rpm_limit=150)

# --- 数据配置 ---
DATA_PATHS = {
    'zh': 'data_detect/finetuned_detection/chinese_final_religious_hate.csv',
    'jp': 'data_detect/finetuned_detection/japanese_final_religious_hate.csv'
}

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

# --- 实验参数 ---
MILESTONES = [20, 30, 50]
STOP_THRESHOLD = 0.30

# --- 全局变量 ---
pool_embeddings_tensor = None

# --- 辅助函数 ---

def clean_text(text):
    """移除模型生成的列表编号、符号等噪声"""
    text = re.sub(r'^[\d\.\-\*\s]+', '', text)
    return text.strip()

def calculate_homogeneity_e5_fast(new_texts, threshold=0.90):
    """
    极速版：只编码新文本，旧向量从全局缓存读取
    """
    global pool_embeddings_tensor
    
    # 1. 编码新文本
    new_prefixed = ["passage: " + t for t in new_texts]
    with torch.no_grad():
        new_embs = embed_model.encode(new_prefixed, convert_to_tensor=True, show_progress_bar=False)
        new_embs = torch.nn.functional.normalize(new_embs, p=2, dim=1)
    
    if pool_embeddings_tensor is None:
        # 第一次运行，初始化缓存
        pool_embeddings_tensor = new_embs
        return 0.0
    
    # 2. 计算新文本与缓存池的相似度 (Tensor 运算在 CPU/GPU 都极快)
    # new_embs: [batch, dim], pool: [N, dim] -> scores: [batch, N]
    sim_matrix = torch.matmul(new_embs, pool_embeddings_tensor.T).cpu().numpy()
    
    # 3. 计算冗余率
    max_sims = np.max(sim_matrix, axis=1)
    redundant_count = np.sum(max_sims > threshold)
    h_rate = redundant_count / len(new_texts)
    
    # 4. 更新缓存池
    pool_embeddings_tensor = torch.cat([pool_embeddings_tensor, new_embs], dim=0)
    
    return h_rate

# --- 并行 API 调用函数 ---

def call_api_once(prompt):
    """单次 API 调用（用于并行执行）"""
    try:
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

def generate_batch_parallel(prompts, max_workers=2):
    """并行发送多个 API 请求"""
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(call_api_once, p) for p in prompts]
        for future in futures:
            batch, error = future.result()
            if error:
                logger.warning(f"API 错误: {error}")
            results.extend(batch)
    return results

# --- 辅助模块 1：种子管理 ---

def get_random_seeds(raw_texts, n=5):
    """从原始数据中随机抽取并格式化种子"""
    sampled = np.random.choice(raw_texts, size=min(len(raw_texts), n), replace=False)
    return "\n".join([f"- {s}" for s in sampled])

# --- 辅助模块 2：数据落盘 ---

def save_to_disk(file_path, data_pool):
    """确保数据实时安全保存"""
    df = pd.DataFrame({'result': data_pool})
    df.to_csv(file_path, index=False, encoding='utf-8-sig')

# --- 核心逻辑：单步执行 ---

def execute_batch_step(lang, formatted_seeds, parallel_requests):
    """执行一次并行请求并返回清洗后的文本列表"""
    prompts = []
    # 随机选择本轮的视角
    current_angles = [np.random.choice(ANGLES[lang]) for _ in range(parallel_requests)]
    
    for i in range(parallel_requests):
        prompts.append(PROMPTS[lang].format(
            formatted_seeds=formatted_seeds,
            angle=current_angles[i],
            batch_size=5
        ))
    
    return generate_batch_parallel(prompts)

# --- 主生成函数 ---

def check_milestones(current_size, history_log):
    """简单的里程碑记录逻辑"""
    for m in MILESTONES:
        if current_size >= m and not any(log['milestone'] == m for log in history_log):
            history_log.append({'milestone': m, 'time': datetime.now().isoformat()})
            logger.info(f"📍 里程碑 {m} 已达成!")

def run_smart_generation(lang):
    """运行数据合成管线"""
    global pool_embeddings_tensor
    
    logger.info(f"\n>>> 开始处理 [{lang.upper()}] 管线")
    
    # 1. 基础准备
    if not os.path.exists(DATA_PATHS[lang]):
        logger.error(f"未找到数据集: {DATA_PATHS[lang]}")
        return []

    df_raw = pd.read_csv(DATA_PATHS[lang])
    raw_texts = df_raw['text'].dropna().unique().tolist()
    
    # 初始化变量
    generated_pool = []
    history_log = []
    pool_embeddings_tensor = None  # 切换语言重置向量缓存
    
    output_file = os.path.join(TEXT_DIR, f"synthetic_{lang}_{timestamp}.csv")
    target_max = MILESTONES[-1]
    pbar = tqdm(total=target_max, desc=f"Progress ({lang})")

    # 2. 初始种子
    formatted_seeds = get_random_seeds(raw_texts)
    
    # 3. 主循环
    while len(generated_pool) < target_max:
        old_pool_size = len(generated_pool)
        
        # --- 策略更新：每生成 10 条文本，彻底更换一次种子 ---
        if old_pool_size > 0 and old_pool_size % 10 == 0:
            formatted_seeds = get_random_seeds(raw_texts)
            pbar.write(f"🔄 已生成 {old_pool_size} 条，重新采样原始种子以保持多样性...")

        # 执行生成
        new_items = execute_batch_step(lang, formatted_seeds, parallel_requests=2)
        
        if not new_items:
            logger.error("API 调用失败，尝试等待...")
            time.sleep(10)
            continue
        
        # 冗余检查与缓存更新
        h_rate = calculate_homogeneity_e5_fast(new_items, threshold=0.90)
        h_rate_display = f"{h_rate:.1%}"

        # 入库与保存
        generated_pool.extend(new_items)
        save_to_disk(output_file, generated_pool)
        
        # 更新进度条
        pbar.update(len(new_items))
        pbar.set_postfix({
            "Redundancy": h_rate_display, 
            "Total": len(generated_pool)
        })
        logger.info(f"\n当前语义冗余度为: {h_rate_display}")

        # 4. 熔断逻辑
        if old_pool_size > 500 and h_rate > STOP_THRESHOLD:
            logger.warning(f"\n[!] 语义冗余率过高 ({h_rate_display})。停止生成。")
            break

        # 5. 里程碑检查
        check_milestones(len(generated_pool), history_log)

    pbar.close()
    logger.info(f"{lang.upper()} 任务结束。最终生成数量: {len(generated_pool)}")
    return history_log

# --- 主程序 ---

if __name__ == "__main__":
    logger.info("="*60)
    logger.info("LLM 数据合成任务启动 (语义模型: E5-small)")
    logger.info(f"运行设备: {device.upper()}")
    logger.info(f"日志文件: {log_file}")
    logger.info("="*60)
    
    results = {}
    for lang in ['zh', 'jp']:
        results[lang] = run_smart_generation(lang)
    
    logger.info("\n" + "="*60)
    logger.info("所有任务完成！")
    logger.info("="*60)
