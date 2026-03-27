import pandas as pd
import random
import os
import sys
import time
import logging
import threading
from google import genai
from google.genai import types
from dotenv import load_dotenv
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------- 环境与日志配置 ----------------
load_dotenv()

try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(os.path.dirname(current_dir)) 
    if root_dir not in sys.path:
        sys.path.append(root_dir)
    from scripts.set_logger import setup_logging
    logger, LOG_FILE_PATH = setup_logging(name='back_translation')
except ImportError:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('Gemini_Hate_Aug')

# ================= 配置区域 =================
CONFIG = {
    "input_file": "data_augmentation/back_translation/data/chinese.csv",
    "output_file": "data_augmentation/back_translation/data/back_translated_chinese.csv",
    "checkpoint_file": "data_augmentation/back_translation/data/chinese_checkpoint_progress.csv",
    "google_api_key": os.environ.get("GEMINI_API_KEY"), 
    "model_id": "gemini-2.5-flash", 
    "text_column": "text",
    "source_lang": "Chinese",       
    "target_lang": "English",    
    "max_workers": 20,             # 根据你的 RPM 1000, 10-15个并发是比较稳妥的
    "similarity_threshold": 0.75, 
    "local_sim_model": "paraphrase-multilingual-MiniLM-L12-v2",
    "aeda_punctuation": ["。", "，", "！", "？", "、", "；", "……", "·"],
}

# 线程锁，防止并行写入文件时冲突
file_lock = threading.Lock()

SAFETY_SETTINGS = [
    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
]

class GeminiHateTranslator:
    def __init__(self, config):
        self.config = config
        try:
            self.client = genai.Client(api_key=config["google_api_key"])
            logger.info(f"Gemini 客户端就绪: {config['model_id']}")
        except Exception as e:
            logger.error(f"Gemini 初始化失败: {e}")
            raise

    def translate_single(self, text, from_lang, to_lang):
        if not text or pd.isna(text) or str(text).strip() == "":
            return None
        
        prompt = f"""
        Instructions: You are a professional linguistic researcher analyzing hate speech. 
        Translate the following text strictly and faithfully from {from_lang} to {to_lang}. 
        Do not sanitize, do not mitigate offensive language, do not alter the sentiment. 
        If the original text contains slurs or hateful content, you MUST translate them with the equivalent offensive terms.

        Text to translate:
        "{text}"
        """

        try:
            response = self.client.models.generate_content(
                model=self.config["model_id"],
                contents=prompt,
                config=types.GenerateContentConfig(
                    safety_settings=SAFETY_SETTINGS,
                    temperature=0.2 
                )
            )
            if response and response.text:
                return response.text.strip().replace('"', '')
            return None
        except Exception as e:
            if "429" in str(e):
                time.sleep(10) # 触发限流，多歇一会儿
                return self.translate_single(text, from_lang, to_lang)
            logger.warning(f"Gemini 异常: {e}")
            return None

def aeda_augment(text, punctuations):
    if not isinstance(text, str) or len(text) < 2: return text
    chars = list(text)
    insert_count = max(1, int(len(chars) * random.uniform(0.1, 0.15)))
    for _ in range(insert_count):
        ins_idx = random.randint(0, len(chars))
        chars.insert(ins_idx, random.choice(punctuations))
    return "".join(chars)

def process_row_worker(row_tuple, translator, config):
    idx, row = row_tuple
    orig_text = str(row[config["text_column"]])
    
    # 执行双向翻译
    en_text = translator.translate_single(orig_text, config["source_lang"], config["target_lang"])
    ja_back = None
    if en_text:
        ja_back = translator.translate_single(en_text, config["target_lang"], config["source_lang"])
    
    result = {
        "index": idx,
        "original": orig_text,
        "en_intermediate": en_text,
        "back_translated": ja_back
    }

    # 使用线程锁实时写入 CSV 检查点
    with file_lock:
        res_df = pd.DataFrame([result])
        file_exists = os.path.isfile(config["checkpoint_file"])
        res_df.to_csv(config["checkpoint_file"], mode='a', index=False, header=not file_exists, encoding='utf-8-sig')
    
    return result

def run():
    if not os.path.exists(CONFIG["input_file"]):
        logger.error(f"找不到输入文件: {CONFIG['input_file']}")
        return

    df = pd.read_csv(CONFIG["input_file"])
    total_count = len(df)
    
    # --- 断点恢复逻辑 ---
    processed_indices = set()
    if os.path.exists(CONFIG["checkpoint_file"]):
        try:
            checkpoint_df = pd.read_csv(CONFIG["checkpoint_file"])
            # 只要 index 在 checkpoint 里出现过，不论是否翻译成功，都视为处理过
            processed_indices = set(checkpoint_df["index"].unique())
            logger.info(f"断点检测：已尝试处理 {len(processed_indices)} / {total_count} 条，跳过...")
        except Exception:
            logger.warning("检查点读取失败，将重新开始。")

    to_process = [row for row in df.iterrows() if row[0] not in processed_indices]

    if to_process:
        translator = GeminiHateTranslator(CONFIG)
        with ThreadPoolExecutor(max_workers=CONFIG["max_workers"]) as executor:
            futures = [executor.submit(process_row_worker, row, translator, CONFIG) for row in to_process]
            # 进度条显示总数 640 条
            for _ in tqdm(as_completed(futures), total=len(futures), desc="增强进度"):
                pass

    # --- 最终质检与合并 ---
    if not os.path.exists(CONFIG["checkpoint_file"]):
        logger.error("没有产生任何检查点数据。")
        return

    logger.info("所有翻译请求已完成，正在加载检查点进行语义质检...")
    all_res_df = pd.read_csv(CONFIG["checkpoint_file"])
    
    # 批量语义计算
    sim_model = SentenceTransformer(CONFIG["local_sim_model"])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sim_model.to(device)

    valid_mask = all_res_df["back_translated"].notna()
    if valid_mask.any():
        valid_rows = all_res_df[valid_mask]
        logger.info(f"开始计算 {len(valid_rows)} 条回译文本的语义相似度...")
        emb_orig = sim_model.encode(valid_rows["original"].tolist(), convert_to_tensor=True, show_progress_bar=True)
        emb_back = sim_model.encode(valid_rows["back_translated"].tolist(), convert_to_tensor=True, show_progress_bar=True)
        scores = torch.diagonal(util.cos_sim(emb_orig, emb_back)).tolist()
        all_res_df.loc[valid_mask, "similarity"] = scores
    else:
        all_res_df["similarity"] = 0.0

    # 组装最终结果
    final_rows = []
    orig_map = df.to_dict('index')

    # 最后组装阶段进度条
    for _, res in tqdm(all_res_df.iterrows(), total=len(all_res_df), desc="最终数据组装"):
        idx = int(res["index"])
        orig_text = res["original"]
        back_text = res["back_translated"]
        sim_score = res.get("similarity", 0.0)
        base_info = orig_map[idx]
        
        def create_entry(text, method):
            entry = base_info.copy()
            entry[CONFIG["text_column"]] = text
            entry["augment_method"] = method
            entry["group_id"] = idx
            return entry

        final_rows.append(create_entry(orig_text, "original"))
        final_rows.append(create_entry(aeda_augment(orig_text, CONFIG["aeda_punctuation"]), "original_AEDA"))

        if not pd.isna(back_text) and sim_score >= CONFIG["similarity_threshold"]:
            if back_text.strip() != orig_text.strip():
                final_rows.append(create_entry(back_text, f"BT_Gemini_Sim{round(sim_score, 2)}"))
                final_rows.append(create_entry(aeda_augment(back_text, CONFIG["aeda_punctuation"]), "BT_AEDA"))

    pd.DataFrame(final_rows).to_csv(CONFIG["output_file"], index=False, encoding='utf-8-sig')
    logger.info(f"✅ 增强完成！最终数据量: {len(final_rows)}")

if __name__ == "__main__":
    run()