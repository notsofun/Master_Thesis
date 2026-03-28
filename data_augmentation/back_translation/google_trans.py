import pandas as pd
import random
import os
import sys
import time
import logging
import threading
import requests  # 使用最基础的请求库
from dotenv import load_dotenv
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

# ---------------- 日志配置 ----------------
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(os.path.dirname(current_dir)) 
    if root_dir not in sys.path:
        sys.path.append(root_dir)
    from scripts.set_logger import setup_logging
    logger, LOG_FILE_PATH = setup_logging(name='google_trans_requests')
except ImportError:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('GT_Requests_Aug')

# ================= 配置区域 =================
CONFIG = {
    "input_file": "data_augmentation/back_translation/data/japanese.csv",
    "output_file": "data_augmentation/back_translation/data/back_translated_japansese_gt.csv",
    "checkpoint_file": "data_augmentation/back_translation/data/ja_gt_checkpoint_progress.csv",
    # 只要这个 Key 是正确的，不需要任何 Google Cloud 配置
    "api_key": os.environ.get("GOOGLE_TRANSLATE_API_KEY"), 
    "text_column": "text",
    "source_lang": "ja", 
    "target_paths": [
        ["en"], ["ko"], ["fr"], ["de"]
    ],
    "max_workers": 20, # 你有 1000 RPM，可以放心开大
    "similarity_threshold": 0.70, 
    "local_sim_model": "paraphrase-multilingual-MiniLM-L12-v2",
    # "aeda_punctuation": ["。", "，", "！", "？", "、", "；", "……", "·"],
    "aeda_punctuation": ["。", "、", "！", "？", "...", "·"], #日语标点
}

file_lock = threading.Lock()

class GoogleHateTranslator:
    def __init__(self, config):
        self.api_key = config["api_key"]
        self.url = "https://translation.googleapis.com/language/translate/v2"
        if not self.api_key:
            logger.error("❌ 未在环境变量中找到 GOOGLE_TRANSLATE_API_KEY")
            raise ValueError("Missing API KEY")

    def translate_single(self, text, from_lang, to_lang):
        """通过 REST API 发送翻译请求，完全绕过 SDK 认证"""
        if not text or pd.isna(text) or str(text).strip() == "":
            return None
        
        # 语言代码映射，确保符合 Google 标准
        lang_map = {"Chinese": "zh-CN", "English": "en", "Korean": "ko", "French": "fr", "German": "de"}
        src = lang_map.get(from_lang, from_lang)
        dest = lang_map.get(to_lang, to_lang)

        params = {
            "q": text,
            "target": dest,
            "source": src,
            "key": self.api_key,
            "format": "text"
        }

        try:
            # 使用 POST 方法
            response = requests.post(self.url, params=params, timeout=10)
            res_data = response.json()

            if response.status_code == 200:
                return res_data['data']['translations'][0]['translatedText']
            elif response.status_code == 429:
                logger.warning("⚠️ 触发频率限制，稍后重试...")
                time.sleep(5)
                return self.translate_single(text, from_lang, to_lang)
            else:
                logger.error(f"❌ API 报错: {res_data}")
                return None
        except Exception as e:
            logger.error(f"🌐 网络请求异常: {e}")
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
    
    # 获取该 index 已完成的路径，用于断点续传
    results = []
    for path in config["target_paths"]:
        current_text = orig_text
        current_lang = config["source_lang"]
        
        success = True
        for target_lang in path:
            translated = translator.translate_single(current_text, current_lang, target_lang)
            if translated:
                current_text = translated
                current_lang = target_lang
            else:
                success = False
                break
        
        if success:
            final_back = translator.translate_single(current_text, current_lang, config["source_lang"])
            if final_back:
                res_entry = {
                    "index": idx,
                    "original": orig_text,
                    "back_translated": final_back,
                    "path": "->".join(path)
                }
                with file_lock:
                    res_df = pd.DataFrame([res_entry])
                    file_exists = os.path.isfile(config["checkpoint_file"])
                    res_df.to_csv(config["checkpoint_file"], mode='a', index=False, header=not file_exists, encoding='utf-8-sig')
                results.append(res_entry)
    return results

def run():
    if not os.path.exists(CONFIG["input_file"]):
        logger.error(f"找不到文件: {CONFIG['input_file']}")
        return

    df = pd.read_csv(CONFIG["input_file"])
    # 筛选宗教仇恨文本
    mask = (df['hate_speech'] == 1) & (df['christianity_related'] == 1)
    target_df = df[mask].copy()
    
    logger.info(f"🚀 开始处理宗教仇恨正样本，共计: {len(target_df)} 条")

    # 断点检测
    processed_indices = {}
    if os.path.exists(CONFIG["checkpoint_file"]):
        cp_df = pd.read_csv(CONFIG["checkpoint_file"])
        # 统计每个索引完成了多少条路径
        processed_indices = cp_df["index"].value_counts().to_dict()

    to_process = [row for row in target_df.iterrows() if processed_indices.get(row[0], 0) < len(CONFIG["target_paths"])]
    logger.info(f"📈 剩余待处理行数: {len(to_process)}")

    if to_process:
        translator = GoogleHateTranslator(CONFIG)
        with ThreadPoolExecutor(max_workers=CONFIG["max_workers"]) as executor:
            futures = [executor.submit(process_row_worker, row, translator, CONFIG) for row in to_process]
            for _ in tqdm(as_completed(futures), total=len(futures), desc="Google REST API 翻译"):
                pass

    # 后期质检与合并
    if not os.path.exists(CONFIG["checkpoint_file"]):
        logger.warning("⚠️ 检查点文件为空。")
        return

    logger.info("🧮 正在执行批量语义质检...")
    all_res_df = pd.read_csv(CONFIG["checkpoint_file"])
    sim_model = SentenceTransformer(CONFIG["local_sim_model"])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sim_model.to(device)

    valid_mask = all_res_df["back_translated"].notna()
    if valid_mask.any():
        valid_rows = all_res_df[valid_mask]
        emb_orig = sim_model.encode(valid_rows["original"].tolist(), convert_to_tensor=True, show_progress_bar=True)
        emb_back = sim_model.encode(valid_rows["back_translated"].tolist(), convert_to_tensor=True, show_progress_bar=True)
        scores = torch.diagonal(util.cos_sim(emb_orig, emb_back)).tolist()
        all_res_df.loc[valid_mask, "similarity"] = scores

    # 合并数据
    final_rows = []
    orig_map = df.to_dict('index')

    for _, res in tqdm(all_res_df.iterrows(), total=len(all_res_df), desc="最终数据整合"):
        idx = int(res["index"])
        orig_text = str(res["original"])
        back_text = str(res["back_translated"])
        sim_score = res.get("similarity", 0.0)
        path_name = res["path"]
        
        base_info = orig_map[idx]
        if sim_score >= CONFIG["similarity_threshold"] and back_text.strip() != orig_text.strip():
            final_rows.append({**base_info, CONFIG["text_column"]: back_text, "augment_method": f"GT_{path_name}", "group_id": idx})
            final_rows.append({**base_info, CONFIG["text_column"]: aeda_augment(back_text, CONFIG["aeda_punctuation"]), "augment_method": f"GT_{path_name}_AEDA", "group_id": idx})

    # 将原始数据也加回去（包含 AEDA）
    for idx, row in target_df.iterrows():
        orig_text = str(row[CONFIG["text_column"]])
        final_rows.append({**row.to_dict(), "augment_method": "original", "group_id": idx})
        final_rows.append({**row.to_dict(), CONFIG["text_column"]: aeda_augment(orig_text, CONFIG["aeda_punctuation"]), "augment_method": "original_AEDA", "group_id": idx})

    # 合并负样本（此处略，逻辑与之前一致）
    
    final_df = pd.DataFrame(final_rows).drop_duplicates(subset=[CONFIG["text_column"]])
    final_df.to_csv(CONFIG["output_file"], index=False, encoding='utf-8-sig')
    logger.info(f"✨ 增强完成！保存至: {CONFIG['output_file']}")

if __name__ == "__main__":
    run()