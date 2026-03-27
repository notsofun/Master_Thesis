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
    logger, LOG_FILE_PATH = setup_logging()
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
    # 定义多个增强路径，以最大化变体数量
    "target_paths": [
        ["English"],               # 路径1: ZH -> EN -> ZH
        ["Korean"],                # 路径2: ZH -> KO -> ZH
        ["French"],
        ['German']
    ],
    "max_workers": 100,             # 匹配 RPM 1000 的并行度
    "similarity_threshold": 0.80,  # 稍微放宽阈值以保留更多变体，但仍需质检
    "local_sim_model": "paraphrase-multilingual-MiniLM-L12-v2",
    "aeda_punctuation": ["。", "，", "！", "？", "、", "；", "……", "·"],
    # "aeda_punctuation": ["。", "、", "！", "？", "...", "·"], #日语标点
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
                    temperature=0.3 # 略微增加温度以获取更多样的翻译
                )
            )
            if response and response.text:
                return response.text.strip().replace('"', '')
            return None
        except Exception as e:
            if "429" in str(e):
                time.sleep(10)
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
    
    results = []
    
    # 针对每一个路径进行转译
    for path in config["target_paths"]:
        current_text = orig_text
        path_name = "ZH"
        
        # 正向转译链
        success = True
        for lang in path:
            translated = translator.translate_single(current_text, path_name, lang)
            if translated:
                current_text = translated
                path_name = lang
            else:
                success = False
                break
        
        # 反向转译回中文
        if success:
            final_back = translator.translate_single(current_text, path_name, config["source_lang"])
            if final_back:
                res_entry = {
                    "index": idx,
                    "original": orig_text,
                    "back_translated": final_back,
                    "path": "->".join(path)
                }
                # 实时保存到检查点
                with file_lock:
                    res_df = pd.DataFrame([res_entry])
                    file_exists = os.path.isfile(config["checkpoint_file"])
                    res_df.to_csv(config["checkpoint_file"], mode='a', index=False, header=not file_exists, encoding='utf-8-sig')
                results.append(res_entry)
    
    return results

def run():
    if not os.path.exists(CONFIG["input_file"]):
        logger.error(f"找不到输入文件: {CONFIG['input_file']}")
        return

    df = pd.read_csv(CONFIG["input_file"])
    
    # --- 核心修改：只扩展宗教仇恨文本 ---
    # 假设标签列名为 'hate_speech' 和 'christianity_related' (或其它宗教标签)
    # 如果你的列名不同，请在此修改逻辑
    mask = (df['hate_speech'] == 1) & (df['christianity_related'] == 1)
    target_df = df[mask].copy()
    total_count = len(target_df)
    
    if total_count == 0:
        logger.warning("未发现符合‘宗教+仇恨’双重条件的样本。请检查标签列。")
        return

    # --- 断点恢复逻辑 ---
    processed_indices = set()
    if os.path.exists(CONFIG["checkpoint_file"]):
        try:
            checkpoint_df = pd.read_csv(CONFIG["checkpoint_file"])
            # 由于一个 index 对应多个 path，我们统计处理完所有 path 的 index
            counts = checkpoint_df["index"].value_counts()
            # 如果某个 index 的记录数等于路径数，说明该样本已增强完成
            processed_indices = set(counts[counts >= len(CONFIG["target_paths"])].index)
            logger.info(f"断点检测：已完全处理 {len(processed_indices)} / {total_count} 条宗教仇恨样本。")
        except Exception as e:
            logger.warning(f"检查点读取失败: {e}，将重新开始。")

    to_process = [row for row in target_df.iterrows() if row[0] not in processed_indices]

    if to_process:
        translator = GeminiHateTranslator(CONFIG)
        with ThreadPoolExecutor(max_workers=CONFIG["max_workers"]) as executor:
            futures = [executor.submit(process_row_worker, row, translator, CONFIG) for row in to_process]
            for _ in tqdm(as_completed(futures), total=len(futures), desc="多路径增强进度"):
                pass

    # --- 质检与合并 ---
    if not os.path.exists(CONFIG["checkpoint_file"]):
        logger.error("无可用检查点数据。")
        return

    logger.info("正在执行批量语义质检...")
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
    else:
        all_res_df["similarity"] = 0.0

    # 组装最终结果
    final_rows = []
    orig_map = df.to_dict('index')

    for _, res in tqdm(all_res_df.iterrows(), total=len(all_res_df), desc="整合最终数据集"):
        idx = int(res["index"])
        orig_text = res["original"]
        back_text = res["back_translated"]
        sim_score = res.get("similarity", 0.0)
        path_name = res.get("path", "unknown")
        
        base_info = orig_map[idx]
        
        def create_entry(text, method):
            entry = base_info.copy()
            entry[CONFIG["text_column"]] = text
            entry["augment_method"] = method
            entry["group_id"] = idx
            return entry

        # 只要相似度达标且发生了变化，就收录
        if not pd.isna(back_text) and sim_score >= CONFIG["similarity_threshold"]:
            if back_text.strip() != orig_text.strip():
                final_rows.append(create_entry(back_text, f"BT_{path_name}_Sim{round(sim_score, 2)}"))
                final_rows.append(create_entry(aeda_augment(back_text, CONFIG["aeda_punctuation"]), f"BT_{path_name}_AEDA"))

    # 将原始数据也加回去（包含原始 AEDA）
    for idx in target_df.index:
        orig_text = target_df.loc[idx, CONFIG["text_column"]]
        final_rows.append({**target_df.loc[idx].to_dict(), "augment_method": "original", "group_id": idx})
        final_rows.append({**target_df.loc[idx].to_dict(), CONFIG["text_column"]: aeda_augment(orig_text, CONFIG["aeda_punctuation"]), "augment_method": "original_AEDA", "group_id": idx})

    # 合并非目标样本（保持原样）
    non_target_list = df[~mask].to_dict('records')
    for row in non_target_list:
        row["augment_method"] = "original"
        row["group_id"] = -1
        final_rows.append(row)

    pd.DataFrame(final_rows).drop_duplicates(subset=[CONFIG["text_column"]]).to_csv(CONFIG["output_file"], index=False, encoding='utf-8-sig')
    logger.info(f"✅ 宗教仇恨增强完成！最终规模: {len(final_rows)}")

if __name__ == "__main__":
    run()