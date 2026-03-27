import pandas as pd
import random
import os
import time
import logging
import deepl 
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('SafeAug_DeepL')

# ================= 配置区域 =================
CONFIG = {
    "input_file": "data_augmentation/back_translation/data/japanese.csv", 
    "output_file": "data_augmentation/back_translation/data/back_translated_japanese.csv",
    "deepl_auth_key": "e026b4e3-39e1-4c67-bc0f-0720f202d377:fx", 
    "text_column": "text",
    "label_columns": ["hate_speech", "christianity_related"], 
    "source_lang": "JA",       
    "target_lang": "EN-US",    
    "aeda_punctuation": ["。", "、", "！", "？", "...", "·"],
    "max_workers": 5           
}

class DataAugmenter:
    def __init__(self, config):
        self.config = config
        self.punctuations = config["aeda_punctuation"]
        try:
            self.translator = deepl.Translator(config["deepl_auth_key"])
            logger.info("DeepL 客户端初始化成功")
        except Exception as e:
            logger.error(f"DeepL 初始化失败: {e}")
            raise

    def back_translate_logic(self, text):
        """JA -> EN-US -> JA"""
        if not text or pd.isna(text) or str(text).strip() == "":
            return None
        try:
            result_en = self.translator.translate_text(text, target_lang=self.config["target_lang"])
            time.sleep(0.1)
            result_ja = self.translator.translate_text(result_en.text, target_lang=self.config["source_lang"])
            return result_ja.text
        except Exception as e:
            logger.warning(f"翻译出错: {e}")
            return None

    def aeda_augment(self, text):
        if not isinstance(text, str) or len(text) < 2:
            return text
        words = list(text)
        insert_count = max(1, int(len(words) * random.uniform(0.1, 0.2)))
        for _ in range(insert_count):
            ins_idx = random.randint(0, len(words))
            words.insert(ins_idx, random.choice(self.punctuations))
        return "".join(words)

def process_row(row, augmenter, config, group_id):
    """处理单行并返回该组的所有增强版本"""
    original_text = str(row[config["text_column"]])
    labels_dict = {col: row[col] for col in config["label_columns"]}
    other_fields = {col: row[col] for col in row.index if col not in config["label_columns"] and col != config["text_column"] and col != 'group_id'}
    
    local_results = []

    def create_entry(new_text, method_name):
        entry = {config["text_column"]: new_text, "method": method_name, "group_id": group_id}
        entry.update(labels_dict)  
        entry.update(other_fields)  
        return entry

    # 1. 原始样本（必须包含相同的 group_id）
    local_results.append(create_entry(original_text, "original"))

    # 2. DeepL 回译
    bt_text = augmenter.back_translate_logic(original_text)
    if bt_text and bt_text.strip() != original_text.strip():
        local_results.append(create_entry(bt_text, "BT_DeepL_EN"))
        # 3. 回译 + AEDA
        local_results.append(create_entry(augmenter.aeda_augment(bt_text), "BT_DeepL_EN_AEDA"))

    # 4. 仅 AEDA
    local_results.append(create_entry(augmenter.aeda_augment(original_text), "AEDA_Only"))
    
    return local_results

def run():
    if not os.path.exists(CONFIG["input_file"]):
        print(f"Error: 未找到输入文件 {CONFIG['input_file']}")
        return

    df = pd.read_csv(CONFIG["input_file"])
    
    condition = (df[CONFIG["label_columns"]] == 1 ).any(axis=1)
    target_df = df[condition].copy()
    non_target_df = df[~condition].copy()

    print(f"🚀 切换至【稳健模式】：单线程串行处理 + 延时防封")
    print(f"📦 待处理正样本: {len(target_df)} 条")

    augmenter = DataAugmenter(CONFIG)
    augmented_rows = []

    # 改为单线程循环，避免并发触发 429 错误
    for g_id, (_, row) in enumerate(tqdm(target_df.iterrows(), total=len(target_df), desc="增强进度")):
        try:
            # 处理单行
            res = process_row(row, augmenter, CONFIG, g_id)
            if res:
                augmented_rows.extend(res)
            
            # 每处理完一条（包含两次翻译），强制休息 1-2 秒
            # 这是保证免费版 API 不报 429 的关键
            time.sleep(1.5) 
            
        except Exception as e:
            logger.error(f"处理第 {g_id} 组时发生不可预知的错误: {e}")
            time.sleep(5) # 出错多睡会儿
            continue

    # 处理负样本
    non_target_rows = []
    for _, row in non_target_df.iterrows():
        neg_entry = row.to_dict()
        neg_entry['method'] = 'original'
        neg_entry['group_id'] = -1 
        non_target_rows.append(neg_entry)
    
    # 合并并保存
    final_df = pd.concat([pd.DataFrame(non_target_rows), pd.DataFrame(augmented_rows)], ignore_index=True)
    final_df.drop_duplicates(subset=[CONFIG["text_column"]], keep='first', inplace=True)
    final_df.to_csv(CONFIG["output_file"], index=False, encoding='utf-8-sig')
    
    print(f"\n✅ 任务完成！")
    print(f"📊 增强后总条数: {len(final_df)}")
    print(f"💾 结果保存至: {CONFIG['output_file']}")

if __name__ == "__main__":
    run()