import pandas as pd
import random
import os
import sys, time
from tqdm import tqdm
from deep_translator import GoogleTranslator
from concurrent.futures import ThreadPoolExecutor, as_completed

# 强制设置：解决多线程下打印日志错乱问题
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('BackTrans')

# ================= 配置区域 =================
CONFIG = {
    "input_file": "data_augmentation/back_translation/data/chinese.csv",
    "output_file": "data_augmentation/back_translation/data/back_translated_chinese.csv",
    "text_column": "text",
    # 修改点：定义需要保留和筛选的多个标签列
    "label_columns": ["hate_speech", "christianity_related"], 
    "source_lang": "zh-CN", 
    "back_trans_langs": ["en", "ja", "fr"], 
    "aeda_punctuation": ["。", "，", "！", "？", "...", "·"],
    "max_workers": (os.cpu_count() or 1) * 2 
}

class DataAugmenter:
    def __init__(self, config):
        self.config = config
        self.punctuations = config["aeda_punctuation"]
        
    def back_translate_logic(self, text, target_lang):
        """核心翻译逻辑：原 -> 目标 -> 原"""
        if not text or pd.isna(text):
            return None
        try:
            translator = GoogleTranslator(source=self.config["source_lang"], target=target_lang)
            forward = translator.translate(text)
            
            back_translator = GoogleTranslator(source=target_lang, target=self.config["source_lang"])
            backward = back_translator.translate(forward)
            return backward
        except Exception as e:
            return None

    def aeda_augment(self, text):
        """AEDA: 随机插入标点符号"""
        if not isinstance(text, str) or len(text) < 2:
            return text
        words = list(text)
        insert_count = max(1, int(len(words) * random.uniform(0.1, 0.2)))
        for _ in range(insert_count):
            ins_idx = random.randint(0, len(words))
            words.insert(ins_idx, random.choice(self.punctuations))
        return "".join(words)

def process_row(row, augmenter, config):
    """单行处理函数，供线程池调用"""
    original_text = str(row[config["text_column"]])
    
    # 提取所有标签的值
    labels_dict = {col: row[col] for col in config["label_columns"]}
    local_results = []

    def create_new_entry(new_text, method_name):
        entry = {config["text_column"]: new_text, "method": method_name}
        entry.update(labels_dict) # 合并标签数据
        return entry

    # 1. 批量处理回译
    for lang in config["back_trans_langs"]:
        bt_text = augmenter.back_translate_logic(original_text, lang)
        time.sleep(0.4) # 稍微降低频率防止被封
        if bt_text and bt_text != original_text:
            local_results.append(create_new_entry(bt_text, f"BT_{lang}"))
            
            # AEDA 增强回译结果
            aeda_bt = augmenter.aeda_augment(bt_text)
            local_results.append(create_new_entry(aeda_bt, f"BT_{lang}_AEDA"))

    # 2. 原始文本 AEDA
    aeda_orig = augmenter.aeda_augment(original_text)
    local_results.append(create_new_entry(aeda_orig, "AEDA_Only"))
    
    return local_results

def run():
    # 1. 加载数据
    if not os.path.exists(CONFIG["input_file"]):
        print(f"Error: 文件 {CONFIG['input_file']} 不存在")
        return

    df = pd.read_csv(CONFIG["input_file"])
    
    condition = (df[CONFIG["label_columns"]] == '是').any(axis=1)
    target_df = df[condition].copy()
    
    print(f"🚀 启动线程数: {CONFIG['max_workers']}")
    print(f"📦 筛选条件: {CONFIG['label_columns']} 中任一为 '是'")
    print(f"📦 待处理目标样本: {len(target_df)} 条")

    if len(target_df) == 0:
        print("⚠️ 没有找到符合条件的样本，请检查标签列名或内容。")
        return

    augmenter = DataAugmenter(CONFIG)
    new_rows = []

    # 2. 多线程执行
    with ThreadPoolExecutor(max_workers=CONFIG["max_workers"]) as executor:
        futures = [executor.submit(process_row, row, augmenter, CONFIG) for _, row in target_df.iterrows()]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="数据增强进度"):
            res = future.result()
            if res:
                new_rows.extend(res)

    # 3. 合并与保存
    aug_df = pd.DataFrame(new_rows)
    final_df = pd.concat([df, aug_df], ignore_index=True)
    
    # 根据文本去重，保留第一次出现的（通常是原数据）
    final_df.drop_duplicates(subset=[CONFIG["text_column"]], keep='first', inplace=True)
    
    final_df.to_csv(CONFIG["output_file"], index=False, encoding='utf-8-sig')
    print(f"✅ 任务完成！增强后总行数: {len(final_df)} (新增 {len(final_df) - len(df)} 条)")
    print(f"💾 结果已保存至: {CONFIG['output_file']}")

if __name__ == "__main__":
    run()