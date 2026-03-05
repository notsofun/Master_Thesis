import pandas as pd
import random
import os
import sys, time
from tqdm import tqdm
from deep_translator import GoogleTranslator
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('SafeAug')

# ================= 配置区域 =================
CONFIG = {
    "input_file": "data_augmentation/back_translation/data/chinese.csv", 
    "output_file": "data_augmentation/back_translation/data/back_translated_chinese.csv",
    "text_column": "text",
    "label_columns": ["hate_speech", "christianity_related"], 
    "source_lang": "zh-CN", 
    # 修改点 1：仅保留英语回译，大幅降低语义漂移风险
    "back_trans_langs": ["en"], 
    "aeda_punctuation": ["。", "，", "！", "？", "...", "·"],
    "max_workers": (os.cpu_count() or 1) * 2 
}

class DataAugmenter:
    def __init__(self, config):
        self.config = config
        self.punctuations = config["aeda_punctuation"]
        
    def back_translate_logic(self, text, target_lang):
        """核心翻译逻辑：原 -> 目标 -> 原"""
        if not text or pd.isna(text) or str(text).strip() == "":
            return None
        try:
            # 中 -> 英
            translator = GoogleTranslator(source=self.config["source_lang"], target=target_lang)
            forward = translator.translate(text)
            
            # 增加极其微小的延迟，防止被谷歌封 IP
            time.sleep(0.2)
            
            # 英 -> 中
            back_translator = GoogleTranslator(source=target_lang, target=self.config["source_lang"])
            backward = back_translator.translate(forward)
            return backward
        except Exception as e:
            logger.warning(f"翻译出错: {e}")
            return None

    def aeda_augment(self, text):
        """AEDA: 随机插入标点符号（不会改变词义）"""
        if not isinstance(text, str) or len(text) < 2:
            return text
        words = list(text)
        # 插入比例控制在 10%-20%，避免标点过多导致乱码
        insert_count = max(1, int(len(words) * random.uniform(0.1, 0.2)))
        for _ in range(insert_count):
            ins_idx = random.randint(0, len(words))
            words.insert(ins_idx, random.choice(self.punctuations))
        return "".join(words)

def process_row(row, augmenter, config, group_id):
    """单行处理函数，由线程池调用"""
    original_text = str(row[config["text_column"]])
    
    labels_dict = {col: row[col] for col in config["label_columns"]}
    other_fields = {col: row[col] for col in row.index if col not in config["label_columns"] and col != config["text_column"] and col != 'group_id'}
    local_results = []

    def create_new_entry(new_text, method_name):
        entry = {config["text_column"]: new_text, "method": method_name, "group_id": group_id}
        entry.update(labels_dict)  
        entry.update(other_fields)  
        return entry

    # 1. 稳健回译 (仅中英中)
    for lang in config["back_trans_langs"]:
        bt_text = augmenter.back_translate_logic(original_text, lang)
        if bt_text and bt_text.strip() != original_text.strip():
            # 保存回译版
            local_results.append(create_new_entry(bt_text, f"BT_{lang}"))
            
            # 增加一个【回译 + AEDA】版，提高数据多样性
            aeda_bt = augmenter.aeda_augment(bt_text)
            local_results.append(create_new_entry(aeda_bt, f"BT_{lang}_AEDA"))

    # 2. 原始文本 AEDA 版
    aeda_orig = augmenter.aeda_augment(original_text)
    local_results.append(create_new_entry(aeda_orig, "AEDA_Only"))
    
    return local_results

def run():
    if not os.path.exists(CONFIG["input_file"]):
        print(f"Error: 未找到输入文件 {CONFIG['input_file']}")
        return

    df = pd.read_csv(CONFIG["input_file"])
    
    # 修改点 2：依然保持对 Hate 或 Religion 相关样本的筛选
    condition = (df[CONFIG["label_columns"]] == '是').any(axis=1)
    target_df = df[condition].copy()
    
    # 自动生成原始样本的唯一 ID，方便后续 GroupSplit
    target_df['group_id'] = range(len(target_df))

    print(f"🚀 开始增强。模式：中英中 + AEDA")
    print(f"📦 待处理正样本: {len(target_df)} 条")

    augmenter = DataAugmenter(CONFIG)
    new_rows = []
    original_rows = []

    # 任务并行执行
    with ThreadPoolExecutor(max_workers=CONFIG["max_workers"]) as executor:
        futures = [executor.submit(process_row, row, augmenter, CONFIG, g_id) 
                   for (_, row), g_id in zip(target_df.iterrows(), target_df['group_id'])]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="增强进度"):
            res = future.result()
            if res:
                new_rows.extend(res)

    # 整理原始正样本
    for _, row in target_df.iterrows():
        original_entry = row.to_dict()
        original_entry['method'] = 'original'
        original_rows.append(original_entry)

    # 整理不需要增强的样本 (负样本)
    non_target_df = df[~condition].copy()
    non_target_df['method'] = 'original'
    non_target_df['group_id'] = -1 # 负样本不设组 ID
    
    # 合并全量数据
    final_df = pd.concat([non_target_df, pd.DataFrame(original_rows), pd.DataFrame(new_rows)], ignore_index=True)
    
    # 最后的安全检查：去重
    final_df.drop_duplicates(subset=[CONFIG["text_column"]], keep='first', inplace=True)
    
    final_df.to_csv(CONFIG["output_file"], index=False, encoding='utf-8-sig')
    print(f"\n✅ 任务完成！")
    print(f"📊 原始样本: {len(df)} 条 -> 增强后: {len(final_df)} 条")
    print(f"💾 文件保存至: {CONFIG['output_file']}")

if __name__ == "__main__":
    run()