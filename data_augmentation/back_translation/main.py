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
    "input_file": "data_augmentation/back_translation/data/japanese.csv",
    "output_file": "data_augmentation/back_translation/data/back_translated_japanese.csv",
    "text_column": "text",
    "label_column": "hate_speech",
    "source_lang": "ja", 
    "back_trans_langs": ["en", "zh-CN", "fr"], 
    "aeda_punctuation": ["。", "、", "！", "？", "...", "·"],
    # 并发控制：如果 CPU 8 核，这里就是 16 线程。
    # 建议不要超过 32，否则极易被 Google 封 IP。
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
            # 第一步：正向翻译
            translator = GoogleTranslator(source=self.config["source_lang"], target=target_lang)
            forward = translator.translate(text)
            
            # 第二步：回译
            back_translator = GoogleTranslator(source=target_lang, target=self.config["source_lang"])
            backward = back_translator.translate(forward)
            return backward
        except Exception as e:
            # 这里不打印详细报错以免刷屏，仅记录
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
    label = row[config["label_column"]]
    local_results = []

    # 1. 批量处理回译
    for lang in config["back_trans_langs"]:
        bt_text = augmenter.back_translate_logic(original_text, lang)
        time.sleep(0.5)
        if bt_text and bt_text != original_text:
            local_results.append({config["text_column"]: bt_text, config["label_column"]: label, "method": f"BT_{lang}"})
            # AEDA 增强回译结果
            aeda_bt = augmenter.aeda_augment(bt_text)
            local_results.append({config["text_column"]: aeda_bt, config["label_column"]: label, "method": f"BT_{lang}_AEDA"})

    # 2. 原始文本 AEDA
    aeda_orig = augmenter.aeda_augment(original_text)
    local_results.append({config["text_column"]: aeda_orig, config["label_column"]: label, "method": "AEDA_Only"})
    
    return local_results

def run():
    # 1. 加载数据
    if not os.path.exists(CONFIG["input_file"]):
        print(f"Error: 文件 {CONFIG['input_file']} 不存在")
        return

    df = pd.read_csv(CONFIG["input_file"])
    target_df = df[df[CONFIG["label_column"]] == '是'].copy()
    
    print(f"🚀 检测到 CPU 核数: {os.cpu_count()}, 启动线程数: {CONFIG['max_workers']}")
    print(f"📦 待处理正样本: {len(target_df)} 条")

    augmenter = DataAugmenter(CONFIG)
    new_rows = []

    # 2. 多线程执行
    with ThreadPoolExecutor(max_workers=CONFIG["max_workers"]) as executor:
        # 提交所有任务
        futures = [executor.submit(process_row, row, augmenter, CONFIG) for _, row in target_df.iterrows()]
        
        # 使用 tqdm 实时查看进度
        for future in tqdm(as_completed(futures), total=len(futures), desc="数据增强进度"):
            res = future.result()
            if res:
                new_rows.extend(res)

    # 3. 合并与保存
    aug_df = pd.DataFrame(new_rows)
    final_df = pd.concat([df, aug_df], ignore_index=True)
    final_df.drop_duplicates(subset=[CONFIG["text_column"]], inplace=True)
    
    final_df.to_csv(CONFIG["output_file"], index=False, encoding='utf-8-sig')
    print(f"✅ 任务完成！增强后总行数: {len(final_df)}")
    print(f"💾 结果已保存至: {CONFIG['output_file']}")

if __name__ == "__main__":
    run()