import os
import json
import pandas as pd
from enum import Enum
from datasets import load_dataset
from tqdm import tqdm

# --- 关键：解决“无法访问网站”的问题 ---
# 设置环境变量，让 datasets 库通过国内镜像站下载数据
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

class DatasetTask(Enum):
    # 现在你可以完全按照官方文档的 name 来写了
    WRIME        = ("shunk031/wrime", "sentence", "Japanese", "ver1")
    JP_SENTIMENT = ("tyqiangz/multilingual-sentiments", "text", "Japanese", "japanese")

class UniversalKeywordPipeline:
    def __init__(self, keywords_file, limit_per_lang=100000):
        self.limit = limit_per_lang
        with open(keywords_file, 'r', encoding='utf-8') as f:
            self.keywords_data = json.load(f)
        self.storage = {"Japanese": []}

    def run_all(self):
        for task in DatasetTask:
            dataset_path, column, json_key, config_name = task.value
            current_lang_list = self.storage[json_key]
            
            print(f"\n>>> 正在处理: {dataset_path} (子集: {config_name})")
            
            try:
                # 按照你要求的标准文档写法，增加 trust_remote_code=True 是因为这些旧数据集含脚本
                # streaming=True 建议保留，因为云服务器内存可能有限
                ds = load_dataset(
                    dataset_path, 
                    name=config_name, 
                    split="train", 
                    streaming=True, 
                    trust_remote_code=True
                )
                
                keywords = self.keywords_data.get(json_key, [])
                
                for entry in tqdm(ds, desc="Scanning"):
                    text = str(entry.get(column, ""))
                    if text and any(kw in text for kw in keywords):
                        current_lang_list.append({
                            "dataset": dataset_path,
                            "content": text.strip(),
                            "lang": json_key
                        })
                    
                    if len(current_lang_list) >= self.limit:
                        break
                        
            except Exception as e:
                print(f"加载 {dataset_path} 失败，原因: {e}")

        self._save_results()

    def _save_results(self):
        all_data = []
        for data_list in self.storage.values():
            all_data.extend(data_list)
        
        if all_data:
            df = pd.DataFrame(all_data)
            df.to_csv("filtered_social_media_data_JA.csv", index=False, encoding="utf-8-sig")
            print(f"✨ 成功！捕获 {len(all_data)} 条。")
        else:
            print("❌ 未能匹配到数据。")

if __name__ == "__main__":
    pipeline = UniversalKeywordPipeline("../final_keywords.json", limit_per_lang=100000)
    pipeline.run_all()