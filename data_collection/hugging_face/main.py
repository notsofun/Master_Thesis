import json
import pandas as pd
from enum import Enum
from datasets import load_dataset
from tqdm import tqdm

# 1. 定义数据集配置 Enum
class DatasetTask(Enum):
    # 格式: (HuggingFace路径, 文本列名, 语言缩写)
    ZHIHU = ("liyucheng/zhihu_26k", 'RESPONSE', 'zh')
    LIVEDOOR_JP  = ("cl-tohoku/livedoor-news-corpus", "title", "ja") # 示例：日文新闻标题

class UniversalKeywordPipeline:
    def __init__(self, keywords_file, limit_per_lang=100):
        self.limit = limit_per_lang
        self.keywords_data = self._load_keywords(keywords_file)
        
        # 建立语言缩写到 JSON key 的映射
        self.lang_map = {
            "zh": "Chinese",
            "en": "English",
            "ja": "Japanese"
        }
        
        # 用于存储最终结果的容器（按语言分类）
        self.storage = {
            "Chinese": [],
            "English": [],
            "Japanese": []
        }

    def _load_keywords(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def run_all(self):
        # 直接遍历所有的 DatasetTask
        for task in DatasetTask:
            dataset_path, column, lang_code = task.value
            json_key = self.lang_map.get(lang_code)
            
            if not json_key or json_key not in self.keywords_data:
                print(f"跳过 {dataset_path}: 找不到对应的语言关键词配置。")
                continue

            keywords = self.keywords_data[json_key]
            current_lang_list = self.storage[json_key]
            
            # 如果该语言已经收集够了，就跳过该语言的其他数据集
            if len(current_lang_list) >= self.limit:
                print(f"语言 {json_key} 已达限额，跳过数据集 {dataset_path}。")
                continue

            print(f"\n--- 正在处理数据集: {dataset_path} (语言: {json_key}) ---")
            
            try:
                # 使用 streaming=True 避免内存溢出
                ds = load_dataset(dataset_path, split="train", streaming=True)
                
                # 开始遍历数据
                for entry in ds:
                    text = str(entry.get(column, ""))
                    
                    # 关键词匹配
                    if any(kw in text for kw in keywords):
                        current_lang_list.append({
                            "dataset": dataset_path,
                            "content": text.replace("\n", " "), # 清理换行符
                            "lang": json_key
                        })
                    
                    # 检测限额
                    if len(current_lang_list) >= self.limit:
                        print(f"命中限额 ({self.limit})，中止当前语言的数据采集。")
                        break
            except Exception as e:
                print(f"加载数据集 {dataset_path} 失败: {e}")

        self._save_results()

    def _save_results(self):
        # 汇总所有列表到一个 DataFrame
        all_data = []
        for lang, data_list in self.storage.items():
            all_data.extend(data_list)
        
        if all_data:
            df = pd.DataFrame(all_data)
            df.to_csv("filtered_social_media_data.csv", index=False, encoding="utf-8-sig")
            print(f"\n任务完成！总计保存 {len(all_data)} 条数据到 filtered_social_media_data.csv")
        else:
            print("\n未搜寻到任何匹配数据。")

# --- 执行 ---
if __name__ == "__main__":
    # 初始化管线，设置每个语言最多保存 50 条
    pipeline = UniversalKeywordPipeline("data_collection/final_keywords.json", limit_per_lang=1)
    pipeline.run_all()