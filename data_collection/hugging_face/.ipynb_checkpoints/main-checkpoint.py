import json
import pandas as pd
from enum import Enum
from datasets import load_dataset
from tqdm import tqdm

# 1. 定义数据集配置 Enum
# 格式: (HuggingFace路径, 文本列名, 语言缩写, 子集/Config名称)
# 如果不需要子集，第四项填 None
class DatasetTask(Enum):
    # --- 中文数据集 ---
    ZHIHU        = ("liyucheng/zhihu_26k", 'RESPONSE', 'zh', None)
    WEIBO_1      = ("vilarin/weibo-2014", "text", 'zh', None)
    WEIBO_MOBILE = ('m4rque2/weibo_automobile', 'text', 'zh', None)
    WEIBO_LONG   = ('Giacinta/weibo', 'longtext_version', 'zh', None)
    
    # --- 日文数据集 ---
    LIVEDOOR_JP  = ("cl-tohoku/livedoor-news-corpus", "title", "ja", None)
    # 示例：需要子集处理的日文数据集
    JP_SENTIMENT = ("tyqiangz/multilingual-sentiments", "text", "ja", "japanese")

class UniversalKeywordPipeline:
    def __init__(self, keywords_file, limit_per_lang=100000):
        self.limit = limit_per_lang
        self.keywords_data = self._load_keywords(keywords_file)
        
        # 语言缩写到 JSON key 的映射
        self.lang_map = {
            "zh": "Chinese",
            "en": "English",
            "ja": "Japanese"
        }
        
        # 按语言分类存储的列表
        self.storage = {
            "Chinese": [],
            "English": [],
            "Japanese": []
        }

    def _load_keywords(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def run_all(self):
        # 遍历所有的 DatasetTask
        for task in DatasetTask:
            # 解构四元组
            dataset_path, column, lang_code, config_name = task.value
            json_key = self.lang_map.get(lang_code)
            
            if not json_key or json_key not in self.keywords_data:
                print(f"跳过 {dataset_path}: 语言配置错误。")
                continue

            keywords = self.keywords_data[json_key]
            current_lang_list = self.storage[json_key]
            
            # 检测该语言总额度是否已满
            if len(current_lang_list) >= self.limit:
                print(f"[{json_key}] 已达上限 {self.limit}，跳过数据集: {dataset_path}")
                continue

            print(f"\n>>> 正在处理: {dataset_path} | 子集: {config_name} | 语言: {json_key}")
            
            try:
                # 统一加载逻辑：动态传入 name (config_name)
                # streaming=True 保证不会爆内存
                ds = load_dataset(dataset_path, name=config_name, split="train", streaming=True)
                
                # 遍历流式数据
                for entry in tqdm(ds, desc=f"Scanning {dataset_path[:15]}...", unit=" lines"):
                    text = str(entry.get(column, ""))
                    
                    if not text:
                        continue

                    # 关键词匹配 (any 逻辑)
                    if any(kw in text for kw in keywords):
                        current_lang_list.append({
                            "dataset": dataset_path,
                            "content": text.replace("\n", " ").strip(),
                            "lang": json_key
                        })
                    
                    # 实时检测限额，一旦满额立即中止当前数据集并检查下一语言
                    if len(current_lang_list) >= self.limit:
                        print(f"\n[!] {json_key} 收集已达限额 ({self.limit})。")
                        break
                        
            except Exception as e:
                print(f"加载数据集 {dataset_path} 失败: {e}")

        self._save_results()

    def _save_results(self):
        all_data = []
        for lang_key, data_list in self.storage.items():
            if data_list:
                # 每个语言也可以单独存一个备查
                # pd.DataFrame(data_list).to_csv(f"filtered_{lang_key}.csv", index=False, encoding="utf-8-sig")
                all_data.extend(data_list)
        
        if all_data:
            df = pd.DataFrame(all_data)
            output_file = "filtered_social_media_data.csv"
            df.to_csv(output_file, index=False, encoding="utf-8-sig")
            print(f"\n✨ 任务全部完成！")
            print(f"总计捕获数据: {len(all_data)} 条")
            print(f"保存路径: {output_file}")
        else:
            print("\n❌ 未扫描到包含指定关键词的任何内容。")

# --- 执行管线 ---
if __name__ == "__main__":
    # 请确保路径正确：data_collection/final_keywords.json
    pipeline = UniversalKeywordPipeline("../final_keywords.json", limit_per_lang=100000)
    pipeline.run_all()