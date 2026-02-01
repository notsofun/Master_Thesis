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
    # ZHIHU        = ("liyucheng/zhihu_26k", 'RESPONSE', 'zh', None)
    # WEIBO_1      = ("vilarin/weibo-2014", "text", 'zh', None)
    # WEIBO_MOBILE = ('m4rque2/weibo_automobile', 'text', 'zh', None)
    # WEIBO_LONG   = ('Giacinta/weibo', 'longtext_version', 'zh', None)
    
    # --- 日文数据集 ---
    WRIME  = ("shunk031/wrime", "sentence", "ja", None)
    # 示例：需要子集处理的日文数据集
    JP_SENTIMENT = ("tyqiangz/multilingual-sentiments", "text", "ja", "japanese")
import json
import pandas as pd
from enum import Enum
from datasets import load_dataset, config
from tqdm import tqdm
import requests
import io

# 强制开启信任远程代码（针对支持该设置的 datasets 版本）
config.HF_DATASETS_TRUST_REMOTE_CODE = True

class DatasetTask(Enum):
    # 格式: (路径, 文本列名, 语言Key, 子集名/文件名)
    # WRIME 需要特殊处理，第四项填入其原始文件名
    WRIME        = ("shunk031/wrime", "sentence", "Japanese", "wrime-v1.tsv")
    JP_SENTIMENT = ("tyqiangz/multilingual-sentiments", "text", "Japanese", "japanese")
    # 如果有中文或其他，按此类推
    # WEIBO      = ("vilarin/weibo-2014", "text", "Chinese", None)

class UniversalKeywordPipeline:
    def __init__(self, keywords_file, limit_per_lang=100000):
        self.limit = limit_per_lang
        with open(keywords_file, 'r', encoding='utf-8') as f:
            self.keywords_data = json.load(f)
        
        self.storage = {"Chinese": [], "English": [], "Japanese": []}

    def run_all(self):
        for task in DatasetTask:
            dataset_path, column, json_key, config_or_file = task.value
            current_lang_list = self.storage[json_key]
            
            if len(current_lang_list) >= self.limit:
                continue

            keywords = self.keywords_data.get(json_key, [])
            print(f"\n>>> 正在处理: {dataset_path} | 语言: {json_key}")

            try:
                # --- 特殊情况：针对报错的 shunk031/wrime 走 Raw 下载通道 ---
                if dataset_path == "shunk031/wrime":
                    # 直接从 HF 仓库下载原始 TSV 文件
                    url = f"https://huggingface.co/datasets/{dataset_path}/resolve/main/{config_or_file}"
                    print(f"检测到旧版脚本，改用 Raw 通道下载: {url}")
                    
                    # 使用 chunksize 流式读取，防止 OOM
                    response = requests.get(url, stream=True)
                    if response.status_code == 200:
                        # 转换成 file-like 对象给 pandas
                        df_iter = pd.read_csv(io.BytesIO(response.content), sep='\t', chunksize=1000)
                        for chunk in tqdm(df_iter, desc="Scanning WRIME"):
                            for _, row in chunk.iterrows():
                                text = str(row.get(column, ""))
                                if text and any(kw in text for kw in keywords):
                                    current_lang_list.append({"dataset": dataset_path, "content": text.strip(), "lang": json_key})
                                if len(current_lang_list) >= self.limit: break
                            if len(current_lang_list) >= self.limit: break
                    else:
                        print(f"下载失败，状态码: {response.status_code}")

                # --- 正常情况：使用 datasets 库流式加载 ---
                else:
                    # 尝试正常加载，不传递 trust_remote_code 参数以避免某些版本报错
                    # 如果报错，load_dataset 内部会自动提示
                    ds = load_dataset(dataset_path, name=config_or_file, split="train", streaming=True)
                    
                    for entry in tqdm(ds, desc=f"Scanning {dataset_path[:15]}"):
                        text = str(entry.get(column, ""))
                        if text and any(kw in text for kw in keywords):
                            current_lang_list.append({"dataset": dataset_path, "content": text.strip(), "lang": json_key})
                        
                        if len(current_lang_list) >= self.limit:
                            print(f"\n[!] {json_key} 已达标。")
                            break
                            
            except Exception as e:
                print(f"跳过 {dataset_path}，加载出错: {e}")

        self._save_results()

    def _save_results(self):
        all_data = []
        for data_list in self.storage.values():
            all_data.extend(data_list)
        
        if all_data:
            df = pd.DataFrame(all_data)
            df.to_csv("filtered_social_media_data_JA.csv", index=False, encoding="utf-8-sig")
            print(f"\n✨ 任务完成！总计捕获: {len(all_data)} 条数据")
        else:
            print("\n❌ 未匹配到任何关键词数据。")

if __name__ == "__main__":
    # 确保 final_keywords.json 路径正确
    pipeline = UniversalKeywordPipeline("../final_keywords.json", limit_per_lang=100000)
    pipeline.run_all()
