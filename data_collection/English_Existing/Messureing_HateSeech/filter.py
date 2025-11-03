from datasets import load_dataset
import csv
import sys

# 保证输出 UTF-8，避免 Windows PowerShell 打印乱码
sys.stdout.reconfigure(encoding='utf-8')

# 加载数据集
ds = load_dataset("ucberkeley-dlab/measuring-hate-speech")

# 输出 CSV 文件
output_csv = "christian_target_texts_all_splits.csv"

# 打开 CSV 文件写入
with open(output_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    # 写表头
    writer.writerow(["split", "text"])
    
    # 遍历每个 split
    for split_name, split_data in ds.items():
        # 筛选 target_religion_christian 为 True 的文本
        filtered_texts = [item["text"] for item in split_data if item.get("target_religion_christian")]
        
        print(f"{split_name}: 筛选到 {len(filtered_texts)} 条文本")
        
        # 写入 CSV
        for text in filtered_texts:
            writer.writerow([split_name, text])

print(f"已保存到 {output_csv}")
