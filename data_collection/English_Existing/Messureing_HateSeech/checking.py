from datasets import load_dataset
import sys
from collections import Counter

# 保证输出 UTF-8，避免 Windows PowerShell 打印乱码
sys.stdout.reconfigure(encoding='utf-8')


ds = load_dataset("ucberkeley-dlab/measuring-hate-speech")


for split_name, split_data in ds.items():
    print(f"=== Split: {split_name} ===")
    # 查看字段
    print("字段 (columns):", split_data.column_names)
    
    # 打印前5条示例
    print("\n前5条示例:")
    for i, item in enumerate(split_data[:5]):
        print(f"{i+1}: {item}")
    
    # 如果有 label 字段，统计分布
    if "label" in split_data.column_names:
        labels = [l for l in split_data["label"]]
        label_counts = Counter(labels)
        print("\n标签分布:")
        for label, count in label_counts.items():
            print(f"{label}: {count}")
    print("\n" + "="*40 + "\n")