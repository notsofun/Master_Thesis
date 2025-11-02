import json
import csv
import sys

# 保证输出 UTF-8，避免 Windows PowerShell 打印乱码
sys.stdout.reconfigure(encoding='utf-8')

# ---------- 配置 ----------
input_json_file = "dataset.json"       # 数据集路径
output_csv_file = "filtered_posts.csv"  # 输出 CSV
target_filter = "Christian"           # 你要筛选的 target
# ---------------------------

# 读取 JSON 数据
with open(input_json_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# 获取全部 target 种类
all_targets = set()
for post in data.values():
    for annotator in post.get("annotators", []):
        all_targets.update(annotator.get("target", []))
print("全部 target 种类:", all_targets)

# 筛选符合 target 的 post_id，并获取原始文本
filtered_posts = []
for post_id, post in data.items():
    for annotator in post.get("annotators", []):
        if target_filter in annotator.get("target", []):
            text = " ".join(post.get("post_tokens", []))
            filtered_posts.append({"post_id": post_id, "text": text})
            break  # 找到就跳出

print(f"筛选到 {len(filtered_posts)} 条符合 target '{target_filter}' 的帖子")

# 保存到 CSV
with open(output_csv_file, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["post_id", "text"])
    writer.writeheader()
    for item in filtered_posts:
        writer.writerow(item)

print(f"已保存到 {output_csv_file}")
