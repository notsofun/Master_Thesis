import os
import pandas as pd
import sys

# 保证输出 UTF-8，避免 Windows PowerShell 打印乱码
sys.stdout.reconfigure(encoding='utf-8')

# --------------------------
# 可自定义的部分：文件及其标签
# --------------------------
files = {
    r"data_collection\English_Existing\HateXplain_Data\filtered_posts.csv": "HateXplain",
    r"data_collection\English_Existing\Jigsaw\christian_comments.csv": "Jigsaw",
    r"data_collection\English_Existing\Messureing_HateSeech\christian_target_texts_all_splits.csv": "Messuring",
    r"data_collection\English_Existing\MLMA\christian_tweets.csv": "MLMA"
}

dfs = []

for file, label in files.items():
    if not os.path.exists(file):
        print(f"文件未找到：{file}，已跳过。")
        continue

    try:
        # 自动检测分隔符
        with open(file, "r", encoding="utf-8", errors="ignore") as f:
            sample = f.read(2048)
        sep = "," if sample.count(",") >= sample.count("\t") else "\t"

        # 尝试读取文件，最多两列
        df = pd.read_csv(
            file,
            sep=sep,
            header=0,
            usecols=[0, 1],        # 只保留前两列
            on_bad_lines="skip",   # 跳过损坏行
            encoding_errors="ignore"
        )

        if df.shape[1] < 2:
            print(f"文件 {file} 列数不足 2，已跳过。")
            continue

        # 统一列名
        df.columns = ["post_id", "text"]

        # 去掉空文本
        df = df.dropna(subset=["text"])
        df["text"] = df["text"].astype(str).str.strip()

        # 增加来源列
        df["source"] = label

        dfs.append(df)
        print(f"成功读取：{file}（{len(df)} 行）")

    except Exception as e:
        print(f"读取文件 {file} 出错：{e}")

if not dfs:
    raise ValueError("没有成功读取的文件，请检查路径或格式。")

# --------------------------
# 合并与去重
# --------------------------
merged = pd.concat(dfs, ignore_index=True)
merged = merged.drop_duplicates(subset="text", keep="first")

# --------------------------
# 输出结果
# --------------------------
output_file = "merged_deduped.csv"
merged.to_csv(output_file, index=False, encoding="utf-8-sig")

print(f"\n合并完成：共 {len(merged)} 条文本，输出到 {output_file}")
