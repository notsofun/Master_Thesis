#!/usr/bin/env python3
"""按代码常量配置切分训练集和验证集。
功能：
  1. 将 '是/否' 等标签统一转为 1/0
  2. 打印异常行及其 annotation_id
  3. 按照分层抽样切分训练/验证集
  4. 输出的 CSV 只包含 1/0 标签
"""
from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd

# ====== 配置区 ======
INPUT_CSV_PATHS = [
    "model_train/classifier/data/final_annotated_Japanese.csv",
    "model_train/classifier/data/final_annotated_Chinese.csv",
]
VAL_SIZE = 0.15
RANDOM_SEED = 42
HATE_COL = "hate_speech"
CHRIST_COL = "christianity_related"
ID_COL = "annotation_id"  # 用于定位的 ID 列

# ====== 逻辑实现 ======

def normalize_label(x):
    """将多种标签形式归一化为 0 或 1"""
    if pd.isna(x):
        return None
    s = str(x).strip().lower()
    if s in {"是", "yes", "1", "1.0", "true", "t"}:
        return 1
    if s in {"否", "no", "0", "0.0", "false", "f"}:
        return 0
    return None

def split_one_file(csv_path: str, val_size: float, random_seed: int, hate_col: str, christ_col: str):
    p = Path(csv_path)
    if not p.exists():
        print(f"跳过：文件不存在 -> {p}")
        return

    df = pd.read_csv(p)

    # 检查列是否存在
    if hate_col not in df.columns or christ_col not in df.columns:
        raise ValueError(f"文件 {p.name} 缺少必要字段: {hate_col} 或 {christ_col}")

    # 1. 创建临时数字列
    df["temp_hate"] = df[hate_col].apply(normalize_label)
    df["temp_christ"] = df[christ_col].apply(normalize_label)

    # 2. 检查并打印无效行
    bad_mask = df["temp_hate"].isna() | df["temp_christ"].isna()
    if bad_mask.any():
        print(f"\n" + "!"*40)
        print(f"❌ 文件 【{p.name}】 发现无法识别的数据行：")
        bad_df = df[bad_mask]
        if ID_COL in df.columns:
            print(f"{'Row Index':<10} | {ID_COL:<15} | {hate_col:<15} | {christ_col:<15}")
            for idx, row in bad_df.iterrows():
                print(f"{idx:<10} | {str(row[ID_COL]):<15} | {str(row[hate_col]):<15} | {str(row[christ_col]):<15}")
        else:
            print(bad_df[[hate_col, christ_col]])
        print("!"*40)
        raise ValueError("请修正以上异常数据后再继续。")

    # 3. 【核心修改】将原始列替换为数字 1/0
    df[hate_col] = df["temp_hate"].astype(int)
    df[christ_col] = df["temp_christ"].astype(int)
    
    # 删除临时列，保持 DataFrame 干净
    df = df.drop(columns=["temp_hate", "temp_christ"])

    # 4. 准备分层
    stratify_series = df[hate_col].astype(str) + "__" + df[christ_col].astype(str)

    try:
        train_df, val_df = train_test_split(
            df,
            test_size=val_size,
            random_state=random_seed,
            shuffle=True,
            stratify=stratify_series,
        )
    except ValueError as e:
        print(f"⚠️ {p.name} 联合分层失败（样本太少），回退为随机切分: {e}")
        train_df, val_df = train_test_split(
            df,
            test_size=val_size,
            random_state=random_seed,
            shuffle=True,
            stratify=None,
        )

    # 5. 保存结果
    out_train = p.with_name(p.stem + "_train.csv")
    out_val = p.with_name(p.stem + "_val.csv")

    train_df.to_csv(out_train, index=False)
    val_df.to_csv(out_val, index=False)

    print(f"\n✅ 处理完成: {p.name}")
    print(f"   - 训练集: {out_train.name} ({len(train_df)} 行)")
    print(f"   - 验证集: {out_val.name} ({len(val_df)} 行)")
    print(f"   - {hate_col} 比例: {train_df[hate_col].mean():.2%} (train) / {val_df[hate_col].mean():.2%} (val)")
    print(f"   - {christ_col} 比例: {train_df[christ_col].mean():.2%} (train) / {val_df[christ_col].mean():.2%} (val)")

if __name__ == "__main__":
    for path in INPUT_CSV_PATHS:
        split_one_file(path, VAL_SIZE, RANDOM_SEED, HATE_COL, CHRIST_COL)