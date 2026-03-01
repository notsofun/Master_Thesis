#!/usr/bin/env python3
"""
在同目录下对所有 finetuning csv 增加一列 `split` (train/val)，
尝试对 `hate_speech` 与 `christianity_related` 做联合分层抽样以保持分布一致。

用法示例:
  python add_split_column.py --val-size 0.1 --seed 42
  # 在脚本目录运行（默认会处理同目录下 *finetuning*.csv）
    python model_train/classifier/data/add_split_column.py

    # 自定义验证集比例并覆盖原文件
    python model_train/classifier/data/add_split_column.py --val-size 0.2 --inplace

    # 指定目录与随机种子
    python model_train/classifier/data/add_split_column.py --dir model_train/classifier/data --seed 123

输出: 为每个处理的文件写入同目录下的 `<orig>_with_split.csv`，可用 `--inplace` 覆盖原文件。
"""
import argparse
from pathlib import Path
import sys
import pandas as pd

def stratified_split(df, hate_col, christ_col, val_size, seed):
    from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

    # build combined label
    if hate_col not in df.columns or christ_col not in df.columns:
        raise KeyError(f"缺少列: {hate_col} 或 {christ_col}")

    combined = df[hate_col].fillna("NA").astype(str) + "__" + df[christ_col].fillna("NA").astype(str)

    # try combined stratify first
    try:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
        train_idx, val_idx = next(splitter.split(df, combined))
        idx_val = set(val_idx)
        mask = [i in idx_val for i in range(len(df))]
        return mask
    except Exception:
        # fallback to stratify only on hate_col
        try:
            train_idx, val_idx = train_test_split(
                df.index, test_size=val_size, random_state=seed, stratify=df[hate_col]
            )
            idx_val = set(val_idx)
            mask = [i in idx_val for i in range(len(df))]
            return mask
        except Exception:
            # last resort: random sample without stratify
            val_count = max(1, int(len(df) * val_size))
            val_idx = set(df.sample(n=val_count, random_state=seed).index)
            mask = [i in val_idx for i in range(len(df))]
            return mask

def process_file(path: Path, val_size: float, seed: int, inplace: bool):
    df = pd.read_csv(path)

    # try common column names (中文/英文可能不同) — adjust if needed
    possible_hate = ["hate_speech", "hate", "is_hate"]
    possible_christ = ["christianity_related", "christian_related", "is_christian"]

    hate_col = next((c for c in possible_hate if c in df.columns), None)
    christ_col = next((c for c in possible_christ if c in df.columns), None)

    if hate_col is None or christ_col is None:
        print(f"跳过 {path.name}：未找到所需列 (hate/christianity)。")
        return

    mask = stratified_split(df, hate_col, christ_col, val_size, seed)

    df["split"] = ["val" if m else "train" for m in mask]

    out_path = path if inplace else path.with_name(path.stem + "_with_split" + path.suffix)
    df.to_csv(out_path, index=False)
    print(f"已写入: {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default=".", help="数据文件夹（默认：脚本所在目录）")
    p.add_argument("--pattern", default="*finetuning*.csv", help="匹配的 CSV 模式")
    p.add_argument("--val-size", type=float, default=0.1, help="验证集比例，默认 0.1")
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    p.add_argument("--inplace", action="store_true", help="覆盖原文件")
    args = p.parse_args()

    base = Path(args.dir)
    # if user passes '.', interpret relative to script location
    if args.dir == ".":
        base = Path(__file__).parent

    files = sorted(base.glob(args.pattern))
    if not files:
        print(f"在 {base} 中未找到匹配 {args.pattern} 的文件。", file=sys.stderr)
        sys.exit(1)

    for f in files:
        try:
            process_file(f, args.val_size, args.seed, args.inplace)
        except KeyError as e:
            print(f"{f.name} 处理失败: {e}")


if __name__ == "__main__":
    main()
