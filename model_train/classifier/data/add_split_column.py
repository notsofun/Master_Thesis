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
import numpy as np

def stratified_split(df, hate_col, christ_col, val_size, seed):
    from sklearn.model_selection import StratifiedShuffleSplit
    
    # --- 核心改进：在切分前先将“是/否”转为数字，确保分层逻辑生效 ---
    label_map = {"是": 1, "否": 0, "yes": 1, "no": 0, 1: 1, 0: 0, "1": 1, "0": 0}
    
    # 临时生成用于分层的编码列，防止 NaN 干扰
    h_coded = df[hate_col].map(label_map).fillna(0).astype(int)
    c_coded = df[christ_col].map(label_map).fillna(0).astype(int)
    
    # 构造联合标签 (例如 0__1, 1__0) 用于多任务分层抽样
    combined = h_coded.astype(str) + "__" + c_coded.astype(str)

    try:
        # StratifiedShuffleSplit 会严格保持 combined 标签在 train/val 中的比例
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
        train_idx, val_idx = next(splitter.split(df, combined))
        
        # 返回一个布尔 mask
        mask = np.zeros(len(df), dtype=bool)
        mask[val_idx] = True
        return mask
    except Exception as e:
        print(f"分层抽样失败（可能是某类样本太少），切换至随机抽样。错误: {e}")
        val_count = max(1, int(len(df) * val_size))
        val_idx = set(df.sample(n=val_count, random_state=seed).index)
        return [i in val_idx for i in range(len(df))]

def process_file(path: Path, val_size: float, seed: int, inplace: bool):
    df = pd.read_csv(path)

    # 1. 自动匹配列名
    possible_hate = ["hate_speech", "hate", "is_hate", "仇恨倾向"] 
    possible_christ = ["christianity_related", "christian_related", "is_christian", "基督教相关"]

    hate_col = next((c for c in possible_hate if c in df.columns), None)
    christ_col = next((c for c in possible_christ if c in df.columns), None)

    if hate_col is None or christ_col is None:
        print(f"跳过 {path.name}：未找到所需列。")
        return

    # 2. 【核心修复】全局标签转换 (映射为数值 0/1)
    # 这样既能让分层逻辑识别，也能让 train.py 识别
    label_map = {"是": 1, "否": 0, "yes": 1, "no": 0, "1": 1, "0": 0, 1: 1, 0: 0}
    
    # 转换并填充缺失值为 0
    df[hate_col] = df[hate_col].map(label_map).fillna(0).astype(int)
    df[christ_col] = df[christ_col].map(label_map).fillna(0).astype(int)

    # 3. 执行分层切分 (此时 df 已经是数值了)
    mask = stratified_split(df, hate_col, christ_col, val_size, seed)
    df["split"] = ["val" if m else "train" for m in mask]

    # 4. 打印分布检查（确保分层成功）
    train_pos = df[df["split"]=="train"][hate_col].sum()
    val_pos = df[df["split"]=="val"][hate_col].sum()
    print(f"文件: {path.name}")
    print(f"  -> 训练集正样本数: {train_pos} / {len(df[df['split']=='train'])}")
    print(f"  -> 验证集正样本数: {val_pos} / {len(df[df['split']=='val'])}")

    # 5. 保存结果
    out_path = path if inplace else path.with_name(path.stem + "_with_split" + path.suffix)
    df.to_csv(out_path, index=False)
    print(f"已保存数值化且包含 split 列的文件: {out_path}\n")

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
