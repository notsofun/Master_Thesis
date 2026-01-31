"""
Filter CSV files by Chinese religious keywords.

功能：
- 加载 fianl_keywors.json 中的 Chinese 关键词列表
- 过滤 all_search_posts.csv（检查 main_content 字段）
- 过滤 extract_zh_religious.csv（检查 text 字段）
- 只保留至少包含一个关键词的行
- 保存过滤后的结果

使用示例：
python scripts/filter_by_keywords.py \
  --keywords_path data_collection/fianl_keywors.json \
  --tieba_csv data_collection/Tieba/all_search_posts.csv \
  --extract_csv data_collection/common_crawl/extract_zh_religious.csv \
  --output_tieba outputs/filtered_all_search_posts.csv \
  --output_extract outputs/filtered_extract_zh_religious.csv
"""

import argparse
import json
import re
from typing import List, Set
import pandas as pd


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--keywords_path", type=str, default="data_collection/fianl_keywors.json",
                   help="关键词 JSON 文件路径")
    p.add_argument("--non_common_csv", type=str, default="data_collection/Tieba/all_search_posts.csv",
                   help="Tieba all_search_posts.csv 路径")
    p.add_argument("--lang", type=str, default='Chinese', help="Japanese or Chinese")
    p.add_argument("--extract_csv", type=str, default="data_collection/common_crawl/extract_zh_religious.csv",
                   help="extract_zh_religious.csv 路径")
    p.add_argument("--output", type=str, default="outputs/merged_filtered_religious_zh.csv",
                   help="合并后的输出路径（已去重）")
    p.add_argument("--case_sensitive", action="store_true", help="关键词匹配是否区分大小写（默认不区分）")
    p.add_argument("--verbose", action="store_true", help="打印详细信息")
    p.add_argument("--dedup_column", type=str, default="text", 
                   help="用于去重的列名（默认 'text'；可指定多列用逗号分隔）")
    return p.parse_args()


def load_keywords(keywords_path: str, lang:str) -> Set[str]:
    """从 JSON 加载 Chinese 关键词列表。"""
    with open(keywords_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    keywords = data.get(lang, [])
    if not keywords:
        raise ValueError("JSON 中未找到 Japanese 关键词列表")
    
    return set(keywords)


def contains_keyword(text: str, keywords: Set[str], case_sensitive: bool = False) -> bool:
    """检查文本是否包含任意一个关键词。"""
    if not text or not isinstance(text, str):
        return False
    
    if not case_sensitive:
        text = text.lower()
        keywords = {kw.lower() for kw in keywords}
    
    for kw in keywords:
        if kw in text:
            return True
    return False


def filter_csv(csv_path: str, text_column: str, keywords: Set[str], 
               case_sensitive: bool = False, verbose: bool = False) -> pd.DataFrame:
    """过滤 CSV，只保留包含关键词的行。"""
    df = pd.read_csv(csv_path, encoding="utf-8")
    
    if text_column not in df.columns:
        raise ValueError(f"列 '{text_column}' 不存在于 {csv_path}。可用列：{list(df.columns)}")
    
    # 创建布尔掩码
    mask = df[text_column].apply(lambda x: contains_keyword(x, keywords, case_sensitive))
    
    filtered_df = df[mask].copy()
    
    if verbose:
        print(f"原始行数: {len(df)}")
        print(f"过滤后行数: {len(filtered_df)}")
        print(f"保留率: {len(filtered_df) / len(df) * 100:.2f}%")
    
    return filtered_df


def merge_and_deduplicate(tieba_df: pd.DataFrame, extract_df: pd.DataFrame, 
                          dedup_column: str = "text", verbose: bool = False) -> pd.DataFrame:
    """
    合并两个 DataFrame，按指定列去重（默认按 'text' 列）。
    Tieba 的 'main_content' 列会重命名为 'text'。
    """
    # 重命名 tieba_df 中的 main_content 为 text
    tieba_copy = tieba_df.copy()
    if "main_content" in tieba_copy.columns:
        tieba_copy = tieba_copy.rename(columns={"main_content": "text"})
    
    # 纵向合并
    merged = pd.concat([extract_df, tieba_copy], ignore_index=True, sort=False)
    
    if verbose:
        print(f"合并前总行数：{len(merged)}")
    
    # 按 dedup_column 去重，保留第一次出现
    dedup_cols = [c.strip() for c in dedup_column.split(",")]
    # 仅当所有指定列存在时才去重
    valid_cols = [c for c in dedup_cols if c in merged.columns]
    if valid_cols:
        merged = merged.drop_duplicates(subset=valid_cols, keep="first")
    else:
        if verbose:
            print(f"警告：指定的去重列 {dedup_cols} 不全存在，按全行去重")
        merged = merged.drop_duplicates(keep="first")
    
    if verbose:
        print(f"去重后总行数：{len(merged)}")
    
    return merged


def analyze_keyword_distribution(df: pd.DataFrame, text_column: str, keywords: Set[str], 
                                  case_sensitive: bool = False, verbose: bool = False) -> None:
    """
    分析关键词在数据中的分布情况。
    统计每个关键词出现的次数和占比。
    """
    if not df[text_column].notna().any():
        print("警告：没有有效的文本数据")
        return
    
    # 统计每个关键词的出现次数
    keyword_counts = {}
    total_matches = 0
    
    for _, row in df.iterrows():
        text = row[text_column]
        if not isinstance(text, str):
            continue
        
        test_text = text if case_sensitive else text.lower()
        for keyword in keywords:
            test_keyword = keyword if case_sensitive else keyword.lower()
            if test_keyword in test_text:
                keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
                total_matches += 1
    
    if total_matches == 0:
        print("警告：没有找到任何关键词匹配")
        return
    
    # 排序并打印结果
    sorted_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n=== 关键词分布分析 ===")
    print(f"总数据行数：{len(df)}")
    print(f"总关键词匹配次数：{total_matches}")
    print(f"出现过关键词的行数：{len(df)}")
    print(f"\n关键词占比排序（Top 20）：")
    print(f"{'排名':<5} {'关键词':<20} {'出现次数':<10} {'行占比':<10} {'匹配占比':<10}")
    print("-" * 65)
    
    for rank, (keyword, count) in enumerate(sorted_keywords[:20], 1):
        row_percentage = (count / len(df)) * 100
        match_percentage = (count / total_matches) * 100
        print(f"{rank:<5} {keyword:<20} {count:<10} {row_percentage:>8.2f}% {match_percentage:>8.2f}%")
    
    if len(sorted_keywords) > 20:
        print(f"... 还有 {len(sorted_keywords) - 20} 个关键词")
    
    # 统计未出现的关键词
    appeared_keywords = set(keyword_counts.keys())
    not_appeared = keywords - appeared_keywords
    if not_appeared:
        print(f"\n未出现的关键词数量：{len(not_appeared)}")
        if verbose and len(not_appeared) <= 20:
            print(f"未出现的关键词：{', '.join(sorted(not_appeared))}")


def main():
    args = get_args()
    
    # 加载关键词
    print(f"加载关键词：{args.keywords_path}")
    keywords = load_keywords(args.keywords_path, args.lang)
    print(f"关键词数量：{len(keywords)}")
    if args.verbose:
        print(f"关键词列表（前20个）：{list(keywords)[:20]}")
    
    # 过滤 Tieba CSV
    print(f"\n处理 {args.non_common_csv}...")
    try:
        filtered_df = filter_csv(args.non_common_csv, "main_content", keywords, 
                            case_sensitive=args.case_sensitive, verbose=True)
    except:
        filtered_df = filter_csv(args.non_common_csv, "text", keywords, 
                            case_sensitive=args.case_sensitive, verbose=True)
    print(f"过滤完成：{len(filtered_df)} 行")
    
    # 过滤 Extract CSV
    print(f"\n处理 {args.extract_csv}...")
    extract_df = filter_csv(args.extract_csv, "text", keywords,
                            case_sensitive=args.case_sensitive, verbose=True)
    print(f"过滤完成：{len(extract_df)} 行")
    
    # 合并和去重
    print(f"\n合并两个 CSV 并去重...")
    merged_df = merge_and_deduplicate(filtered_df, extract_df, 
                                       dedup_column=args.dedup_column,
                                       verbose=True)
    
    # 分析关键词分布
    analyze_keyword_distribution(merged_df, "text", keywords, 
                                 case_sensitive=args.case_sensitive, 
                                 verbose=args.verbose)
    
    # 保存合并结果
    merged_df.to_csv(args.output, index=False, encoding="utf-8")
    print(f"\n已保存到 {args.output}")
    
    # 总结
    print(f"\n=== 总结 ===")
    print(f"非common过滤：{len(filtered_df)} 行")
    print(f"Extract 过滤：{len(extract_df)} 行")
    print(f"合并后去重：{len(merged_df)} 行")
    print(f"输出文件：{args.output}")


if __name__ == "__main__":
    main()
