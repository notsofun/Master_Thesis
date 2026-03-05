import pandas as pd
import os

def process_csv_files(input_paths, output_path):
    """
    读取多个CSV，按 'text' 字段去重，转换标签并输出统计
    """
    df_list = []
    
    print("--- 阶段 1: 读取文件 ---")
    for path in input_paths:
        if os.path.exists(path):
            # 读取数据
            df = pd.read_csv(path)
            print(f"读取成功: {path} | 原始行数: {len(df)}")
            df_list.append(df)
        else:
            print(f"跳过: 找不到文件 {path}")

    if not df_list:
        print("错误: 未加载任何有效数据。")
        return

    # 1. 合并所有 DataFrame
    full_df = pd.concat(df_list, ignore_index=True)
    initial_count = len(full_df)

    # 2. 按 'text' 字段去重
    if 'text' in full_df.columns:
        # keep='first' 表示保留重复项中的第一条
        full_df.drop_duplicates(subset=['text'], keep='first', inplace=True)
        after_dedup_count = len(full_df)
        print(f"\n--- 阶段 2: 去重处理 ---")
        print(f"合并后总行数: {initial_count}")
        print(f"按 'text' 去重后行数: {after_dedup_count}")
        print(f"移除了 {initial_count - after_dedup_count} 条重复记录")
    else:
        print("\n警告: 未在文件中找到 'text' 字段，无法执行去重。")
        after_dedup_count = initial_count

    # 3. 字段转换：是/否 -> 1/0
    target_columns = ["christianity_related", "hate_speech"]
    print(f"\n--- 阶段 3: 标签转换与统计 ---")
    
    for col in target_columns:
        if col in full_df.columns:
            # 转换逻辑：处理空格并映射
            full_df[col] = full_df[col].astype(str).str.strip().map({'是': 1, '否': 0}).fillna(0).astype(int)
            
            # 计算占比
            positive_count = full_df[col].sum()
            percentage = (positive_count / after_dedup_count) * 100 if after_dedup_count > 0 else 0
            print(f"项目 [{col}]: 1(是) = {positive_count} | 0(否) = {after_dedup_count - positive_count} | 占比 = {percentage:.2f}%")
        else:
            print(f"提示: 字段 '{col}' 不存在，跳过。")

    # 4. 保存结果
    full_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n结果已保存至: {output_path}")

# --- 配置区 ---
if __name__ == "__main__":
    # 在这里指定你的文件路径
    files_to_combine = [
        "data_augmentation/back_translation/data/back_translated_chinese.csv",
        "model_train/classifier/data/chinese_finetuning.csv",
        # "C:/path/to/your/file3.csv" 
    ]
    
    # 指定输出路径
    final_output = "model_train/classifier/data/chinese_finetuning_2.csv"
    
    # 执行
    process_csv_files(files_to_combine, final_output)