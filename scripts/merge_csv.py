import pandas as pd
import os

def process_csv_files(input_paths, output_path):
    df_list = []
    
    print("--- 阶段 1: 读取文件 ---")
    for path in input_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"读取成功: {path} | 行数: {len(df)}")
            df_list.append(df)

    if not df_list: return

    full_df = pd.concat(df_list, ignore_index=True)
    initial_count = len(full_df)

    print(f"\n--- 阶段 2: 去重处理 ---")
    
    # 关键修改：按这两个字段同时去重
    # 这样：ID相同但 method不同（一个是NaN，一个是back_translation）的记录会同时保留
    dedup_columns = ['annotation_id', 'augment_method']
    
    # 为了统计方便，先处理一下 NaN，防止 drop_duplicates 对 NaN 处理不一致
    full_df['augment_method'] = full_df['augment_method'].fillna('original')
    
    full_df.drop_duplicates(subset=dedup_columns, keep='first', inplace=True)
    
    after_dedup_count = len(full_df)
    print(f"合并后总行数: {initial_count}")
    print(f"去重后行数: {after_dedup_count}")
    print(f"实际移除（完全重复）的记录: {initial_count - after_dedup_count}")

    # 3. 标签分布统计
    target_columns = ["christianity_related", "hate_speech"]
    print(f"\n--- 阶段 3: 标签分布统计 ---")
    
    for col in target_columns:
        if col in full_df.columns:
            # 确保统计的是数值或清洗后的标签
            stats = full_df[col].value_counts(dropna=False)
            print(f"项目 [{col}] 分布:")
            for val, count in stats.items():
                print(f"  - 值 {val}: {count} 条 ({count/after_dedup_count:.2%})")

    # 4. 保存
    full_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n结果已保存至: {output_path}")

# --- 配置区 ---
if __name__ == "__main__":
    # 在这里指定你的文件路径
    files_to_combine = [
        "data_augmentation/back_translation/data/back_translated_chinese.csv",
        "model_train/classifier/data/final_annotated_chinese_train.csv",
    ]
    
    # 指定输出路径
    final_output = "model_train/classifier/data/chinese_finetuning_final.csv"
    
    # 执行
    process_csv_files(files_to_combine, final_output)