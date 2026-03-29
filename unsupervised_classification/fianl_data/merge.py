import pandas as pd
import os

def merge_datasets():
    # 1. 定义文件路径
    file_llm = r'data_detect\finetuned_detection\chinese_final_religious_hate_llm.csv'
    file_original = r'data_detect\finetuned_detection\chinese_final_religious_hate.csv'
    output_file = r'unsupervised_classification\fianl_data\chinese_religious_hate.csv'

    print("正在读取数据...")
    
    # 2. 读取 CSV 文件
    # 假设原始文件中有 text 列，如果没有请根据实际列名修改
    df_llm = pd.read_csv(file_llm)
    df_original = pd.read_csv(file_original)
    print(f"原始数据行数{len(df_original)}\n增强数据{len(df_llm)}")

    # 3. 新增 source 标注列
    df_llm['source'] = 'llm'
    df_original['source'] = 'original'

    # 4. 合并数据
    # 只保留需要的 text 和 source 两列
    # 注意：如果你的原始列名不是 'text'，请把下面的 'text' 替换为实际列名
    combined_df = pd.concat([df_llm[['response', 'source']], df_original[['text', 'source']]], ignore_index=True)

    # 5. 保存结果
    combined_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"合并完成！总行数: {len(combined_df)}")
    print(f"结果已保存至: {output_file}")

if __name__ == "__main__":
    # 确保运行路径正确，或者你可以手动创建目录
    try:
        merge_datasets()
    except FileNotFoundError as e:
        print(f"错误：找不到文件，请检查路径是否正确。\n{e}")
    except Exception as e:
        print(f"发生错误：{e}")