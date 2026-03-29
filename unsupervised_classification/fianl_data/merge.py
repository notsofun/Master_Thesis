import pandas as pd
import os

def merge_datasets():
    # 1. 定义文件路径
    file_llm = r'data_detect\finetuned_detection\japanese_final_religious_hate_llm.csv'
    file_original = r'data_detect\finetuned_detection\japanese_final_religious_hate.csv'
    output_path = r'unsupervised_classification\fianl_data'
    output_file = os.path.join(output_path, 'japanese_religious_hate.csv')

    # 确保输出目录存在
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"创建目录: {output_path}")

    print("正在读取数据...")
    
    # 2. 读取 CSV 文件
    df_llm = pd.read_csv(file_llm)
    df_original = pd.read_csv(file_original)
    
    print(f"原始数据行数: {len(df_original)}")
    print(f"增强数据行数: {len(df_llm)}")

    # 3. 统一列名：将 LLM 数据的 'response' 改名为 'text'
    # 这样 concat 时它们会自动对齐到同一列
    df_llm = df_llm.rename(columns={'response': 'text'})

    # 4. 新增 source 标注列
    df_llm['source'] = 'llm'
    df_original['source'] = 'original'

    # 5. 合并数据
    # 只提取 text 和 source 这两列进行合并
    combined_df = pd.concat([df_llm[['text', 'source']], df_original[['text', 'source']]], ignore_index=True)

    # 6. 保存结果
    combined_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print("-" * 30)
    print(f"合并完成！")
    print(f"总行数: {len(combined_df)}")
    print(f"结果已保存至: {output_file}")

if __name__ == "__main__":
    try:
        merge_datasets()
    except FileNotFoundError as e:
        print(f"错误：找不到文件，请检查路径是否正确。\n{e}")
    except KeyError as e:
        print(f"错误：列名不匹配，请检查原始文件中是否存在 'response' 或 'text' 列。\n{e}")
    except Exception as e:
        print(f"发生错误：{e}")