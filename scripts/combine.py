import pandas as pd
import json
import os,re

def process_csv_data(csv_configs, keyword_json_path, target_language, output_path):
    """
    csv_configs: 列表，形如 [('path1.csv', 'col_a'), ('path2.csv', 'col_b')]
    keyword_json_path: 关键词 JSON 文件路径
    target_language: 指定筛选的语言键名 (如 'Chinese')
    output_path: 结果保存路径
    """
    
    # 1. 加载关键词列表
    with open(keyword_json_path, 'r', encoding='utf-8') as f:
        keyword_dict = json.load(f)
    
    # 获取指定语言的关键词，并确保不为空
    keywords = keyword_dict.get(target_language, [])
    if not keywords:
        print(f"警告: 在 JSON 中未找到语言 '{target_language}' 的关键词。")
        return

    all_texts = []

    # 2. 读取、提取并合并 CSV 指定列
    for csv_path, column_name in csv_configs:
        if not os.path.exists(csv_path):
            print(f"文件不存在，跳过: {csv_path}")
            continue
            
        # 读取 CSV（指定编码以防中文乱码）
        df = pd.read_csv(csv_path, encoding_errors='ignore')
        
        if column_name in df.columns:
            # 提取指定列，转换为字符串，并存入列表
            all_texts.extend(df[column_name].astype(str).tolist())
        else:
            print(f"列名 '{column_name}' 不在文件 {csv_path} 中，已跳过。")

    # 3. 合并与去重
    # 使用 set 去重后再转回 Series
    unique_series = pd.Series(list(set(all_texts)))
    print(f"合并去重后的初始条数: {len(unique_series)}")

    # 4. 关键词筛选
    # 使用 re.escape 对每个关键词进行转义，防止关键词里的特殊字符导致正则报错
    # 然后用 | 拼接成一个正则表达式
    pattern = '|'.join([re.escape(str(k)) for k in keywords])
    
    # 筛选包含任意一个关键词的文本
    filtered_series = unique_series[unique_series.str.contains(pattern, case=False, na=False)]
    
    # 筛选包含任意一个关键词的文本 (case=False 表示不区分大小写)
    filtered_series = unique_series[unique_series.str.contains(pattern, case=False, na=False)]
    
    print(f"关键词筛选后的条数: {len(filtered_series)}")

    # 5. 保存为单列 CSV
    # 给这一列起个名字叫 'text'
    final_df = pd.DataFrame(filtered_series, columns=['text'])

    # 新增：剔除长度超过 140 的行
    # .str.len() 会计算每个字符串的长度
    final_df = final_df[final_df['text'].str.len() <= 140]
    
    print(f"长度过滤（<=140）后的条数: {len(final_df)}")
    
    final_df.to_csv(output_path, index=False, encoding='utf-8-sig')

# --- 使用示例 ---
if __name__ == "__main__":
    # 输入配置
    configs = [
        ('data_collection/Tieba/all_search_posts.csv', 'main_content'),
        ('data_collection/hugging_face/filtered_social_media_data.csv', 'content'),
        # ('data_collection/common_crawl/extract_zh_religious.csv', 'text')
    ]
    
    json_path = 'data_collection/final_keywords.json'  # 你的关键词库路径
    lang = 'Chinese'            # 想要筛选的语言
    output = 'data_collection/Tieba/final_cleaned_data.csv'
    
    process_csv_data(configs, json_path, lang, output)