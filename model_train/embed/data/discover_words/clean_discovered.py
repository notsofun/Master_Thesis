import pandas as pd
import re

def normalize_word(text, lang):
    if pd.isna(text): return ""
    # 1. 转换为小写，并去掉所有空格以及标点符号（如 / , . - _ 等）
    # 这样 "圣母/圣母婊" 会变成 "圣母圣母婊"
    text = re.sub(r'[\s\W_]+', '', str(text).lower())
    
    # 2. 针对日语的简易去词尾
    if lang == 'jp' and len(text) > 2:
        text = re.sub(r'(る|た|ない|ます|じ|て|の)$', '', text)
    
    return text

def auto_clean_candidates(input_file, output_file, similarity_threshold=0.8):
    # 1. 加载数据
    df = pd.read_csv(input_file)
    print(f"原始数据条数: {len(df)}")

    # 2. 基础预处理
    df['candidate_word'] = df['candidate_word'].astype(str).str.strip()
    df['seed_word'] = df['seed_word'].astype(str).str.strip()
    
    # 3. 英文停用词片段（针对你 CSV 中出现的常见碎裂词）
    en_noise_fragments = {'u s', 'n t', 'mr', 'oh', 're', 'the', 'this', 'that', 'with', 'from', 'actually'}

    def filter_logic(row):
        # 保持原始引用，不要覆盖
        orig_word = str(row['candidate_word'])
        orig_seed = str(row['seed_word'])
        lang = row['language']
        sim = row['similarity']

        # --- 第一层：硬性过滤 ---
        if sim < similarity_threshold:
            return False
        
        # 使用标准化后的版本进行对比
        norm_word = normalize_word(orig_word, lang)
        norm_seed = normalize_word(orig_seed, lang)

        # 1. 解决“伪善 者” vs “伪善者”：标准化后相等则过滤
        if norm_word == norm_seed:
            return False
            
        # 2. 长度检查（基于标准化后的内容）
        if len(norm_word) < (2 if lang != 'en' else 3):
            return False
        
        # 3. 包含关系过滤：
        # 如果你不需要“圣母”在种子词“圣母/圣母婊”中出现，使用标准化版本比对
        if norm_word in norm_seed or norm_seed in norm_word:
            return False

        # 4. 纯数字或特殊符号不要
        if re.match(r'^[0-9\W_]+$', orig_word):
            return False

        # --- 第二层：针对原始英文片段的噪音过滤 ---
        if lang == 'en':
            if len(orig_word) < 4 or orig_word.lower() in en_noise_fragments:
                return False
        
        return True
    
    # 执行过滤
    df_cleaned = df[df.apply(filter_logic, axis=1)].copy()

    # --- 新增：形态去重逻辑 ---
    # 为每一行生成一个“特征指纹”
    df_cleaned['fingerprint'] = df_cleaned.apply(lambda r: normalize_word(r['candidate_word'], r['language']), axis=1)
    
    # 如果多个词标准化后指纹一样（比如“信じ”和“信じる”都会变成“信”），只保留相似度最高的那一个
    df_cleaned = df_cleaned.sort_values('similarity', ascending=False)
    df_cleaned = df_cleaned.drop_duplicates(subset=['seed_word', 'fingerprint'], keep='first')
    
    # 移除临时的指纹列
    df_cleaned = df_cleaned.drop(columns=['fingerprint'])

    # --- 第三层：统计聚合（第二层优化） ---
    # 计算每个候选词在多少个不同的种子词下出现过
    df_cleaned['occurrence_count'] = df_cleaned.groupby('candidate_word')['seed_word'].transform('count')
    df_cleaned = df_cleaned[df_cleaned.groupby('candidate_word')['seed_word'].transform('count') >= 1]
    
    # 按语言和相似度排序，方便查阅
    df_cleaned = df_cleaned.sort_values(by=['language', 'similarity'], ascending=[True, False])

    # 4. 保存结果
    df_cleaned.to_csv(output_file, index=False)
    print(f"清洗完成！剩余候选词: {len(df_cleaned)}")
    print(f"精选结果已保存至: {output_file}")

# 使用示例
auto_clean_candidates('model_train/embed/data/discovered_candidates.csv', 'model_train/embed/data/refined_candidates.csv', similarity_threshold=0.85)