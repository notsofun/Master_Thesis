import pandas as pd
from gliner import GLiNER
from collections import Counter
from tqdm import tqdm
import os

def extract_hate_targets(df):
    print("正在加载轻量级多语言 GLiNER 模型 (首次运行会自动下载)...")
    # 使用官方推荐的多语言版本，对中英日支持极好
    model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")
    
    # 核心：为你 RQ1 量身定制的“靶子”提取标签
    # 我们不仅要传统实体，还要抓取“特定群体”和“身份标签”
    labels = [
        "Religious Group",      # 宗教群体 (如：天主教徒、统一教会)
        "Political Group",      # 政治群体 (如：左翼、Liberal)
        "Specific Person",      # 特定人物 (如：特朗普、安倍)
        "Social Identity",      # 社会身份 (如：圣母、女权、LGBT)
        "Organization"          # 组织机构 (如：教会、自民党)
    ]

    print(f"开始处理文本，共 {len(df)} 条记录。标签设定为: {labels}")
    
    # 存储提取结果
    extracted_entities = []
    
    # 使用 tqdm 显示本地运行进度条
    for text in tqdm(df['text'].fillna(""), desc="提取实体中"):
        if not str(text).strip():
            extracted_entities.append([])
            continue
            
        # 提取实体，阈值 (threshold) 设为 0.3，保持较高的召回率
        entities = model.predict_entities(str(text), labels, threshold=0.3)
        
        # 整理该句子的所有实体，转为小写以方便后续统计 (针对英文)
        doc_entities = [ent["text"].lower().strip() for ent in entities]
        extracted_entities.append(doc_entities)
        
    df['hate_targets'] = extracted_entities
    return df

def analyze_topic_targets(df, output_dir="unsupervised_classification/RQ1/data"):
    print("\n--- 正在汇总每个 Topic 的核心攻击目标 ---")
    
    # 过滤掉噪声簇 (-1)
    valid_df = df[df['topic'] != -1]
    
    topic_target_summary = []
    
    # 按 Topic 分组统计靶子
    for topic_id, group in valid_df.groupby('topic'):
        # 展平该 topic 下所有的实体列表
        all_targets = [target for sublist in group['hate_targets'] for target in sublist]
        
        # 统计词频
        target_counts = Counter(all_targets)
        
        # 提取出现频率最高的前 10 个靶子
        top_10_targets = target_counts.most_common(10)
        
        topic_target_summary.append({
            "Topic_ID": topic_id,
            "Topic_Size": len(group),
            "Top_Targets": [f"{t[0]}({t[1]})" for t in top_10_targets]
        })
        
    summary_df = pd.DataFrame(topic_target_summary)
    
    # 打印前 5 个有效话题的目标
    print(summary_df.head().to_string())
    
    # 保存结果，供你写论文时制表
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'rq1_topic_targets_summary.csv')
    summary_df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ RQ1 目标提取完成！已保存至 {save_path}")
    
    return summary_df

# ==========================================
# 运行示例
# ==========================================
if __name__ == "__main__":
    # 假设你的 df 已经包含了 'text' 和 BERTopic 跑出来的 'topic' 列
    df = pd.read_csv(r'unsupervised_classification\topic_modeling_results\sixth\data\document_topic_mapping.csv')
    
    # 1. 运行提取 (本地 CPU 跑 3000 条大概需要 5-15 分钟)
    df_with_targets = extract_hate_targets(df) # 这里的 df 传入你实际的 DataFrame
    
    # 2. 汇总统计
    summary_df = analyze_topic_targets(df_with_targets)