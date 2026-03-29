import pandas as pd
import torch
import os
import json
from tqdm import tqdm
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# --- 1. 初始化双抽取模型 ---
def init_models():
    print("正在初始化 NuExtract 与 UniversalNER 模型 (使用 4-bit 量化以减轻负担)...")
    
    # NuExtract: 擅长严格按 JSON 格式提取
    nuextract_name = "numind/NuExtract-tiny" 
    # UniversalNER: 擅长开放域命名实体识别
    uniner_name = "Universal-NER/UniNER-7B-type-all" # 或使用更轻量的版本

    # 通用量化配置
    load_kwargs = {"device_map": "auto", "torch_dtype": torch.float16, "load_in_4bit": True}

    # 加载 NuExtract
    tokenizer_nu = AutoTokenizer.from_pretrained(nuextract_name)
    model_nu = AutoModelForCausalLM.from_pretrained(nuextract_name, **load_kwargs)
    
    # 加载 UniversalNER (示例逻辑，实际运行时可根据配置切换)
    # 注意：如果显存有限，可以串行执行：先跑完 NuExtract，释放显存，再跑 UniNER
    return (model_nu, tokenizer_nu)

# --- 2. 核心抽取逻辑 ---
def extract_atomic_entities(text, model_nu, tokenizer_nu, labels):
    """
    使用指令驱动的方式，强制模型只提取原子实体。
    """
    # 定义 NuExtract 的 JSON 模板
    schema = {label: [] for label in labels}
    
    # 构造 Prompt：NuExtract 需要特定的输入格式
    # <|input|> 后接文本，<|schema|> 后接 JSON 结构
    input_text = f"""<|input|>\n{text}\n<|schema|>\n{json.dumps(schema, ensure_ascii=False)}"""
    
    input_ids = tokenizer_nu(input_text, return_tensors="pt").to("cuda")
    outputs = model_nu.generate(**input_ids, max_new_tokens=256, early_stopping=True)
    
    prediction = tokenizer_nu.decode(outputs[0], skip_special_tokens=True)
    
    # 提取 JSON 部分
    try:
        # NuExtract 会在末尾生成填充好的 JSON
        res_json = prediction.split("<|output|>")[-1].strip()
        extracted = json.loads(res_json)
    except:
        return []

    # 展平所有标签下的实体
    all_entities = []
    for val in extracted.values():
        if isinstance(val, list):
            all_entities.extend([str(i).lower().strip() for i in val])
    
    return list(set(all_entities))

def extract_hate_targets_v2(df):
    labels = ["Religious Group", "Political Group", "Specific Person", "Social Identity", "Organization"]
    
    # 初始化模型
    model_nu, tokenizer_nu = init_models()
    
    extracted_results = []
    
    for text in tqdm(df['text'].fillna(""), desc="双模型联合抽取中"):
        if not str(text).strip():
            extracted_results.append([])
            continue
            
        # 1. 第一路：NuExtract 抽取
        nu_ents = extract_atomic_entities(str(text), model_nu, tokenizer_nu, labels)
        
        # 2. (可选) 第二路：UniversalNER 抽取
        # 这里建议根据你的本地资源决定是否并行。如果显存够，逻辑同上。
        # uni_ents = extract_from_uniner(str(text), labels)
        
        # 3. 去重与清洗：合并并剔除长度超过 10 个字符的“伪实体”（句子）
        combined = list(set(nu_ents))
        clean_entities = [e for e in combined if len(e) < 15 and len(e) > 1]
        
        extracted_results.append(clean_entities)
        
    df['hate_targets'] = extracted_results
    return df

# --- 3. 统计逻辑 (保持你的代码结构) ---
def analyze_topic_targets(df, output_dir="unsupervised_classification/RQ1/data"):
    print("\n--- 正在汇总每个 Topic 的原子靶子 (清理版) ---")
    valid_df = df[df['topic'] != -1]
    topic_target_summary = []
    
    for topic_id, group in valid_df.groupby('topic'):
        all_targets = [target for sublist in group['hate_targets'] for target in sublist]
        target_counts = Counter(all_targets)
        top_10_targets = target_counts.most_common(10)
        
        topic_target_summary.append({
            "Topic_ID": topic_id,
            "Topic_Size": len(group),
            "Top_Targets": [f"{t[0]}({t[1]})" for t in top_10_targets]
        })
        
    summary_df = pd.DataFrame(topic_target_summary)
    os.makedirs(output_dir, exist_ok=True)
    summary_df.to_csv(os.path.join(output_dir, 'rq1_atomic_targets_summary.csv'), index=False, encoding='utf-8-sig')
    return summary_df

if __name__ == "__main__":
    df = pd.read_csv(r'unsupervised_classification\topic_modeling_results\sixth\data\document_topic_mapping.csv')
    df_with_targets = extract_hate_targets_v2(df)
    summary_df = analyze_topic_targets(df_with_targets)