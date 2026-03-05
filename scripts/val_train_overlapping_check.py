import pandas as pd
from difflib import SequenceMatcher
import tqdm

def string_similarity(a, b):
    """计算两个字符串的相似度"""
    return SequenceMatcher(None, a, b).ratio()

def check_leakage(csv_path, text_col='text', split_col='split'):
    # 1. 读取数据
    df = pd.read_csv(csv_path)
    
    # 2. 分离训练集和验证集
    train_df = df[df[split_col] == 'train'].copy()
    val_df = df[df[split_col] == 'val'].copy()
    
    train_texts = set(train_df[text_col].astype(str).tolist())
    val_texts = val_df[text_col].astype(str).tolist()
    
    print(f"--- 数据规模 ---")
    print(f"训练集 (Train): {len(train_df)} 条")
    print(f"验证集 (Val): {len(val_df)} 条")
    print("-" * 30)

    # 3. 完全重合检测 (Exact Match)
    exact_matches = [t for t in val_texts if t in train_texts]
    exact_rate = len(exact_matches) / len(val_texts) * 100
    
    print(f"【判定 1：完全一致】")
    print(f"验证集中有 {len(exact_matches)} 条文本在训练集中完全出现。")
    print(f"完全重合率: {exact_rate:.2f}%")

    # 4. 模糊匹配检测 (针对回译、AEDA等孪生兄弟)
    # 我们设定相似度阈值为 0.85 (通常回译后的相似度在此区间)
    THRESHOLD = 0.85
    leakage_count = 0
    
    print(f"\n【判定 2：模糊重合检测 (相似度 > {THRESHOLD})】")
    print("正在逐条扫描验证集，对比训练集相似度...")
    
    # 使用 tqdm 显示进度条
    for val_t in tqdm.tqdm(val_texts):
        # 快速过滤：如果已经完全重合了，直接算作泄露
        if val_t in train_texts:
            leakage_count += 1
            continue
            
        # 模糊对比逻辑
        is_leaked = False
        for train_t in train_texts:
            if string_similarity(val_t, train_t) > THRESHOLD:
                is_leaked = True
                break
        
        if is_leaked:
            leakage_count += 1

    leakage_rate = leakage_count / len(val_texts) * 100
    print(f"\n识别到潜在“孪生兄弟”总数: {leakage_count} 条")
    print(f"总体泄露风险率: {leakage_rate:.2f}%")

    if leakage_rate > 5:
        print("\n结论：⚠️ 风险极高！验证集已被污染，训练出的 0.96 F1 极大可能是虚高。")
    else:
        print("\n结论：✅ 数据隔离尚可，F1 指标可信度较高。")

if __name__ == "__main__":
    # 请修改为你的 CSV 文件路径
    csv_file = "model_train/classifier/data/chinese_finetuning_2_with_split.csv" 
    check_leakage(csv_file)