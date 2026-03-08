# data_detect/finetuned_detection/main.py
import pandas as pd
import torch
import os, sys
import requests
import logging
from transformers import AutoTokenizer
from tqdm import tqdm
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir)) 

if root_dir not in sys.path:
    sys.path.append(root_dir)

from scripts.set_logger import setup_logging
# 导入你本地定义的模型类
from model_train.classifier.models.base_model import MultiTaskClassifier

# ==========================================
# 1. 配置日志 (Logger)
# ==========================================
logger, LOG_FILE_PATH = setup_logging(name="fine-tuned_predictions")

# ==========================================
# 2. 配置参数
# ==========================================
MODEL_NAME = "thu-coai/roberta-base-cold"
WEIGHTS_URL = "https://huggingface.co/Zhidian2025/Master-Thesis-Models/resolve/main/Thu-Chinese-hate-v1.pt"
LOCAL_WEIGHTS_PATH = "model_train/classifier/Chinese/thu_best_multitask_model_back_translated_both_focal_loss.pt"

INPUT_CSV = "data_collection/Tieba/final_cleaned_data.csv"
TEXT_COLUMN = "text"
OUTPUT_CSV = "data_detect/finetuned_detection/chinese_predictions.csv"

BATCH_SIZE = 32 
MAX_LENGTH = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def download_weights(url, save_path):
    if os.path.exists(save_path):
        logger.info(f"检测到本地权重文件: {save_path}，跳过下载。")
        return
    logger.info(f"正在从远程下载权重文件...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    with open(save_path, 'wb') as f, tqdm(
        desc="下载进度", total=total_size, unit='iB', unit_scale=True, unit_divisor=1024
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)

def log_statistics(df, col_name, task_label):
    """使用 logger 打印类别分布统计"""
    counts = df[col_name].value_counts().to_dict()
    total = len(df)
    logger.info(f"=== {task_label} 分布报告 ===")
    for label in [0, 1]:
        count = counts.get(label, 0)
        percentage = (count / total) * 100
        label_str = "Positive (1)" if label == 1 else "Negative (0)"
        logger.info(f"  > {label_str}: {count} 条 ({percentage:.2f}%)")

def main():
    # 确保输出目录存在
    output_dir = os.path.dirname(OUTPUT_CSV)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # 1. 准备权重
    try:
        download_weights(WEIGHTS_URL, LOCAL_WEIGHTS_PATH)
    except Exception as e:
        logger.error(f"下载权重失败: {e}")
        return

    # 2. 加载模型
    logger.info(f"正在加载 Tokenizer 和模型至设备: {DEVICE}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = MultiTaskClassifier(MODEL_NAME)
        
        state_dict = torch.load(LOCAL_WEIGHTS_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        logger.info("模型加载成功，进入评估模式。")
    except Exception as e:
        logger.error(f"模型初始化失败: {e}")
        return

    # 3. 读取数据
    if not os.path.exists(INPUT_CSV):
        logger.error(f"输入文件不存在: {INPUT_CSV}")
        return
        
    logger.info(f"正在读取数据: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("").astype(str)
    texts = df[TEXT_COLUMN].tolist()
    
    all_rel = []
    all_hate = []

    # 4. 批量预测
    logger.info(f"开始推理，总计数据量: {len(texts)}")
    with torch.no_grad():
        # 使用 tqdm 配合 logger 建议设置 mininterval 减少屏幕刷新频率
        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="预测进度"):
            batch_texts = texts[i : i + BATCH_SIZE]
            
            inputs = tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                max_length=MAX_LENGTH, 
                return_tensors="pt"
            ).to(DEVICE)

            rel_logits, hate_logits = model(inputs['input_ids'], inputs['attention_mask'])
            
            # 使用 Sigmoid 判定
            rel_preds = (torch.sigmoid(rel_logits) > 0.5).int().cpu().numpy().tolist()
            hate_preds = (torch.sigmoid(hate_logits) > 0.5).int().cpu().numpy().tolist()
            
            # 兼容性处理
            if isinstance(rel_preds, int): rel_preds = [rel_preds]
            if isinstance(hate_preds, int): hate_preds = [hate_preds]

            all_rel.extend(rel_preds)
            all_hate.extend(hate_preds)

    # 5. 结果处理
    df['is_religion_related'] = all_rel
    df['is_hate_speech'] = all_hate
    
    # 打印最终统计日志
    logger.info("推理结束，开始生成统计报告：")
    log_statistics(df, 'is_religion_related', "宗教相关性任务")
    log_statistics(df, 'is_hate_speech', "仇恨言论检测任务")

    # 保存
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    logger.info(f"预测结果已保存至: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()