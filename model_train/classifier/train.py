# python model_train/classifier/train.py
from torch.utils.data import Dataset, DataLoader
import torch, os, sys
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from tqdm import tqdm
import pandas as pd
import torch.nn as nn

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir)) 

if root_dir not in sys.path:
    sys.path.append(root_dir)

from scripts.set_logger import setup_logging
from config import CONFIG
from dataset import MultiTaskDataset
from models.base_model import MultiTaskClassifier

# 初始化日志
logger, LOG_FILE_PATH = setup_logging()

# ==========================================
# 4. 训练与验证引擎
# ==========================================
def train():
    logger.info(f"日志将保存至: {LOG_FILE_PATH}")
    # 加载数据
    df = pd.read_csv(CONFIG["csv_path"])
    train_df = df[df[CONFIG["split_col"]] == CONFIG["train_val_split"][0]].reset_index(drop=True)
    val_df = df[df[CONFIG["split_col"]] == CONFIG["train_val_split"][1]].reset_index(drop=True)
    
    logger.info(f"训练集大小: {len(train_df)}, 验证集大小: {len(val_df)}")

    # 计算仇恨文本的权重
    neg_count = (train_df[CONFIG["hate_label_col"]] == 0).sum()
    pos_count = (train_df[CONFIG["hate_label_col"]] == 1).sum()
    hate_weight = torch.tensor([neg_count / max(pos_count, 1)]).to(CONFIG["device"])
    logger.info(f"自动设置仇恨分类权重 (pos_weight): {hate_weight.item():.2f}")

    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
    logger.info(f"本次微调的基座模型是{CONFIG['model_name']}")
    model = MultiTaskClassifier(CONFIG["model_name"]).to(CONFIG["device"])
    logger.info(f"我们使用该设备{CONFIG['device']}")

    train_dataset = MultiTaskDataset(train_df, tokenizer, CONFIG)
    val_dataset = MultiTaskDataset(val_df, tokenizer, CONFIG)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"])

    # 损失函数与优化器
    criterion_rel = nn.BCEWithLogitsLoss()
    criterion_hate = nn.BCEWithLogitsLoss(pos_weight=hate_weight)
    optimizer = AdamW(model.parameters(), lr=CONFIG["lr"])
    
    best_score = -1
    
    for epoch in range(CONFIG["epochs"]):
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]"):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(CONFIG["device"])
            attention_mask = batch['attention_mask'].to(CONFIG["device"])
            rel_labels = batch['rel_labels'].to(CONFIG["device"])
            hate_labels = batch['hate_labels'].to(CONFIG["device"])
            
            rel_logits, hate_logits = model(input_ids, attention_mask)
            
            loss_rel = criterion_rel(rel_logits, rel_labels)
            loss_hate = criterion_hate(hate_logits, hate_labels)
            
            loss = loss_rel + loss_hate 
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # --- 验证阶段 ---
        model.eval()
        # 新增 val_texts 用来记录原始文本
        val_results = {"rel_true": [], "rel_pred": [], "hate_true": [], "hate_pred": [], "texts": []}
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                input_ids = batch['input_ids'].to(CONFIG["device"])
                attention_mask = batch['attention_mask'].to(CONFIG["device"])
                
                rel_logits, hate_logits = model(input_ids, attention_mask)
                
                # 解码文本并存入结果
                batch_texts = [tokenizer.decode(g, skip_special_tokens=True) for g in input_ids]
                val_results["texts"].extend(batch_texts)
                
                val_results["rel_true"].extend(batch['rel_labels'].cpu().numpy())
                val_results["rel_pred"].extend((torch.sigmoid(rel_logits).cpu().numpy() > 0.5).astype(int))
                val_results["hate_true"].extend(batch['hate_labels'].cpu().numpy())
                val_results["hate_pred"].extend((torch.sigmoid(hate_logits).cpu().numpy() > 0.5).astype(int))

        # 计算指标
        metrics = {
            "rel_acc": accuracy_score(val_results["rel_true"], val_results["rel_pred"]),
            "rel_f1": f1_score(val_results["rel_true"], val_results["rel_pred"]),
            "hate_acc": accuracy_score(val_results["hate_true"], val_results["hate_pred"]),
            "hate_f1": f1_score(val_results["hate_true"], val_results["hate_pred"]),
            "hate_recall": recall_score(val_results["hate_true"], val_results["hate_pred"]),
            "total_loss": total_loss / len(train_loader)
        }
        
        logger.info(f"Epoch {epoch+1} 结果: {metrics}")

        # --- 保存逻辑 ---
        current_score = metrics[CONFIG["monitor_metric"]]
        
        if current_score > best_score:
            best_score = current_score
            
            save_dir = os.path.dirname(CONFIG["save_path"])
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            
            torch.save(model.state_dict(), CONFIG["save_path"])
            logger.info(f">>> 检测到更好的 {CONFIG['monitor_metric']}: {best_score:.4f}, 模型已保存。")

            # ==========================================
            # 新增：错误分析导出逻辑 (仅针对 Hate 任务)
            # ==========================================
            error_analysis_list = []
            for t, true_val, pred_val in zip(val_results["texts"], val_results["hate_true"], val_results["hate_pred"]):
                if true_val != pred_val:
                    error_analysis_list.append({
                        "Text": t,
                        "True_Hate": int(true_val),
                        "Pred_Hate": int(pred_val),
                        "Error_Type": "漏报(FN)" if true_val == 1 else "误报(FP)"
                    })
            
            if error_analysis_list:
                error_df = pd.DataFrame(error_analysis_list)
                error_log_path = os.path.join(save_dir, f"error_analysis_epoch_{epoch+1}.csv")
                error_df.to_csv(error_log_path, index=False, encoding='utf_8_sig')
                logger.info(f"!!! 已导出错误样本至: {error_log_path}")
            # ==========================================

if __name__ == "__main__":
    try:
        logger.info("开始训练流程...")
        train()
        logger.info("训练正常结束。")
    except Exception as e:
        logger.error("!!! 训练过程中发生崩溃 !!!")
        logger.exception(e) 
        sys.exit(1)