from torch.utils.data import Dataset, DataLoader
from transformers import  AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from tqdm import tqdm
import logging
from config import CONFIG
from dataset import MultiTaskDataset
from models.base_model import MultiTaskClassifier
import pandas as pd
import torch
import torch.nn as nn


# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==========================================
# 4. 训练与验证引擎
# ==========================================
def train():
    # 加载数据
    df = pd.read_csv(CONFIG["csv_path"])
    train_df = df[df[CONFIG["split_col"]] == CONFIG["train_val_split"][0]].reset_index(drop=True)
    val_df = df[df[CONFIG["split_col"]] == CONFIG["train_val_split"][1]].reset_index(drop=True)
    
    logger.info(f"训练集大小: {len(train_df)}, 验证集大小: {len(val_df)}")

    # 计算仇恨文本的权重 (针对 134/2000 的不平衡)
    # pos_weight = negative_samples / positive_samples
    neg_count = (train_df[CONFIG["hate_label_col"]] == 0).sum()
    pos_count = (train_df[CONFIG["hate_label_col"]] == 1).sum()
    hate_weight = torch.tensor([neg_count / max(pos_count, 1)]).to(CONFIG["device"])
    logger.info(f"自动设置仇恨分类权重 (pos_weight): {hate_weight.item():.2f}")

    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
    model = MultiTaskClassifier(CONFIG["model_name"]).to(CONFIG["device"])

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
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG["epochs"]} [Train]"):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(CONFIG["device"])
            attention_mask = batch['attention_mask'].to(CONFIG["device"])
            rel_labels = batch['rel_labels'].to(CONFIG["device"])
            hate_labels = batch['hate_labels'].to(CONFIG["device"])
            
            rel_logits, hate_logits = model(input_ids, attention_mask)
            
            loss_rel = criterion_rel(rel_logits, rel_labels)
            loss_hate = criterion_hate(hate_logits, hate_labels)
            
            # 总损失平衡 (可以根据需要调整权重)
            loss = loss_rel + loss_hate 
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # --- 验证阶段 ---
        model.eval()
        val_results = {"rel_true": [], "rel_pred": [], "hate_true": [], "hate_pred": []}
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                input_ids = batch['input_ids'].to(CONFIG["device"])
                attention_mask = batch['attention_mask'].to(CONFIG["device"])
                
                rel_logits, hate_logits = model(input_ids, attention_mask)
                
                # Sigmoid 激活并转为 0/1 判定
                val_results["rel_true"].extend(batch['rel_labels'].cpu().numpy())
                val_results["rel_pred"].extend(torch.sigmoid(rel_logits).cpu().numpy() > 0.5)
                val_results["hate_true"].extend(batch['hate_labels'].cpu().numpy())
                val_results["hate_pred"].extend(torch.sigmoid(hate_logits).cpu().numpy() > 0.5)

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
        # 默认监控 hate_f1，因为数据不平衡时 Acc 没有参考价值
        current_score = metrics[CONFIG["monitor_metric"]]
        
        if current_score > best_score:
            best_score = current_score
            torch.save(model.state_dict(), CONFIG["save_path"])
            logger.info(f">>> 检测到更好的 {CONFIG['monitor_metric']}: {best_score:.4f}, 模型已保存。")

if __name__ == "__main__":
    train()