# python model_train/classifier/train.py
from torch.utils.data import Dataset, DataLoader
import torch, os, sys
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from tqdm import tqdm
import pandas as pd
import torch.nn as nn
from torch.nn import functional as F

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir)) 

if root_dir not in sys.path:
    sys.path.append(root_dir)

from scripts.set_logger import setup_logging
from config import CONFIG
from dataset import MultiTaskDataset
from models.base_model import MultiTaskClassifier

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: [batch, 1], targets: [batch, 1]
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        
        # --- 关键改进：根据标签选择 alpha_t ---
        # 如果 targets 是 1，权重是 self.alpha
        # 如果 targets 是 0，权重是 1 - self.alpha
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        
        F_loss = alpha_t * (1 - pt) ** self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        return F_loss

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

    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
    model = MultiTaskClassifier(CONFIG["model_name"]).to(CONFIG["device"])

    train_dataset = MultiTaskDataset(train_df, tokenizer, CONFIG)
    val_dataset = MultiTaskDataset(val_df, tokenizer, CONFIG)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"])
    
    # --- 自动计算 Alpha ---
    alpha_rel = 1 - train_df[CONFIG["rel_label_col"]].mean()
    alpha_hate = 1 - train_df[CONFIG["hate_label_col"]].mean()
    criterion_rel = FocalLoss(alpha=alpha_rel, gamma=3)
    criterion_hate = FocalLoss(alpha=alpha_hate, gamma=3)

    # ================= 修改点 1: 学习率与优化器设置 =================
    # 建议将 CONFIG["lr"] 调小，例如 5e-6 或 1e-5
    optimizer = AdamW(model.parameters(), lr=CONFIG["lr"])
    
    total_steps = len(train_loader) * CONFIG["epochs"]
    # 设置 Warmup Step 为总步数的 10%
    warmup_steps = int(total_steps * 0.1)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps
    )
    # ============================================================

    best_score = -1
    # ================= 修改点 2: 提前停止计数器 =================
    patience = 3  # 如果连续 3 轮指标不涨，就停止
    no_improve_epochs = 0
    # ============================================================
    
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
            scheduler.step()  # 更新学习率
            total_loss += loss.item()
    
        # --- 验证阶段 ---
        model.eval()
        val_results = {"rel_true": [], "rel_pred": [], "hate_true": [], "hate_pred": [], "texts": []}
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                input_ids = batch['input_ids'].to(CONFIG["device"])
                attention_mask = batch['attention_mask'].to(CONFIG["device"])
                
                rel_logits, hate_logits = model(input_ids, attention_mask)
                
                # 解码用于错误分析
                batch_texts = [tokenizer.decode(g, skip_special_tokens=True) for g in input_ids]
                val_results["texts"].extend(batch_texts)
                
                val_results["rel_true"].extend(batch['rel_labels'].cpu().numpy())
                val_results["rel_pred"].extend((torch.sigmoid(rel_logits).cpu().numpy() > 0.5).astype(int))
                val_results["hate_true"].extend(batch['hate_labels'].cpu().numpy())
                val_results["hate_pred"].extend((torch.sigmoid(hate_logits).cpu().numpy() > 0.3).astype(int))

        # 计算指标
        rel_f1 = f1_score(val_results["rel_true"], val_results["rel_pred"])
        hate_f1 = f1_score(val_results["hate_true"], val_results["hate_pred"])
        metrics = {
            "rel_f1": rel_f1,
            "hate_f1": hate_f1,
            "combined_f1": (rel_f1 + hate_f1) / 2,
            "hate_recall": recall_score(val_results["hate_true"], val_results["hate_pred"]),
            "total_loss": total_loss / len(train_loader),
            "lr": optimizer.param_groups[0]['lr'] # 记录当前学习率
        }
        
        logger.info(f"Epoch {epoch+1} 结果: {metrics}")

        # --- 保存逻辑与提前停止逻辑 ---
        current_score = metrics[CONFIG["monitor_metric"]]
        
        if current_score > best_score:
            best_score = current_score
            no_improve_epochs = 0 # 重置计数器
            
            save_dir = os.path.dirname(CONFIG["save_path"])
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            torch.save(model.state_dict(), CONFIG["save_path"])
            logger.info(f">>> 指标提升，模型已保存。")

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

        else:
            no_improve_epochs += 1
            logger.info(f"--- 指标未提升 (已持续 {no_improve_epochs} 轮) ---")

        # 触发提前停止
        if no_improve_epochs >= patience:
            logger.info(f"早停触发！连续 {patience} 轮没有进步，停止训练以防过拟合。")
            break
            
if __name__ == "__main__":
    try:
        logger.info("开始训练流程...")
        train()
        logger.info("训练正常结束。")
    except Exception as e:
        logger.error("!!! 训练过程中发生崩溃 !!!")
        logger.exception(e) 
        sys.exit(1)