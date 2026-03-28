import os, sys, torch, tqdm
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import KFold  # 核心：引入 KFold
from tqdm import tqdm
# 基础路径配置
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

# --- FocalLoss 与阈值搜索函数保持原样 ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        F_loss = alpha_t * (1 - pt) ** self.gamma * BCE_loss
        return torch.mean(F_loss) if self.reduction == 'mean' else torch.sum(F_loss)

def find_best_threshold(y_true, y_probs):
    best_threshold, best_f1 = 0.5, 0
    thresholds = np.linspace(0.01, 0.95, 95)
    for t in thresholds:
        y_pred = (y_probs >= t).astype(int)
        current_f1 = f1_score(y_true, y_pred, zero_division=0)
        if current_f1 > best_f1:
            best_f1, best_threshold = current_f1, t
    return best_threshold, best_f1

# ==========================================
# 5-Fold 交叉验证训练引擎
# ==========================================
def train_cross_validation():
    logger.info(f"开始 5-Fold 交叉验证训练流程...")
    
    # 1. 加载全量数据（建议把之前的 train 和 val 合并成一个文件，或者在这里 concat）
    # 在 train_cross_validation 函数开头
    df1 = pd.read_csv(CONFIG["train_csv_path"])
    df2 = pd.read_csv(CONFIG["val_csv_path"])
    all_df = pd.concat([df1, df2]).reset_index(drop=True)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
    
    fold_results = [] # 存储每一折的最佳成绩

    # 2. 循环每一折
    for fold, (train_idx, val_idx) in enumerate(kf.split(all_df)):
        logger.info(f"\n" + "="*30 + f" 正在训练 Fold {fold+1}/5 " + "="*30)
        
        train_df = all_df.iloc[train_idx].reset_index(drop=True)
        val_df = all_df.iloc[val_idx].reset_index(drop=True)
        
        # 重新初始化模型、优化器等（每折都是全新的开始）
        model = MultiTaskClassifier(CONFIG["model_name"]).to(CONFIG["device"])
        
        # 针对这一折的数据应用你的过采样逻辑
        train_dataset = MultiTaskDataset(train_df, tokenizer, CONFIG, is_train=True, lang='zh')
        val_dataset = MultiTaskDataset(val_df, tokenizer, CONFIG, is_train=False, lang='zh')
        
        train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"])
        
        # 为当前折计算 Alpha
        alpha_rel = 1 - train_df[CONFIG["rel_label_col"]].mean()
        alpha_hate = 1 - train_df[CONFIG["hate_label_col"]].mean()
        criterion_rel = FocalLoss(alpha=alpha_rel, gamma=2)
        criterion_hate = FocalLoss(alpha=alpha_hate, gamma=2)

        optimizer = AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=0.05)
        total_steps = len(train_loader) * CONFIG["epochs"]
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps*0.1), num_training_steps=total_steps)

        best_fold_score = -1
        fold_patience_counter = 0
        fold_save_path = CONFIG["save_path"].replace(".pt", f"_fold_{fold}.pt")

        for epoch in range(CONFIG["epochs"]):
            # --- 训练逻辑 (保持原样) ---
            model.train()
            for batch in tqdm(train_loader, desc=f"Fold {fold+1} Epoch {epoch+1} [Train]"):
                optimizer.zero_grad()
                input_ids, mask = batch['input_ids'].to(CONFIG["device"]), batch['attention_mask'].to(CONFIG["device"])
                rel_labels, hate_labels = batch['rel_labels'].to(CONFIG["device"]), batch['hate_labels'].to(CONFIG["device"])
                
                rel_logits, hate_logits = model(input_ids, mask)
                loss = criterion_rel(rel_logits, rel_labels) + 2.0 * criterion_hate(hate_logits, hate_labels)
                
                loss.backward()
                optimizer.step()
                scheduler.step()

            # --- 验证与阈值优化逻辑 ---
            model.eval()
            val_results = {"rel_true": [], "rel_probs": [], "hate_true": [], "hate_probs": []}
            with torch.no_grad():
                for batch in val_loader:
                    input_ids, mask = batch['input_ids'].to(CONFIG["device"]), batch['attention_mask'].to(CONFIG["device"])
                    rel_logits, hate_logits = model(input_ids, mask)
                    val_results["rel_probs"].extend(torch.sigmoid(rel_logits).cpu().numpy())
                    val_results["hate_probs"].extend(torch.sigmoid(hate_logits).cpu().numpy())
                    val_results["rel_true"].extend(batch['rel_labels'].cpu().numpy())
                    val_results["hate_true"].extend(batch['hate_labels'].cpu().numpy())

            best_rel_t, best_rel_f1 = find_best_threshold(np.array(val_results["rel_true"]), np.array(val_results["rel_probs"]))
            best_hate_t, best_hate_f1 = find_best_threshold(np.array(val_results["hate_true"]), np.array(val_results["hate_probs"]))
            
            combined_f1 = (best_rel_f1 + best_hate_f1) / 2
            logger.info(f"Fold {fold+1} Epoch {epoch+1}: Rel_F1={best_rel_f1:.4f}, Hate_F1={best_hate_f1:.4f}, Combined={combined_f1:.4f}")

            # 保存当前折的最佳模型
            if combined_f1 > best_fold_score:
                best_fold_score = combined_f1
                fold_patience_counter = 0
                torch.save(model.state_dict(), fold_save_path)
                logger.info(f">>> Fold {fold+1} 指标提升，模型保存至 {fold_save_path}")
            else:
                fold_patience_counter += 1
                if fold_patience_counter >= 3: # Fold 内早停
                    logger.info(f"Fold {fold+1} 触发早停。")
                    break
        
        fold_results.append(best_fold_score)

    # 3. 输出 5-Fold 最终统计成绩
    logger.info("\n" + "#"*50)
    logger.info(f"5-Fold 训练完成！")
    logger.info(f"各折最佳成绩: {fold_results}")
    logger.info(f"平均 Combined F1: {np.mean(fold_results):.4f} (+/- {np.std(fold_results):.4f})")
    logger.info("#"*50)

if __name__ == "__main__":
    try:
        train_cross_validation()
    except Exception as e:
        logger.error(f"崩溃了: {e}")
        sys.exit(1)