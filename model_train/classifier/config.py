import torch

# ==========================================
# 1. 配置中心 (在此修改你的字段和参数)
# ==========================================
CONFIG = {
    "model_name": "studio-ousia/luke-japanese-base",  # 也可以换成 'hfl/chinese-roberta-wwm-ext'
    "csv_path": "your_data.csv",            # 你的CSV文件路径
    
    # 字段映射
    "text_col": "text_content",             # CSV中存放文本的列名
    "rel_label_col": "is_religion",         # CSV中宗教相关的列名 (1/0)
    "hate_label_col": "is_hate",             # CSV中仇恨文本的列名 (1/0)
    "split_col": "dataset_type",            # 区分训练/验证的列名
    "train_val_split": ("train", "val"),    # 对应的值：(训练集名称, 验证集名称)
    
    # 超参数
    "max_len": 64,                          # 短句多，64-128即可
    "batch_size": 16,
    "epochs": 10,
    "lr": 2e-5,
    "warmup_steps": 0,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    
    # 保存策略
    "monitor_metric": "hate_f1",            # 可选: 'hate_f1', 'hate_acc', 'rel_f1', 'total_loss'
    "save_path": "best_multitask_model.pt",
}