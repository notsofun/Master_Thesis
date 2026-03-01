import pandas as pd
import torch
from torch.utils.data import Dataset

class MultiTaskDataset(Dataset):
    def __init__(self, dataframe, tokenizer, config):
        # --- 核心修改：在初始化时直接剔除标签缺失的行 ---
        initial_len = len(dataframe)
        # 只要 rel_label 或 hate_label 中有一个是 NaN，就删掉该行
        self.data = dataframe.dropna(subset=[config["rel_label_col"], config["hate_label_col"]]).reset_index(drop=True)
        
        filtered_len = len(self.data)
        if initial_len > filtered_len:
            print(f">>> 已跳过 {initial_len - filtered_len} 行包含 NaN 的脏数据。")
            
        self.tokenizer = tokenizer
        self.config = config

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = str(row[self.config["text_col"]])
        rel_label = row[self.config["rel_label_col"]]
        hate_label = row[self.config["hate_label_col"]]

        def str_to_binary(label):
            # 由于已经在 init 删除了 NaN，这里只需要处理正常的转换逻辑
            if isinstance(label, str):
                label = label.strip()
                if label in ["是", "1", "yes", "Yes", "1.0"]:
                    return 1
                return 0
            try:
                return int(float(label))
            except:
                return 0

        rel_label = str_to_binary(rel_label)
        hate_label = str_to_binary(hate_label)

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.config["max_len"],
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'rel_labels': torch.tensor(rel_label, dtype=torch.float),
            'hate_labels': torch.tensor(hate_label, dtype=torch.float)
        }