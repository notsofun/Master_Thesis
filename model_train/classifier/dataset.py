from torch.utils.data import Dataset, DataLoader
import torch

# ==========================================
# 2. 数据集抽象
# ==========================================
class MultiTaskDataset(Dataset):
    def __init__(self, dataframe, tokenizer, config):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.config = config

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = str(row[self.config["text_col"]])
        rel_label = row[self.config["rel_label_col"]]
        hate_label = row[self.config["hate_label_col"]]

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