import pandas as pd
import torch
import random
from torch.utils.data import Dataset

class MultiTaskDataset(Dataset):
    def __init__(self, dataframe, tokenizer, config, is_train=False, lang='zh'):
        # 1. 基础清理
        self.data = dataframe.dropna(subset=[config["rel_label_col"], config["hate_label_col"]]).reset_index(drop=True)
        self.tokenizer = tokenizer
        self.config = config
        self.is_train = is_train
        
        # 中日文标点库
        self.punctuations = {
            'zh': ["，", "。", "！", "？", "...", "、", "·"],
            'jp': ["、", "。", "！", "？", "…", "「", "」", "—"]
        }.get(lang, ["，", "。"])

        if is_train:
            print(f">>> [训练模式] 原始数据量: {len(self.data)}")
            
            # --- 第一步：全量 AEDA 翻倍 ---
            aeda_data = self.data.copy()
            aeda_data[config["text_col"]] = aeda_data[config["text_col"]].apply(self.aeda_augment)
            
            # 合并：一份原样 + 一份 AEDA
            self.data = pd.concat([self.data, aeda_data], ignore_index=True)
            print(f">>> 全量 AEDA 翻倍完成，当前总量: {len(self.data)}")

            # --- 第二步：对合并后的正样本进行过采样 ---
            def quick_binary(label):
                if isinstance(label, str):
                    label = label.strip()
                    if label in ["是", "1", "yes", "Yes", "1.0"]: return 1
                try:
                    if int(float(label)) == 1: return 1
                except: pass
                return 0

            is_hate = self.data[config["hate_label_col"]].apply(quick_binary)
            hate_samples = self.data[is_hate == 1]
            
            if len(hate_samples) > 0:
                # 因为前面已经翻倍了，这里 multiplier 如果设为 10，实际总正样本就是 129 * 2 * 10
                multiplier = 10 
                print(f">>> 正在对翻倍后的 {len(hate_samples)} 条正样本进行 {multiplier} 倍过采样...")
                repeated_samples = [hate_samples] * (multiplier - 1)
                self.data = pd.concat([self.data] + repeated_samples, ignore_index=True)
                
                # 随机打乱防止 Batch 聚集
                self.data = self.data.sample(frac=1, random_state=42).reset_index(drop=True)
                print(f">>> 最终训练集规模: {len(self.data)}")

    def aeda_augment(self, text):
        """AEDA: 随机在文本中插入标点"""
        if not isinstance(text, str) or len(text) < 5: return text
        chars = list(text)
        insert_count = random.randint(1, 3)
        for _ in range(insert_count):
            ins_idx = random.randint(0, len(chars))
            chars.insert(ins_idx, random.choice(self.punctuations))
        return "".join(chars)

    def str_to_binary(self, label):
        if isinstance(label, str):
            label = label.strip()
            if label in ["是", "1", "yes", "Yes", "1.0"]: return 1
            return 0
        try:
            return int(float(label))
        except: return 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = str(row[self.config["text_col"]])
        rel_label = self.str_to_binary(row[self.config["rel_label_col"]])
        hate_label = self.str_to_binary(row[self.config["hate_label_col"]])

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