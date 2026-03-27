# python model_eval/classifier/eval_original.py
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, recall_score, precision_score, classification_report
from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification


class ChineseEvaluator:
    """封装 THU-COAI RoBERTa 逻辑"""
    def __init__(self, device):
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained('thu-coai/roberta-base-cold')
        self.model = BertForSequenceClassification.from_pretrained('thu-coai/roberta-base-cold')
        self.model.to(device).eval()

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0].cpu()
            probs = torch.softmax(logits, dim=-1)
            pred_label = torch.argmax(logits, dim=-1).item()
        return pred_label, float(probs[1]) # 返回标签和属于'1'(hate)的概率

class JapaneseEvaluator:
    """封装 LUKE 逻辑"""
    def __init__(self, device):
        self.device = device
        # 假设 model_info.tokenizer/model 路径如下
        model_path = "kubota/luke-large-defamation-detection-japanese"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, trust_remote_code=True)
        self.model.to(device).eval()

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(inputs['input_ids'], inputs['attention_mask'])
            logits = outputs.logits[0][:3].detach().cpu().numpy() if hasattr(outputs, "logits") else outputs[0][:3].detach().cpu().numpy()
        
        # 你的归一化逻辑
        minimum = np.min(logits)
        if minimum < 0: logits = logits - minimum
        probs = logits / np.sum(logits)
        
        # 你的判定逻辑: 2是attack则为1(hate)，否则为0
        pred_label = 1 if np.argmax(probs) == 2 else 0
        conf = float(probs[2] if np.argmax(probs) == 2 else probs[0])
        return pred_label, conf
        

def run_eval():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 配置不同语言的测试任务
    tasks = [
        {
            "name": "Chinese_COLD",
            "evaluator_class": ChineseEvaluator,
            "test_csv": "model_train/classifier/data/final_annotated_Chinese_val.csv",  # 填入你的中文测试集路径
            "label_col": "hate_speech",
            "text_col": "text"
        },
        {
            "name": "Japanese_LUKE",
            "evaluator_class": JapaneseEvaluator,
            "test_csv": "model_train/classifier/data/final_annotated_Japanese_val.csv", # 填入你的日文测试集路径
            "label_col": "hate_speech",              # 你的 CSV 中标注标签的列名
            "text_col": "text"
        }
    ]

    all_summary = []

    for task in tasks:
        print(f"\n>>> 正在验证: {task['name']}")
        
        # 1. 初始化对应的模型
        engine = task['evaluator_class'](device)
        
        # 2. 读取数据
        df = pd.read_csv(task['test_csv'])
        
        y_true = []
        y_pred = []
        
        # 3. 循环推理 (因为逻辑中包含复杂的自定义后处理，这里采用简单循环)
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Testing {task['name']}"):
            label, _ = engine.predict(str(row[task['text_col']]))
            y_true.append(int(row[task['label_col']]))
            y_pred.append(label)
        
        # 4. 指标计算
        f1 = f1_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        pre = precision_score(y_true, y_pred)
        
        print(f"\n[{task['name']} 结果]")
        print(classification_report(y_true, y_pred))
        
        all_summary.append({
            "Model": task['name'],
            "F1": round(f1, 4),
            "Recall": round(rec, 4),
            "Precision": round(pre, 4)
        })
        
        # 释放显存
        del engine
        torch.cuda.empty_cache()

    # 5. 打印对比表格
    print("\n" + "="*30)
    print(" 最终评估汇总 ")
    print("="*30)
    summary_df = pd.DataFrame(all_summary)
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    run_eval()