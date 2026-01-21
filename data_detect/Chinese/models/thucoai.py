from data_detect.base import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
import torch

class THUCOAIModel(BaseModel):
    """THU-COAI RoBERTa 模型 wrapper (二分类: 0=非仇恨, 1=仇恨)"""
    
    def __init__(self, device="cpu"):
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained('thu-coai/roberta-base-cold')
        self.model = BertForSequenceClassification.from_pretrained('thu-coai/roberta-base-cold')
        self.model.to(device)
        self.model.eval()

    def score(self, text: str) -> dict:
        """
        返回预测标签和置信度
        Returns: {"label": 0/1, "prob": float(0.0-1.0)}
        """
        model_input = self.tokenizer(
            [text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            model_output = self.model(**model_input, return_dict=False)
            logits = model_output[0].cpu()[0]  # 获取第一个样本的 logits
            
            # 使用 softmax 获取概率
            probs = torch.softmax(logits, dim=-1)
            
            # 预测标签（取概率最大的类别）
            pred_label = torch.argmax(logits, dim=-1).item()
            
            # 该标签的置信度
            confidence = float(probs[pred_label])
        
        return {"label": pred_label, "prob": confidence}
