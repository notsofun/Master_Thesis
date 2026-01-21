from data_detect.base import BaseModel
from data_detect.Japanese.constants import ModelName, ModelInfo, HateScore
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import numpy as np
import torch

class NormalModel(BaseModel):
    def __init__(self, model_info: ModelInfo,  device="cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_info.tokenizer)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_info.model, trust_remote_code=True)
        self.model.to(device)
        self.model.eval()

    def score(self, text: str) -> dict:
        """
        返回预测标签和置信度
        Returns: {"label": 0/1, "prob": float(0.0-1.0)}
        """
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).to(self.device)

        with torch.no_grad():
            result = self.model(**inputs).logits
            
            # 使用 softmax 获取概率
            probs = torch.softmax(result, dim=-1)
            
            # 预测标签
            pred_label = torch.argmax(result, dim=-1).item()
            
            # 该标签的置信度
            confidence = float(probs[0][pred_label])

        return {"label": pred_label, "prob": confidence}