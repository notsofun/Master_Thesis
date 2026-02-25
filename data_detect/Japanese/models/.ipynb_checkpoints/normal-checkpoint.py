import sys
import os

# 获取当前脚本的根目录（Master_Thesis）
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)
from data_detect.base import BaseModel
from data_detect.Japanese.constants import ModelName, ModelInfo, HateScore
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import numpy as np
import torch

class NormalModel(BaseModel):
    def __init__(self, model_info: ModelInfo,  device="cpu"):
        self.device = self._normalize_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_info.tokenizer)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_info.model, trust_remote_code=True)
        self.model.to(self.device)
        self.model.eval()
    
    def _normalize_device(self, device):
        """将device标准化为 'cpu', 'cuda:0' 等格式"""
        if device == "cpu":
            return "cpu"
        elif device.startswith("cuda"):
            return "cuda:0" if device == "cuda" else device
        else:
            return "cpu"

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