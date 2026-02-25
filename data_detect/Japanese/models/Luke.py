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

class LukeModel(BaseModel):
    def __init__(self, device="cpu"):
        model_info = ModelName.LUKE.value
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
        inputs = self.tokenizer.encode_plus(text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            logits = self.model(
                inputs["input_ids"],
                inputs["attention_mask"]
            ).logits[0][:3]  # 获取前三个类别的 logits

        inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512
            ).to(self.device)
            
        with torch.no_grad():
            # 2. 使用 **inputs 自动解包字典，这样返回的对象一定包含 .logits
            outputs = self.model(**inputs)
            logits = outputs.logits[0][:3]

        logits_np = logits.detach().cpu().numpy()
        minimum = np.min(logits_np)
        if minimum < 0:
            logits_np = logits_np - minimum
        
        # 归一化为概率分布
        probs = logits_np / np.sum(logits_np)
        scores = HateScore(
            non_attack=float(probs[0]),
            gray_zone=float(probs[1]),
            attack=float(probs[2])
        )
        
        # 判断是否为仇恨言论（attack 类别概率最高）
        max_score = max(scores.non_attack, scores.gray_zone, scores.attack)
        label = 1 if max_score == scores.attack else 0
        
        # 返回该标签的置信度
        confidence = scores.attack if label == 1 else scores.non_attack
        
        return {"label": label, "prob": float(confidence)}