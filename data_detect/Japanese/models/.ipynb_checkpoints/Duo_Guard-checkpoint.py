import sys
import os

# 获取当前脚本的根目录（Master_Thesis）
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)
from data_detect.base import BaseModel
from data_detect.Japanese.constants import ModelName, ModelInfo, HateScore
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

class DuoGuardModel(BaseModel):
    def __init__(self, device="cpu"):
        self.device = self._normalize_device(device)
        self.model_info = ModelName.DUO_GUARD.value
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_info.tokenizer)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load the DuoGuard-0.5B model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_info.model,
            torch_dtype=torch.bfloat16
        ).to(self.device)
        self.model.eval()
    
    def _normalize_device(self, device):
        """将device标准化为 'cpu', 'cuda:0' 等格式"""
        if device == "cpu":
            return "cpu"
        elif device.startswith("cuda"):
            if not torch.cuda.is_available():
                return "cpu"
            return "cuda:0" if device == "cuda" else device
        else:
            return "cpu"

    def score(self, text: str) -> dict:
        """
        返回预测标签和置信度
        Returns: {"label": 0/1, "prob": float(0.0-1.0)}
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.sigmoid(logits)

        category_names = [
            "Violent crimes",
            "Non-violent crimes",
            "Sex-related crimes",
            "Child sexual exploitation",
            "Specialized advice",
            "Privacy",
            "Intellectual property",
            "Indiscriminate weapons",
            "Hate",
            "Suicide and self-harm",
            "Sexual content",
            "Jailbreak prompts",
        ]

        prob_vector = probabilities[0].tolist()
        max_index = prob_vector.index(max(prob_vector))
        hate_index = category_names.index("Hate")
        
        # 判断是否为仇恨
        label = 1 if max_index == hate_index else 0
        
        # 返回对应标签的置信度
        confidence = prob_vector[hate_index] if label == 1 else (1.0 - prob_vector[hate_index])

        return {"label": label, "prob": float(confidence)}