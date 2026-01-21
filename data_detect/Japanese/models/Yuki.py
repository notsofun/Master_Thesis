from data_detect.base import BaseModel
from data_detect.Japanese.constants import ModelName, ModelInfo, HateScore
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

class YukiModel(BaseModel):
    def __init__(self, device="cpu"):
        model_info = ModelName.YUKI.value
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_info.model,
            use_fast=True,
            trust_remote_code=True,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_info.model,
            torch_dtype=torch.float32,   # 强制全精度
            device_map={"": device},      # 绑定到 GPU/CPU
            trust_remote_code=True,
        ).eval()

    def score(self, text: str) -> dict:
        """
        返回预测标签和置信度
        Returns: {"label": 0/1, "prob": float(0.0-1.0)}
        """
        head_weights = torch.load("classification_head.pth", map_location=self.device)
        head = torch.nn.Linear(1, 1, bias=False).to(self.device)
        head.weight.data = head_weights

        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).to(self.device)

        with torch.no_grad():
            out = self.model(**inputs).logits
            out = out.to(head.weight.dtype)   # dtype 对齐
            logits = head(out[:, -1])
            
            # 使用 sigmoid 获取置信度（0-1 范围）
            confidence = torch.sigmoid(logits[0]).item()
            
            # 判断标签（阈值为 0.5）
            threshold = 0.5
            label = 1 if confidence > threshold else 0

        return {"label": label, "prob": float(confidence)}