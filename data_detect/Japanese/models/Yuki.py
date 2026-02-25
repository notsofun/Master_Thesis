import sys
import os
from pathlib import Path
# 获取当前脚本的根目录（Master_Thesis）
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)
from data_detect.base import BaseModel
from data_detect.Japanese.constants import ModelName, ModelInfo, HateScore
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

class YukiModel(BaseModel):
    def __init__(self, device="cpu"):
        model_info = ModelName.YUKI.value
        
        # 1. 统一获取设备对象
        self.device = self._get_clean_device(device)
        
        # 2. 加载 Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_info.model,
            use_fast=True,
            trust_remote_code=True,
        )

        # yuki.py 在 data_detect/Japanese/models/ 下
        self.current_dir = Path(__file__).parent.absolute()
        # 找到 pth 文件的准确位置 (向上退一级到 Japanese/ 下)
        self.pth_path = self.current_dir.parent / "classification_head.pth"
        is_cuda = self.device.type == "cuda"
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_info.model,
            torch_dtype=torch.float32,
            # 如果是 CUDA，用 auto 自动分配；如果是 CPU，设为 None
            device_map="auto" if is_cuda else None,
            trust_remote_code=True,
        )

        # 4. 只有在【非】自动分配（即 CPU 模式）时，才需要手动 .to()
        if not is_cuda:
            self.model.to(self.device)
            
        self.model.eval()

    def score(self, text: str) -> dict:
        """
        返回预测标签和置信度
        Returns: {"label": 0/1, "prob": float(0.0-1.0)}
        """
        if not self.pth_path.exists():
                    raise FileNotFoundError(f"找不到权重文件，请检查路径: {self.pth_path}")

        # 使用绝对路径加载
        head_weights = torch.load(self.pth_path, map_location=self.device)
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