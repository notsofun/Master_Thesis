from data_detect.base import BaseModel
from data_detect.Japanese.constants import ModelName, ModelInfo, HateScore
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

class KubotaModel(BaseModel):
    def __init__(self, device="cpu"):
        self.device = self._normalize_device(device)
        self.model_info = ModelName.KUBOTA.value
        # pipeline 的 device 参数: -1 表示 CPU, 0+ 表示 GPU 索引
        if self.device == "cpu":
            pipe_device = -1
        else:
            # 从 'cuda:0' 提取 GPU 索引
            gpu_id = int(self.device.split(":")[1]) if ":" in self.device else 0
            pipe_device = gpu_id
        self.pipe = pipeline(model=self.model_info.model, device=pipe_device)
    
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
        result = self.pipe(text)[0]   # 取第一个结果 dict
        rule = {
            "中傷性のない発言": 0,
            "脅迫的な発言": 1,
            "侮蔑的な発言": 2,
            "名誉を低下させる発言": 3,
        }

        label_str = result["label"]
        class_id = rule.get(label_str, 0)   # 找不到就当作 0
        score = result.get("score", 0.5)
        
        # 如果类别是 2 或 3，则判为 1，否则 0
        label = 1 if class_id in (2, 3) else 0

        return {"label": label, "prob": float(score)}