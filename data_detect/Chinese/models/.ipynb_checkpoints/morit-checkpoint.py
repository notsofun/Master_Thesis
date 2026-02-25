import sys
import os

# 获取当前脚本的根目录（Master_Thesis）
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)
from data_detect.base import BaseModel
from transformers import pipeline

class MoritModel(BaseModel):
    """Morit 零样本分类模型 wrapper (使用中文标签进行分类)"""
    
    def __init__(self, device="cpu"):
        """
        Args:
            device: 设备 (cpu 或 cuda, 如 cuda:0)
        """
        self.device = device
        self.candidate_labels = ["仇恨言论", "正常言论"]
        
        # 智能解析 device 参数
        if device == "cpu":
            pipeline_device = -1
        elif isinstance(device, int):
            pipeline_device = device
        elif "cuda" in device:
            # 尝试从 "cuda:1" 中提取 1，如果没有指定数字则默认为 0
            try:
                pipeline_device = int(device.split(":")[-1]) if ":" in device else 0
            except ValueError:
                pipeline_device = 0
        else:
            pipeline_device = -1

        self.classifier = pipeline(
            "zero-shot-classification",
            model="morit/chinese_xlm_xnli",
            device=pipeline_device
        )

    def score(self, text: str) -> dict:
        """
        返回预测标签和置信度
        Returns: {"label": 0/1, "prob": float(0.0-1.0)}
        """
        try:
            result = self.classifier(text, self.candidate_labels, multi_class=False)
            
            # 获取得分最高的标签和分数
            top_label = result['labels'][0]
            top_score = result['scores'][0]
            
            # 将标签映射到 0 或 1
            # "仇恨言论" -> 1, "正常言论" -> 0
            label = 1 if top_label == "仇恨言论" else 0
            
            return {"label": label, "prob": float(top_score)}
        except Exception as e:
            raise RuntimeError(f"Error during prediction: {e}")
