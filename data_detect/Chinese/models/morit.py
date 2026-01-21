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
        
        # 初始化零样本分类器
        device_id = 0 if device == "cpu" else int(device.split(":")[-1])
        self.classifier = pipeline(
            "zero-shot-classification",
            model="morit/chinese_xlm_xnli",
            device=device_id if device != "cpu" else -1
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
