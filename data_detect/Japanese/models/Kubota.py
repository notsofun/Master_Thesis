import sys
import os

# 获取当前脚本的根目录（Master_Thesis）
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)
from data_detect.base import BaseModel
from data_detect.Japanese.constants import ModelName
from transformers import pipeline

class KubotaModel(BaseModel):
    def __init__(self, device="cpu"):
        self.model_info = ModelName.KUBOTA.value

        self.device = self._get_clean_device(device)
        
        # 2. 转换给 pipeline 使用的设备格式
        # 如果是 cuda，提取其索引（若无索引则默认为 0）；如果是 cpu，设为 -1
        if self.device.type == "cuda":
            pipe_device = self.device.index if self.device.index is not None else 0
        else:
            pipe_device = -1

        # 3. 初始化 pipeline
        self.pipe = pipeline(
            model=self.model_info.model, 
            device=pipe_device,
            trust_remote_code=True # 建议加上，防止某些模型加载失败
        )
    
    def score(self, text: str) -> dict:
        """
        返回预测标签和置信度
        Returns: {"label": 0/1, "prob": float(0.0-1.0)}
        """
        result = self.pipe(text, truncation=True, max_length=512)[0]   # 取第一个结果 dict
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