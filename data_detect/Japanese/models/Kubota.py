from data_detect.base import BaseModel
from data_detect.Japanese.constants import ModelName, ModelInfo, HateScore
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

class KubotaModel(BaseModel):
    def __init__(self, device="cpu"):
        self.device = device
        self.model_info = ModelName.KUBOTA.value
        self.pipe = pipeline(model=self.model_info.model)

    def score(self, text):
        result = self.pipe(text)[0]   # 取第一个结果 dict
        rule = {
            "中傷性のない発言": 0,
            "脅迫的な発言": 1,
            "侮蔑的な発言": 2,
            "名誉を低下させる発言": 3,
        }

        label = result["label"]
        class_id = rule.get(label, 0)   # 找不到就当作 0

        # 如果类别是 2 或 3，则判为 1，否则 0
        return 1 if class_id in (2, 3) else 0