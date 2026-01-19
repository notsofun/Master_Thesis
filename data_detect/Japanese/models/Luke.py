from data_detect.base import BaseModel
from data_detect.Japanese.constants import ModelName, ModelInfo, HateScore
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import numpy as np

class LukeModel(BaseModel):
    def __init__(self, device="cpu"):
        model_info = ModelName.LUKE.value
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_info.tokenizer)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_info.model, trust_remote_code=True)

    def score(self, text: str) -> int:
        inputs = self.tokenizer.encode_plus(text, return_tensors="pt")
        logits = self.model(
            inputs["input_ids"],
            inputs["attention_mask"]
        ).detach().numpy()[0][:3]

        minimum = np.min(logits)
        if minimum < 0:
            logits = logits - minimum
        score = logits / np.sum(logits)
        score = HateScore(non_attack=float(score[0]), gray_zone=float(score[1]), attack=float(score[2]))
        max_score = max(score.non_attack, score.gray_zone, score.attack)
        return 1 if max_score == score.attack else 0