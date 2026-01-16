from data_detect.base import BaseModel
from constants import ModelName, ModelInfo, HateScore
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import numpy as np
import torch

class NormalModel(BaseModel):
    def __init__(self, model_info: ModelInfo,  device="cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_info.tokenizer)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_info.model, trust_remote_code=True)

    def score(self, text:str) -> int:
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)

        result = self.model(**inputs).logits

        pred_label = torch.argmax(result, dim=-1).item()

        return pred_label