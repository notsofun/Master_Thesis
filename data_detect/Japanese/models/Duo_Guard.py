from data_detect.base import BaseModel
from data_detect.Japanese.constants import ModelName, ModelInfo, HateScore
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

class DuoGuardModel(BaseModel):
    def __init__(self, device="cpu"):
        self.device = device
        self.model_info = ModelName.DUO_GUARD.value
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_info.tokenizer)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load the DuoGuard-0.5B model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_info.model,
            torch_dtype=torch.bfloat16
        ).to(self.device)

    def score(self, text):
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

        return 1 if max_index == hate_index else 0