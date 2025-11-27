# hate_detector.py
# 这个是可扩展的
from dataclasses import dataclass
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import logging, torch

@dataclass
class HateScore:
    non_attack: float
    gray_zone: float
    attack: float

class ModelWrapper:
    def __init__(self, base_model_name: str, fine_tuned_model: str = None, device="cpu"):
        """
        base_model_name: tokenizer base (e.g., 'studio-ousia/luke-japanese-base-lite')
        fine_tuned_model: if provided, path or HF id to the model weights to use for classification
        """
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        model_name = fine_tuned_model if fine_tuned_model else base_model_name
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True)
        self.device = device
        self.model.to(self.device)

    def score_text(self, text: str) -> HateScore:
        inputs = self.tokenizer.encode_plus(text, return_tensors="pt")
        logits = self.model(
            inputs["input_ids"],
            inputs["attention_mask"]
        ).detach().numpy()[0][:3]

        minimum = np.min(logits)
        if minimum < 0:
            logits = logits - minimum
        score = logits / np.sum(logits)
        return HateScore(non_attack=float(score[0]), gray_zone=float(score[1]), attack=float(score[2]))

    @staticmethod
    def is_hate(score: HateScore) -> int:
        max_score = max(score.non_attack, score.gray_zone, score.attack)
        return 1 if max_score == score.attack else 0

class HateSpeechDetector:
    def __init__(self,logger, model_specs: list, device="cpu"):
        """
        model_specs: list of dicts, each dict: {"name": "friendly name", "tokenizer_base": "...", "model": "..."}
        """
        import torch
        self.torch = torch
        self.device = device
        self.models = {}
        self.logger = logger
        for spec in model_specs:
            name = spec["name"]
            base = spec.get("tokenizer_base", spec.get("model", spec.get("base_model")))
            model_path = spec.get("model", None)
            self.logger.info(f"[INFO] Loading model {name} -> {model_path or base}")
            wrapper = ModelWrapper(base, fine_tuned_model=model_path, device=self.device)
            self.models[name] = wrapper

    def run_on_texts(self, texts):
        """
        texts: iterable of str
        return: dict of DataFrames keyed by model name with columns: text, HS
        """
        outputs = {}
        total = len(texts)

        for name, wrapper in self.models.items():
            self.logger.info(f"Start running model: {name} on {total} texts")

            rows = []
            for text in tqdm(texts, desc=f"Running {name}", leave=False):
                try:
                    score = wrapper.score_text(text)
                    hs = wrapper.is_hate(score)
                    rows.append({"text": text, "HS": hs})
                except Exception as e:
                    rows.append({"text": text, "HS": None, "error": str(e)})

            df = pd.DataFrame(rows)
            outputs[name] = df

            hs_series = pd.to_numeric(df["HS"], errors="coerce").fillna(0).astype(int)
            hs_count = hs_series.sum()
            valid = df["HS"].notna().sum()
            ratio = (hs_count / valid * 100) if valid > 0 else 0


            self.logger.info(
                f"[{name}] Finished. Processed: {valid}/{total} valid texts. "
                f"Hate Speech: {hs_count} ({ratio:.2f}%)."
            )

        return outputs

