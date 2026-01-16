# hate_detector.py
# 这个是可扩展的
from dataclasses import dataclass
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, AutoModelForCausalLM
from tqdm import tqdm
from enum import Enum
import torch, os
from huggingface_hub import hf_hub_download
from typing import List, Dict, Optional, Any
from data_detect.Japanese.constants import ModelInfo, ModelName, HateScore
from data_detect.Japanese.factory import ModelFactory

current_dir = os.path.dirname(os.path.abspath(__file__))


class HateSpeechDetector:
    def __init__(self,logger, model_specs: list[ModelInfo], device="cpu"):
        """
        model_specs: list of dicts, each dict: {"name": "friendly name", "tokenizer_base": "...", "model": "..."}
        """
        import torch
        self.torch = torch
        self.device = device
        self.models = {}
        self.logger = logger
        for spec in model_specs:
            self.logger.info(f"[INFO] Loading model {spec.name} -> {spec.model or spec.tokenizer}")
            wrapper = ModelFactory.create_model(spec, device=self.device)
            self.models[spec.name] = wrapper

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
            for idx, text in enumerate(tqdm(texts, desc=f"Running {name}", leave=False, ncols=100)):
                try:
                    hs = wrapper.score_text(text)
                    rows.append({"text": text, "HS": hs})
                except Exception as e:
                    self.logger.warning(
                        f"[{name}] Error processing text idx={idx}: {text[:50]}... - {e}"
                    )
                    rows.append({"text": text, "HS": None, "error": str(e)})

            df = pd.DataFrame(rows)
            outputs[name] = df

            try:
                hs_series = pd.to_numeric(df["HS"], errors="coerce").fillna(0).astype(int)
                hs_count = hs_series.sum()
                valid = df["HS"].notna().sum()
                ratio = (hs_count / valid * 100) if valid > 0 else 0
            except Exception as e:
                self.logger.error(f"[{name}] Error computing summary stats: {e}")
                hs_count = valid = ratio = 0

            self.logger.info(
                f"[{name}] Finished. Processed: {valid}/{total} valid texts. "
                f"Hate Speech: {hs_count} ({ratio:.2f}%)."
            )

        return outputs

