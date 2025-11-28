# 我持保留态度，改完还多了几十行……
from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Type
from datetime import datetime

import pandas as pd
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
    PreTrainedTokenizerBase,
)

# ---------- logging ----------
logger = logging.getLogger("hate_detector_refactored")
if not logger.handlers:
    handler = logging.StreamHandler()
    fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ---------- model spec ----------
@dataclass(frozen=True)
class ModelSpec:
    name: str
    tokenizer: Optional[str] = None
    model: Optional[str] = None
    score_method: Optional[str] = None  # optional, used if special handling required


# ---------- scorer registry & base class ----------
SCORER_REGISTRY: Dict[str, Type["BaseScorer"]] = {}


def register_scorer(name: str):
    def _inner(cls: Type["BaseScorer"]):
        SCORER_REGISTRY[name] = cls
        return cls

    return _inner


class BaseScorer:
    """Abstract scorer; implementations should override load() and score_batch()."""

    def __init__(self, spec: ModelSpec, device: Optional[str] = None, **kwargs):
        self.spec = spec
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._loaded = False

    def load(self):
        """Load model/tokenizer/resources. Idempotent."""
        self._loaded = True

    def score_batch(self, texts: List[str]) -> List[Optional[int]]:
        """Score a batch of texts. Return a list aligned with texts, each entry 0/1 or None on failure."""
        raise NotImplementedError

    def assert_loaded(self):
        if not self._loaded:
            self.load()


# ---------- concrete scorers ----------
@register_scorer("transformer")
class TransformerScorer(BaseScorer):
    """Generic single-label transformer-based classifier.
    Expects model to return logits for classes; by default we take argmax and map one or more labels to hate.
    """

    def __init__(self, spec: ModelSpec, device: Optional[str] = None, hate_label_ids: Optional[List[int]] = None, **kwargs):
        super().__init__(spec, device=device)
        self.tokenizer: Optional[PreTrainedTokenizerBase] = None
        self.model = None
        self.hate_label_ids = set(hate_label_ids or [1])  # default: label id 1 means hate
        self.batch_size = kwargs.get("batch_size", 32)

    def load(self):
        logger.info("Loading TransformerScorer for %s on %s", self.spec.model or self.spec.name, self.device)
        if not self.spec.tokenizer or not self.spec.model:
            raise ValueError("TransformerScorer requires tokenizer and model in ModelSpec")

        self.tokenizer = AutoTokenizer.from_pretrained(self.spec.tokenizer, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.spec.model)
        self.model.to(self.device)
        self.model.eval()
        self._loaded = True

    def score_batch(self, texts: List[str]) -> List[Optional[int]]:
        self.assert_loaded()
        results: List[Optional[int]] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            try:
                inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    out = self.model(**inputs)
                    logits = out.logits
                    preds = torch.argmax(logits, dim=-1).cpu().numpy().tolist()

                # map preds to binary
                for p in preds:
                    results.append(1 if int(p) in self.hate_label_ids else 0)
            except Exception as e:
                logger.exception("TransformerScorer batch failed: %s", e)
                # on failure, append None for this batch
                results.extend([None] * len(batch))

        return results


@register_scorer("multilabel_sigmoid")
class MultiLabelScorer(BaseScorer):
    """Multi-label model that uses sigmoid on logits. Determine hate by either: max-score label name or specific index threshold.
    For flexibility, caller can provide `label_names` and `hate_label_name` or `hate_index`.
    """

    def __init__(
        self,
        spec: ModelSpec,
        device: Optional[str] = None,
        label_names: Optional[List[str]] = None,
        hate_label_name: Optional[str] = None,
        hate_index: Optional[int] = None,
        threshold: float = 0.5,
        batch_size: int = 16,
        **kwargs,
    ):
        super().__init__(spec, device=device)
        self.tokenizer = None
        self.model = None
        self.label_names = label_names
        self.hate_label_name = hate_label_name
        self.hate_index = hate_index
        self.threshold = threshold
        self.batch_size = batch_size

    def load(self):
        logger.info("Loading MultiLabelScorer for %s on %s", self.spec.model or self.spec.name, self.device)
        if not self.spec.tokenizer or not self.spec.model:
            raise ValueError("MultiLabelScorer requires tokenizer and model in ModelSpec")
        self.tokenizer = AutoTokenizer.from_pretrained(self.spec.tokenizer, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.spec.model)
        self.model.to(self.device)
        self.model.eval()
        self._loaded = True

    def score_batch(self, texts: List[str]) -> List[Optional[int]]:
        self.assert_loaded()
        results: List[Optional[int]] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            try:
                inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    out = self.model(**inputs)
                    logits = out.logits
                    probs = torch.sigmoid(logits).cpu().numpy()

                for prob_vec in probs:
                    # determine hate by index or by name
                    if self.hate_index is not None:
                        is_hate = float(prob_vec[self.hate_index]) >= self.threshold
                    elif self.hate_label_name and self.label_names:
                        try:
                            idx = self.label_names.index(self.hate_label_name)
                            is_hate = float(prob_vec[idx]) >= self.threshold
                        except ValueError:
                            is_hate = False
                    else:
                        # fallback: take argmax and consider it hate only if argmax corresponds to hate_index
                        argmax = int(np.argmax(prob_vec))
                        is_hate = (argmax == self.hate_index) if self.hate_index is not None else False

                    results.append(1 if is_hate else 0)

            except Exception as e:
                logger.exception("MultiLabelScorer batch failed: %s", e)
                results.extend([None] * len(batch))

        return results


@register_scorer("pipeline")
class PipelineScorer(BaseScorer):
    """Wraps transformers.pipeline for classification-like models (e.g., kubota)."""

    def __init__(self, spec: ModelSpec, device: Optional[str] = None, hate_label_sets: Optional[List[str]] = None, batch_size: int = 16, **kwargs):
        super().__init__(spec, device=device)
        self.pipe = None
        self.hate_label_sets = set(hate_label_sets or [])
        self.batch_size = batch_size

    def load(self):
        logger.info("Loading PipelineScorer for %s", self.spec.model or self.spec.name)
        if not self.spec.model:
            raise ValueError("PipelineScorer requires model in ModelSpec")
        self.pipe = pipeline("text-classification", model=self.spec.model, device=0 if "cuda" in self.device else -1)
        self._loaded = True

    def score_batch(self, texts: List[str]) -> List[Optional[int]]:
        self.assert_loaded()
        results: List[Optional[int]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            try:
                outs = self.pipe(batch, truncation=True)
                for out in outs:
                    # out expected like {'label': '中傷性のない発言', 'score': 0.95}
                    lab = out.get("label")
                    if lab is None:
                        results.append(None)
                    else:
                        # consider hate if label text is in hate_label_sets
                        is_hate = any(h in lab for h in self.hate_label_sets)
                        results.append(1 if is_hate else 0)
            except Exception as e:
                logger.exception("PipelineScorer batch failed: %s", e)
                results.extend([None] * len(batch))
        return results


# ---------- factory ----------
def create_scorer(spec: ModelSpec, device: Optional[str] = None, **kwargs) -> BaseScorer:
    # choose scorer type via heuristics or spec.score_method
    method = (spec.score_method or "").lower()
    if method == "duo_guard" or "multilabel" in method:
        return MultiLabelScorer(spec, device=device, **kwargs)
    if method == "pipeline" or method == "kubota" or method == "kubota_score":
        return PipelineScorer(spec, device=device, **kwargs)
    # default generic transformer
    return TransformerScorer(spec, device=device, **kwargs)


# ---------- HateSpeechDetector (uses scorers) ----------
class HateSpeechDetector:
    def __init__(self, specs: List[ModelSpec], device: Optional[str] = None, batch_size: int = 32):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.specs = specs
        self.scorers: Dict[str, BaseScorer] = {}

        for spec in specs:
            # create scorer with sensible defaults
            # map some known models to scorer types and settings
            kwargs = {"batch_size": batch_size}
            if spec.name.lower() == "duoguard":
                kwargs.update({"label_names": None, "hate_label_name": "Hate", "hate_index": None, "threshold": 0.5, "batch_size": min(16, batch_size)})
            if spec.name.lower() in ("kubota", "yacis"):
                kwargs.update({"hate_label_sets": ["侮蔑", "名誉を低下"], "batch_size": min(16, batch_size)})

            scorer = create_scorer(spec, device=self.device, **kwargs)
            try:
                scorer.load()
                self.scorers[spec.name] = scorer
            except Exception as e:
                logger.exception("Failed to load scorer for %s: %s", spec.name, e)
                # skip loading but keep the entry as None to preserve ordering
                self.scorers[spec.name] = None

    def run_on_texts(self, texts: List[str]) -> Dict[str, pd.DataFrame]:
        """Run all loaded scorers on given texts and return DataFrame results keyed by model name.
        Uses batch inference per-scorer and writes results into pandas DataFrame with columns: text, HS
        """
        outputs: Dict[str, pd.DataFrame] = {}

        for name, scorer in self.scorers.items():
            logger.info("Running scorer: %s", name)
            if scorer is None:
                logger.warning("Scorer %s is not available (failed to load). Skipping.", name)
                continue

            rows: List[Dict[str, Any]] = []
            # process in batches for memory-efficiency
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]
                try:
                    preds = scorer.score_batch(batch)
                except Exception as e:
                    logger.exception("Scoring batch failed for %s: %s", name, e)
                    preds = [None] * len(batch)

                for t, p in zip(batch, preds):
                    rows.append({"text": t, "HS": int(p) if p is not None else None})

                # small logging
                logger.info("%s: processed %d/%d", name, min(i + self.batch_size, len(texts)), len(texts))

            df = pd.DataFrame(rows)
            outputs[name] = df

            # summary
            hs_series = pd.to_numeric(df["HS"], errors="coerce").fillna(0).astype(int)
            hs_count = int(hs_series.sum())
            valid = int(df["HS"].notna().sum())
            ratio = (hs_count / valid * 100) if valid > 0 else 0.0
            logger.info("[%s] Finished. Processed: %d/%d valid texts. Hate Speech: %d (%.2f%%)", name, valid, len(texts), hs_count, ratio)

        return outputs


# ---------- example default specs (mirror of previous file names) ----------
DEFAULT_SPECS = [
    ModelSpec(name="luke_offensiveness", tokenizer="studio-ousia/luke-japanese-base-lite", model="TomokiFujihara/luke-japanese-base-lite-offensiveness-estimation", score_method="transformer"),
    ModelSpec(name="DuoGuard", tokenizer="Qwen/Qwen2.5-1.5B", model="DuoGuard/DuoGuard-1.5B-transfer", score_method="multilabel"),
    ModelSpec(name="kubota", tokenizer=None, model="kubota/luke-large-defamation-detection-japanese", score_method="pipeline"),
    ModelSpec(name="yuki", tokenizer="yukismd/HateSpeechClassification-japanese-gpt-neox-3-6b-instruction-ppo", model="yukismd/HateSpeechClassification-japanese-gpt-neox-3-6b-instruction-ppo", score_method="transformer"),
    ModelSpec(name="kit", tokenizer="kit-nlp/electra-small-japanese-discriminator-cyberbullying", model="kit-nlp/electra-small-japanese-discriminator-cyberbullying", score_method="transformer"),
]


# If run as script, provide a tiny smoke-test (no heavy models will be downloaded unless user runs it)
if __name__ == "__main__":
    logger.info("Running smoke example for hate_detector_refactored")
    sample_texts = [
        "これはテストです。",
        "お前は死ね！",
        "あの人は犯罪者だ。",
    ]

    # initialize detector but avoid loading heavy models by default; use empty spec to show flow
    specs = [ModelSpec(name="dummy")]
    detector = HateSpeechDetector(specs, device="cpu", batch_size=2)
    # The dummy scorer won't load any model; run_on_texts will skip it
    results = detector.run_on_texts(sample_texts)
    print(results)
