# pipeline.py
import os
import math
import random
import logging
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np
from data_detect.Japanese.config import *
from data_detect.utils import compute_sample_size, random_sample_indices, ensure_dir, save_annotation_csv
from data_detect.Japanese.constants import ModelInfo, ModelName, HateScore
from data_detect.Japanese.factory import ModelFactory
from tqdm import tqdm

logger = logging.getLogger("ensemble_pipeline")

def _binary_entropy(p: float) -> float:
    # handle edge cases
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


class HatePipeline:
    """Pipeline that accepts a list of `ModelWrapper` instances and performs ensemble scoring
    and stratified sampling for annotation export.
    If `models` is None, we will attempt to create adapters for the models listed in `ModelName`.
    """
    def __init__(self, logger, input_csv: str, models: Optional[List[ModelInfo]] = None,
                 population: int = DEFAULT_POPULATION, margin: float = DEFAULT_MARGIN,
                 sample_override: int = None, output_dir: str = OUTPUT_DIR, device: str = "cpu"):
        self.input_csv = input_csv
        self.population = population
        self.margin = margin
        self.logger = logger
        self.sample_override = sample_override
        self.output_dir = output_dir
        ensure_dir(self.output_dir)
        self.device = device

        # models: list of ModelWrapper
        if models is None:
            self.logger.info("No ModelWrapper list provided; creating adapters from ModelName enum.")
            self._models_initalize()
        else:
            self._models_initalize(models=models)

    def _models_initalize(self, models=None):
        if models is None:
            models = ModelName
        self.models = []
        for m in models:
            try:
                adapter = ModelFactory.create_model(self.logger, m, device=self.device)
                self.models.append((m.name, adapter))
            except Exception as e:
                self.logger.warning(f"Failed to init adapter for {m.name}: {e}")

        self.logger.info(f"Successfully initialized {self.models} models")
        return self

    def _load_texts(self) -> pd.Series:
        df = pd.read_csv(self.input_csv)
        if "text" not in df.columns:
            raise ValueError("input csv must have a 'text' column")
        return df["text"].astype(str)

    def evaluate_ensemble(self, texts: List[str]) -> pd.DataFrame:
        """Run all models over texts and compute per-sample ensemble statistics.
        Returns DataFrame with columns:
        - text, model_votes (list), model_probs (list), votes_str (e.g., '2/1'),
          avg_prob, entropy, conflict(bool)
        """
        n_models = len(self.models)
        if n_models == 0:
            raise RuntimeError("No models available for ensemble evaluation")

        # collect predictions per model
        model_preds = []  # list of lists (per model)
        names = []
        for name, m in self.models:
            self.logger.info(f"Predicting with {name} on {len(texts)} texts")
            preds = m.predict(list(texts))
            # normalize into dicts
            model_preds.append(preds)
            names.append(name)

        rows = []
        for i, t in enumerate(texts):
            probs = []
            labels = []
            for preds in model_preds:
                p = preds[i].get("prob", 0.0)
                lbl = int(preds[i].get("label", 0))
                probs.append(float(p))
                labels.append(int(lbl))

            avg_prob = float(np.mean(probs)) if len(probs) > 0 else 0.0
            # votes: count of label==1
            vote_for = sum(labels)
            vote_against = n_models - vote_for
            votes_str = f"{vote_for}/{vote_against}"
            unanimous = (vote_for == 0) or (vote_for == n_models)
            conflict = not unanimous

            # compute entropy from avg_prob (binary)
            ent = _binary_entropy(avg_prob)

            rows.append({
                "text": t,
                "model_names": names,
                "model_labels": labels,
                "model_probs": probs,
                "model_votes": votes_str,
                "avg_prob": avg_prob,
                "entropy": ent,
                "conflict": bool(conflict),
                "vote_for": vote_for,
                "vote_against": vote_against,
            })

        return pd.DataFrame(rows)

    def generate_annotation_set(self, evaluated_df: pd.DataFrame, total_n: int = 4000,
                                keywords: Optional[List[str]] = None) -> pd.DataFrame:
        """Stratified sampling per spec:
        - 30% consistent hate (all models vote hate) and high prob
        - 30% consistent non-hate AND contains keywords
        - 40% conflicts or uncertain (entropy high or avg_prob in [0.4,0.6])
        Returns sampled DataFrame with sampling_strategy_tag column.
        """
        if keywords is None:
            keywords = []

        n1 = int(total_n * 0.3)
        n2 = int(total_n * 0.3)
        n3 = total_n - n1 - n2

        # consistent hate: vote_for == n_models and avg_prob high
        n_models = len(self.models)
        consistent_hate = evaluated_df[(evaluated_df["vote_for"] == n_models) & (evaluated_df["avg_prob"] >= 0.8)].copy()

        # consistent non-hate with keywords
        if len(keywords) > 0:
            contains_kw = evaluated_df["text"].apply(lambda s: any(kw in s for kw in keywords))
            consistent_non = evaluated_df[(evaluated_df["vote_for"] == 0) & (contains_kw)].copy()
        else:
            # if no keywords provided, fall back to consistent non-hate overall
            consistent_non = evaluated_df[evaluated_df["vote_for"] == 0].copy()

        # conflicts or uncertain: conflict True OR avg_prob between 0.4 and 0.6
        uncertain = evaluated_df[(evaluated_df["conflict"] == True) | ((evaluated_df["avg_prob"] >= 0.4) & (evaluated_df["avg_prob"] <= 0.6))].copy()

        sampled = []

        def _sample_from(df: pd.DataFrame, k: int, tag: str):
            if df.empty or k <= 0:
                return pd.DataFrame()
            k = min(k, len(df))
            return df.sample(n=k, random_state=42).assign(sampling_strategy_tag=tag)

        s1 = _sample_from(consistent_hate, n1, "consistent_hate")
        s2 = _sample_from(consistent_non, n2, "consistent_non_hate")
        s3 = _sample_from(uncertain, n3, "conflict_or_uncertain")

        sampled_df = pd.concat([s1, s2, s3], ignore_index=True)

        # if not enough samples due to small groups, fill from remaining pool
        if len(sampled_df) < total_n:
            remaining_pool = evaluated_df.drop(sampled_df.index, errors="ignore")
            need = total_n - len(sampled_df)
            if not remaining_pool.empty and need > 0:
                fill = remaining_pool.sample(n=min(need, len(remaining_pool)), random_state=42).assign(sampling_strategy_tag="fill")
                sampled_df = pd.concat([sampled_df, fill], ignore_index=True)

        # ensure columns requested by user
        final = sampled_df[["text", "model_votes", "avg_prob", "sampling_strategy_tag"]].copy()
        final = final.rename(columns={"avg_prob": "average_prob"})

        return final

    def export_for_annotation(self, df: pd.DataFrame, out_path: str, fmt: str = "csv") -> str:
        """Export DataFrame to CSV or JSONL. Returns path on success."""
        ensure_dir(os.path.dirname(out_path))
        if fmt == "csv":
            df.to_csv(out_path, index=False)
        else:
            # jsonl
            df.to_json(out_path, orient="records", lines=True, force_ascii=False)
        return out_path

    def run_detection(self, total_annotation_n: int = 4000, keywords: Optional[List[str]] = None):
        texts = self._load_texts()
        total = len(texts)
        self.logger.info(f"Loaded {total} texts from {self.input_csv}")

        computed_n = compute_sample_size(self.population, margin=self.margin)
        self.logger.info(f"Computed sample size (finite-pop correction) = {computed_n}")
        sample_n = self.sample_override or min(computed_n, DEFAULT_MAX_SAMPLE)
        self.logger.info(f"Using sample size = {sample_n}")

        idxs = random_sample_indices(total, sample_n)
        sampled_texts = texts.iloc[idxs].reset_index(drop=True)

        self.logger.info("Evaluating ensemble on sampled texts...")
        evaluated = self.evaluate_ensemble(sampled_texts.tolist())

        self.logger.info("Generating annotation set via stratified sampling...")
        annotation_df = self.generate_annotation_set(evaluated, total_n=total_annotation_n, keywords=keywords)

        out_csv = os.path.join(self.output_dir, f"to_annotate_ensemble_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv")
        try:
            self.export_for_annotation(annotation_df, out_csv, fmt="csv")
            self.logger.info(f"Saved annotation CSV to {out_csv}")
        except Exception as e:
            self.logger.error(f"Failed to save annotation CSV: {e}")
            out_csv = None

        return {
            "evaluated": evaluated,
            "annotation_df": annotation_df,
            "annotation_csv": out_csv,
            "n_models": len(self.models)
        }

    # 以下为微调的 scaffold（与之前保留）
    def finetune_model(self, model_name_or_path: str, train_csv: str, val_csv: str, output_dir: str, epochs:int=3):
        """
        示例：使用 Hugging Face Trainer 对文本二分类做微调
        train_csv/val_csv 都应包含 columns: text,label (label 0/1)
        这里给出可直接运行的 template
        """
        from datasets import load_dataset, Dataset
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
        import numpy as np
        import torch

        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)

        ds_train = Dataset.from_pandas(train_df[["text","label"]])
        ds_val = Dataset.from_pandas(val_df[["text","label"]])

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        def preprocess(batch):
            return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)
        ds_train = ds_train.map(preprocess, batched=True)
        ds_val = ds_val.map(preprocess, batched=True)
        ds_train.set_format(type="torch", columns=["input_ids","attention_mask","label"])
        ds_val.set_format(type="torch", columns=["input_ids","attention_mask","label"])

        model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, num_labels=2)

        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            num_train_epochs=epochs,
            save_total_limit=2,
            logging_steps=50
        )

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            from sklearn.metrics import accuracy_score, precision_recall_fscore_support
            acc = accuracy_score(labels, preds)
            p, r, f, _ = precision_recall_fscore_support(labels, preds, average="binary")
            return {"accuracy": acc, "precision": p, "recall": r, "f1": f}

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=ds_train,
            eval_dataset=ds_val,
            compute_metrics=compute_metrics
        )

        trainer.train()
        trainer.save_model(output_dir)
        return output_dir

    def evaluate_against_gemini(self, model_wrapper, sample_texts: List[str], gemini_results: List[Dict]):
        """
        给定微调后或原模型的 wrapper（须实现 score_text + is_hate），计算与 gemini 的一致率。
        """
        hs_preds = []
        for t in sample_texts:
            score = model_wrapper.score_text(t)
            hs = model_wrapper.is_hate(score)
            hs_preds.append(hs)
        gemini_preds = [1 if (g.get("is_hate") is True) else 0 for g in gemini_results]
        total = len(hs_preds)
        matches = sum(1 for a,b in zip(hs_preds, gemini_preds) if a==b)
        acc = matches/total if total>0 else 0.0
        return {"matches": matches, "total": total, "accuracy": acc}
