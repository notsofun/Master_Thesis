# pipeline.py
import os
import math
import logging
from typing import List, Dict, Optional, Any
from data_detect.Chinese.config import DEFAULT_MAX_SAMPLE as ChineseMAX
from data_detect.Japanese.config import DEFAULT_MAX_SAMPLE as JapaneseMAX
import pandas as pd
import numpy as np
from data_detect.utils import compute_sample_size, random_sample_indices, ensure_dir, Language
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
    Supports both Japanese and Chinese models.
    If `models` is None, we will attempt to create adapters for the models listed in `ModelName`.
    """
    def __init__(self, logger, input_csv: str, models: Optional[List[Any]] = None,
                 population: int = 100000, margin: float = 0.05,
                 sample_override: int = None, output_dir: str = None, device: str = "cpu", 
                 model_factory=None):
        self.input_csv = input_csv
        self.population = population
        self.margin = margin
        self.logger = logger
        self.sample_override = sample_override
        self.output_dir = output_dir or os.path.join(os.path.dirname(__file__), "output")
        ensure_dir(self.output_dir)
        self.device = device
        self.lan = Language.Japanese
        
        # 通用的 factory (支持日文和中文)
        self.model_factory = model_factory
        
        # models: list of model enums
        if models is None:
            self.logger.info("No models provided; attempting to create adapters from Japanese ModelName enum.")
            try:
                from data_detect.Japanese.constants import ModelName
                from data_detect.Japanese.factory import ModelFactory as JapaneseFactory
                models = ModelName
                self.model_factory = JapaneseFactory
            except ImportError:
                raise RuntimeError("Default Japanese models not available")
        else:
            # 从第一个 model 推断 factory
            if self.model_factory is None:
                self._infer_factory_from_models(models)
        
        self._models_initalize(models=models)

    def _infer_factory_from_models(self, models):
        """根据 model 类型推断要使用的 factory"""
        if not models:
            return
        
        first_model = models[0]
        model_class_name = first_model.__class__.__name__
        
        if "Chinese" in model_class_name or "ChineseModelName" in str(type(first_model)):
            from data_detect.Chinese.factory import ChineseModelFactory
            self.model_factory = ChineseModelFactory
            self.lan = Language.CHINESE
        else:
            from data_detect.Japanese.factory import ModelFactory as JapaneseFactory
            self.model_factory = JapaneseFactory
            self.lan = Language.Japanese

    def _models_initalize(self, models=None):
        self.models = []
        for m in models:
            try:
                adapter = self.model_factory.create_model(self.logger, m, device=self.device)
                # 获取模型名称
                model_name = m.value.name if hasattr(m, 'value') else m.name
                self.models.append((model_name, adapter))
            except Exception as e:
                self.logger.warning(f"Failed to init adapter for {m}: {e}")

        self.logger.info(f"Successfully initialized {len(self.models)} models")
        return self

    def _load_texts(self) -> pd.Series:
        df = pd.read_csv(self.input_csv)
        # Try 'text' first, fallback to 'main_content' for Chinese datasets
        if self.lan == Language.Japanese:
            return df["text"].astype(str)
        elif self.lan == Language.CHINESE:
            self.logger.info("Using 'main_content' column instead of 'text'")
            return df["text"].astype(str)
        else:
            raise ValueError("input csv must have a 'text' or 'main_content' column")

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
        - Primary: conflict_or_uncertain samples:
          1. 多模型判断不一致 (conflict == True)
          2. 多模型一致但置信度低 (unanimous && avg_prob < 0.5)
          3. 平均置信度在不确定范围内 (avg_prob in [0.4, 0.6])
        - If insufficient, fill remaining equally from consistent_hate and consistent_non_hate
        Returns sampled DataFrame with sampling_strategy_tag column.
        """
        if keywords is None:
            keywords = []

        n_models = len(self.models)
        
        # 1. 高置信度的确定样本
        consistent_hate = evaluated_df[(evaluated_df["vote_for"] == n_models) & (evaluated_df["avg_prob"] >= 0.8)].copy()
        
        # 2. 一致但低置信度的样本 (多个模型都同意但都不确定)
        low_confidence_unanimous = evaluated_df[
            ((evaluated_df["vote_for"] == n_models) | (evaluated_df["vote_for"] == 0)) & 
            (evaluated_df["avg_prob"] < 0.5)
        ].copy()
        
        # 3. 非仇恨一致样本 (with or without keywords)
        if len(keywords) > 0:
            contains_kw = evaluated_df["text"].apply(lambda s: any(kw in s for kw in keywords))
            consistent_non = evaluated_df[(evaluated_df["vote_for"] == 0) & (contains_kw)].copy()
        else:
            consistent_non = evaluated_df[evaluated_df["vote_for"] == 0].copy()
        
        # 4. 不确定/冲突样本 (模型判断不一致 OR 置信度在不确定范围)
        uncertain = evaluated_df[
            (evaluated_df["conflict"] == True) | 
            ((evaluated_df["avg_prob"] >= 0.4) & (evaluated_df["avg_prob"] <= 0.6))
        ].copy()
        
        # 从低置信度中移除已经在 uncertain 中的
        low_confidence_unanimous = low_confidence_unanimous[
            ~low_confidence_unanimous.index.isin(uncertain.index)
        ]
        
        # 合并为主要的采样池：不确定 + 低置信度一致
        # 修改这一行，增加 subset 参数
        primary_pool = pd.concat(
            [uncertain, low_confidence_unanimous], 
            ignore_index=False
        ).drop_duplicates(subset=['text']) # 或者使用 subset=[evaluated_df.index.name] 如果你有唯一索引

        def _sample_from(df: pd.DataFrame, k: int, tag: str):
            if df.empty or k <= 0:
                return pd.DataFrame()
            k = min(k, len(df))
            return df.sample(n=k, random_state=42).assign(sampling_strategy_tag=tag)

        # 主要采样：从不确定/冲突 + 低置信度样本中采样
        s_primary = _sample_from(primary_pool, total_n, "uncertain_or_low_confidence")
        
        sampled_df = s_primary.copy()

        # 如果不足，从高置信度样本中补充（一比一）
        if len(sampled_df) < total_n:
            need = total_n - len(sampled_df)
            need_from_hate = (need + 1) // 2  # 天花板除法
            need_from_non_hate = need // 2     # 地板除法
            
            s_hate = _sample_from(consistent_hate, need_from_hate, "consistent_hate_high_conf")
            s_non = _sample_from(consistent_non, need_from_non_hate, "consistent_non_hate")
            
            sampled_df = pd.concat([sampled_df, s_hate, s_non], ignore_index=True)

        # 确保输出列
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
        max_samp = ChineseMAX if self.lan == Language.CHINESE else JapaneseMAX
        sample_n = self.sample_override or max(computed_n, ChineseMAX)
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