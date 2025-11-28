# pipeline.py
import os, random
import pandas as pd
import time
from config import *
from utils import compute_sample_size, random_sample_indices, ensure_dir, save_annotation_csv
from hate_detector import HateSpeechDetector, ModelName
from gemini_client import GeminiClient
from typing import List, Dict
from tqdm import tqdm
class HatePipeline:
    def __init__(self, logger, input_csv: str, gemini_api_key: str,
                 population: int = DEFAULT_POPULATION, margin: float = DEFAULT_MARGIN,
                 sample_override: int = None, output_dir: str = OUTPUT_DIR, device="cpu"):
        self.input_csv = input_csv
        self.gemini_api_key = gemini_api_key
        self.population = population
        self.margin = margin
        self.logger = logger
        self.models = [m.value for m in ModelName]
        self.sample_override = sample_override
        self.output_dir = output_dir
        ensure_dir(self.output_dir)
        self.device = device

    def _load_texts(self) -> pd.Series:
        df = pd.read_csv(self.input_csv)
        if "text" not in df.columns:
            raise ValueError("input csv must have a 'text' column")
        return df["text"].astype(str)

    def run_detection(self):
        texts = self._load_texts()
        total = len(texts)
        self.logger.info(f"Loaded {total} texts from {self.input_csv}")

        # 计算样本量
        computed_n = compute_sample_size(self.population, margin=self.margin)
        self.logger.info(f"Computed sample size (finite-pop correction) = {computed_n}")
        sample_n = self.sample_override or min(computed_n, DEFAULT_MAX_SAMPLE)
        self.logger.info(f"Using sample size = {sample_n}")

        # ───────── 1. 先抽样 ─────────
        idxs = random_sample_indices(total, sample_n)
        sampled_texts = texts.iloc[idxs].reset_index(drop=True)

        # ───────── 2. 再对抽样文本做 HS 检测 ─────────
        detector = HateSpeechDetector(logger=self.logger,model_specs=self.models,device=self.device)

        self.logger.info("Running HS detection on sampled texts...")
        sample_results = detector.run_on_texts(
            sampled_texts
        )

        # ───────── 3. Gemini 判定 ─────────
        gemini = GeminiClient(self.logger,self.gemini_api_key, model_name=GEMINI_MODEL)
        gemini_results = []

        self.logger.info("Running Gemini classification...")
        for t in tqdm(sampled_texts, desc="Gemini", ncols=120, smoothing=0.1, colour="green"):
            try:
                g = gemini.classify(t)
                jitter = random.uniform(0, 1)
                time.sleep(jitter+1) # 每次分类休息一秒，防止被Google拦截
                gemini_results.append(g)
            except Exception as e:
                self.logger.error(f"Error:{e} when generating results of Gemini")

        # ───────── 4. 统计一致性 ─────────
        self.logger.info("Computing model–Gemini agreement...")

        stats = {}

        for name, texts in tqdm(sample_results.items(), desc="Stats", ncols=100):
            try:
                hs_labels = texts.get("HS", pd.Series()).fillna(0).astype(int).tolist()
                
                gem_labels = []
                for i, g in enumerate(gemini_results):
                    try:
                        gem_labels.append(1 if g.get("is_hate") else 0)
                    except Exception as sub_e:
                        gem_labels.append(0)  # 出错则用默认值
                        self.logger.warning(f"Error processing gemini_results[{i}]: {sub_e}")
                
                # 对齐长度，防止报错
                min_len = min(len(hs_labels), len(gem_labels))
                hs_labels = hs_labels[:min_len]
                gem_labels = gem_labels[:min_len]

                total = len(hs_labels)
                matches = sum(1 for a, b in zip(hs_labels, gem_labels) if a == b)
                accuracy = matches / total if total > 0 else 0.0

                stats[name] = {"matches": matches, "total": total, "accuracy": accuracy}
            
            except Exception as e:
                self.logger.error(f"Error when calculating agreement for '{name}': {e}")
                stats[name] = {"matches": 0, "total": 0, "accuracy": 0.0}

        # ───────── 5. 选择最佳模型 ─────────
        try:
            best_model = max(stats.items(), key=lambda x: x[1].get("accuracy", 0.0))
            best_name = best_model[0]
            best_stats = best_model[1]
        except Exception as e:
            self.logger.error(f"Failed to select best model: {e}")
            best_name = None
            best_stats = {"accuracy": 0.0}

        self.logger.info(f"Model accuracies vs Gemini: { {k:v.get('accuracy', 0.0) for k,v in stats.items()} }")
        if best_name:
            self.logger.info(f"Selected best model: {best_name} with accuracy {best_stats.get('accuracy',0.0):.4f}")
        else:
            self.logger.warning("No valid best model could be selected.")

        # ───────── 6. 生成待标注 CSV ─────────
        rows = []

        if best_name and best_name in sample_results:
            df_best = sample_results[best_name]
            
            for idx, line in df_best.iterrows():
                try:
                    text = line.get('text', "")
                    hs_label = int(line.get("HS", -1)) if line.get("HS") is not None else None
                    gem_label = 1 if line.get("is_hate") is True else 0
                    rows.append({
                        "text": text,
                        "hs_model_label": hs_label,
                        "gemini_label": gem_label,
                        "human_label": ""
                    })
                except Exception as e:
                    self.logger.warning(f"Skipping line {idx} due to error: {e}")
        else:
            self.logger.warning(f"No data found for best model '{best_name}'")

        # 保存 CSV
        out_csv = os.path.join(self.output_dir, f"to_annotate_{best_name or 'unknown'}.csv")
        try:
            save_annotation_csv(rows, out_csv)
            self.logger.info(f"Saved annotation CSV to {out_csv}")
        except Exception as e:
            self.logger.error(f"Failed to save annotation CSV: {e}")
            out_csv = None

        return {
            "stats": stats,
            "best_model": best_name,
            "best_stats": best_stats,
            "annotation_csv": out_csv
        }

    
    # 以下为微调的 scaffold（需要真实标注数据）
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
