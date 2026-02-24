#!/usr/bin/env python3
"""
Evaluate classification models under data_detect for Chinese/Japanese.

Usage examples:
  python eval_pipeline.py --lang japanese --csv path/to/test.csv --max 1000
  python eval_pipeline.py --lang chinese --csv path/to/ch_test.csv --text-col content

This script dynamically loads model modules from data_detect/<Lang>/models,
instantiates model classes that expose `score(text)` and computes metrics.
"""
import argparse
import importlib.util
import inspect
import os
from pathlib import Path
import sys
from typing import List

import pandas as pd
from tqdm import tqdm
import logging
import datetime

try:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
except Exception:
    accuracy_score = None


DATA_ROOT = Path(__file__).resolve().parents[2] / "data_detect"

# fixed output/log directories under model_eval/classifier
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
LOG_DIR = Path(__file__).resolve().parent / "logs"


def setup_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    if not root.handlers:
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        root.addHandler(sh)
        fh = logging.FileHandler(LOG_DIR / f"eval_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log", encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
    root.setLevel(logging.INFO)
    return root


def find_model_classes(module):
    classes = []
    for name, obj in inspect.getmembers(module, inspect.isclass):
        # skip builtins and imports
        if obj.__module__ != module.__name__:
            continue
        # require a score method
        if hasattr(obj, "score"):
            classes.append(obj)
    return classes


def load_module_from_path(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def autodetect_text_column(df: pd.DataFrame) -> str:
    candidates = ["text", "content", "post", "sentence", "comment", "body"]
    for c in candidates:
        if c in df.columns:
            return c
    # fallback to first non-target column
    for c in df.columns:
        if c != "hate_speech":
            return c
    raise ValueError("Could not find a text column in CSV")


def normalize_label(v: str) -> int:
    if pd.isna(v):
        return 0
    s = str(v).strip()
    if s in ("是", "1", "yes", "True", "true"):
        return 1
    if s in ("否", "0", "no", "False", "false"):
        return 0
    # fallback: attempt numeric
    try:
        return int(float(s))
    except Exception:
        return 0


def evaluate_models(lang: str, csv_path: str, max_samples: int = None, text_col: str = None, device: str = "cpu"):
    lang = lang.lower()
    logger = setup_logging()
    model_dir = DATA_ROOT / ("Chinese" if lang.startswith("ch") else "Japanese") / "models"
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    df = pd.read_csv(csv_path)
    if text_col is None:
        text_col = autodetect_text_column(df)

    if "hate_speech" not in df.columns:
        raise ValueError("CSV must contain a 'hate_speech' column with values 是/否")

    if max_samples:
        df = df.sample(frac=1, random_state=42).reset_index(drop=True).head(max_samples)

    y_true = [normalize_label(v) for v in df["hate_speech"].tolist()]
    texts = df[text_col].astype(str).tolist()

    results = []

    # iterate over python files in model_dir
    for py in sorted(model_dir.glob("*.py")):
        # skip __init__.py
        if py.name.startswith("__"):
            continue
        try:
            module = load_module_from_path(py)
        except Exception as e:
            logger.exception(f"Failed to load {py.name}: {e}")
            continue

        classes = find_model_classes(module)
        if not classes:
            logger.warning(f"No model classes found in {py.name}")
            continue

        for cls in classes:
            model_name = f"{py.stem}.{cls.__name__}"
            logger.info(f"Instantiating {model_name}...")
            try:
                # prefer constructor with device arg when available
                sig = inspect.signature(cls)
                if "device" in sig.parameters:
                    inst = cls(device=device)
                else:
                    inst = cls()
            except Exception as e:
                logger.exception(f"Failed to instantiate {model_name}: {e}")
                continue

            y_pred = []
            probs = []
            for t in tqdm(texts, desc=f"Scoring {model_name}"):
                try:
                    out = inst.score(t)
                    if isinstance(out, dict) and "label" in out:
                        y_pred.append(int(out.get("label", 0)))
                        probs.append(float(out.get("prob", 0.0)))
                    elif isinstance(out, (list, tuple)):
                        # permissive: label first
                        y_pred.append(int(out[0]))
                        probs.append(float(out[1]) if len(out) > 1 else 0.0)
                    else:
                        y_pred.append(int(out))
                        probs.append(0.0)
                except Exception as e:
                    logger.exception(f"Error scoring example: {e}")
                    y_pred.append(0)
                    probs.append(0.0)

            # metrics
            acc = None
            prec = rec = f1 = None
            if accuracy_score is not None:
                try:
                    acc = accuracy_score(y_true, y_pred)
                    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
                except Exception:
                    acc = None

            cm = None
            try:
                cm = confusion_matrix(y_true, y_pred).tolist()
            except Exception:
                cm = None

            results.append({
                "model": model_name,
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "confusion_matrix": cm,
            })

    out_df = pd.DataFrame(results)
    # save results to fixed output directory
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"eval_results_{lang}_{ts}.csv"
    try:
        out_df.to_csv(out_path, index=False, encoding="utf-8")
        logger.info(f"Saved evaluation results to {out_path}")
        logger.info("Summary:\n" + out_df.to_string())
    except Exception:
        logger.exception("Failed to save evaluation results")

    return out_df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lang", required=True, choices=["chinese", "japanese", "china", "ja", "zh"], help="Language models to evaluate")
    p.add_argument("--csv", required=True, help="Path to test CSV with 'hate_speech' column")
    p.add_argument("--max", type=int, default=None, help="Max number of test samples to use")
    p.add_argument("--text-col", default=None, help="Text column name in CSV (auto-detected if omitted)")
    p.add_argument("--device", default="cpu", help="Device to pass to model constructors")
    args = p.parse_args()

    evaluate_models(args.lang, args.csv, max_samples=args.max, text_col=args.text_col, device=args.device)


if __name__ == "__main__":
    main()
