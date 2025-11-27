# utils.py
import math
import random
import pandas as pd
from typing import List, Tuple
import os
from config import RANDOM_SEED

random.seed(RANDOM_SEED)

def compute_sample_size(population: int, margin: float=0.03, z: float=1.96, p: float=0.5) -> int:
    """
    计算样本量（finite population correction）。
    n0 = z^2 * p(1-p) / e^2
    n = n0 / (1 + (n0 - 1)/N)
    """
    e = margin
    n0 = (z**2) * p * (1 - p) / (e**2)
    n = n0 / (1 + (n0 - 1) / population)
    return math.ceil(n)

def random_sample_indices(total: int, k: int, seed:int=None) -> List[int]:
    rnd = random.Random(seed if seed is not None else RANDOM_SEED)
    return rnd.sample(range(total), k)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_annotation_csv(rows, output_path):
    """
    rows: list of dict with keys: text, hs_label, gemini_label, human_label
    """
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return df
