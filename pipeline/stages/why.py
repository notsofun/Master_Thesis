"""
pipeline/stages/why.py — WHY Stage Wrapper (RQ3 Moral Motivation Projection)
===========================================================================
Reuses unsupervised_classification/RQ3/main.py.
Calls via subprocess, avoiding module-level path initialization conflicts.

collect() reads rq3_bias_matrix.csv / rq3_anova_results.csv / rq3_summary.csv,
converts to unified schema.
"""

import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RQ3_SCRIPT   = PROJECT_ROOT / "unsupervised_classification" / "RQ3" / "main.py"
RQ3_DATA_DIR = PROJECT_ROOT / "unsupervised_classification" / "RQ3" / "data"

# RQ3 output key files
BIAS_CSV  = RQ3_DATA_DIR / "rq3_bias_matrix.csv"
ANOVA_CSV = RQ3_DATA_DIR / "rq3_anova_results.csv"
SUMMARY_CSV = RQ3_DATA_DIR / "rq3_summary.csv"

# 7 moral axis names (consistent with RQ3/main.py)
MORAL_AXES = ["Harm", "Fairness", "Loyalty", "Authority", "Sanctity", "RealThreat", "SymThreat"]


def run(config: dict) -> bool:
    """Call RQ3 script to complete moral motivation projection.
    Return True = success, False = failure.

    Config fields (from config.yaml why section):
      max_docs   : int  -- debug: process only first N documents
      batch_size : int  -- E5 encoding batch size
      from_bias  : bool -- skip axis building+encoding, start from existing bias_matrix
    """
    why_cfg = config.get("why", {})

    cmd = [sys.executable, str(RQ3_SCRIPT)]

    max_docs = why_cfg.get("max_docs")
    if max_docs:
        cmd += ["--max-docs", str(max_docs)]

    batch_size = why_cfg.get("batch_size", 64)
    cmd += ["--batch-size", str(batch_size)]

    if why_cfg.get("from_bias", False):
        cmd.append("--from-bias")

    logger.info(f"[WHY] Starting RQ3 script: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))

    if result.returncode != 0:
        logger.error(f"[WHY] RQ3 script exited with code {result.returncode}, check logs above")
        return False

    logger.info("[WHY] ✅ RQ3 completed")
    return True


def collect(config: dict) -> tuple[list[dict], list[dict]]:
    """Read RQ3 results, return (per-document moral bias record list, statistical summary list).

    Per-document records: MoralBiasRecord.to_dict() format
      topic / lang / text / bias: {axis → float}

    Statistical summary: rq3_summary.csv converted to dict list (for paper)
    """
    records: list[dict] = []
    summary: list[dict] = []

    # ── per-document 道德轴偏移 ────────────────────────────────────────────
    if not BIAS_CSV.exists():
        logger.warning(f"[WHY] 未找到 {BIAS_CSV.name}，跳过 collect")
        return records, summary

    logger.info(f"[WHY] 读取 {BIAS_CSV.name}")
    df = pd.read_csv(BIAS_CSV)

    # bias_matrix 格式：text / lang / topic / Harm / Fairness / ... 各轴列
    axis_cols = [c for c in df.columns if c in MORAL_AXES]

    for _, row in df.iterrows():
        bias = {ax: round(float(row[ax]), 4) for ax in axis_cols if ax in row}
        rec = {
            "topic": int(row.get("topic", -1)),
            "lang":  str(row.get("lang", "")),
            "text":  str(row.get("text", ""))[:300],  # 截断避免输出过大
            "bias":  bias,
        }
        records.append(rec)

    logger.info(f"[WHY] ✅ 收集 {len(records)} 条道德轴偏移记录")

    # ── ANOVA 统计摘要 ─────────────────────────────────────────────────────
    if SUMMARY_CSV.exists():
        logger.info(f"[WHY] 读取 {SUMMARY_CSV.name}")
        summary_df = pd.read_csv(SUMMARY_CSV)
        # 补充 ANOVA p-values（如果有）
        if ANOVA_CSV.exists():
            anova_df = pd.read_csv(ANOVA_CSV)
            summary = summary_df.to_dict(orient="records")
            # 把 ANOVA 结果附加为独立条目
            summary += [{"_type": "anova", **r} for r in anova_df.to_dict(orient="records")]
        else:
            summary = summary_df.to_dict(orient="records")
        logger.info(f"[WHY] 摘要: {len(summary)} 行")
    elif ANOVA_CSV.exists():
        logger.info(f"[WHY] 读取 {ANOVA_CSV.name}")
        summary = pd.read_csv(ANOVA_CSV).to_dict(orient="records")

    return records, summary
