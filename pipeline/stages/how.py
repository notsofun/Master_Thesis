"""
pipeline/stages/how.py — HOW Stage Wrapper (RQ2 Rhetorical Strategy Extraction)
==============================================================================
Reuses unsupervised_classification/RQ2/rq2_pipeline_v2.py.
Calls via subprocess, avoiding module-level path initialization conflicts.

collect() reads rq2_framing_labeled.csv and rq2_aggregated_summary.csv,
converts to unified schema.
"""

import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RQ2_SCRIPT   = PROJECT_ROOT / "unsupervised_classification" / "RQ2" / "rq2_pipeline_v2.py"
RQ2_DATA_DIR = PROJECT_ROOT / "unsupervised_classification" / "RQ2" / "data"

# RQ2 output key files
LABELED_CSV  = RQ2_DATA_DIR / "rq2_framing_labeled.csv"
SUMMARY_CSV  = RQ2_DATA_DIR / "rq2_aggregated_summary.csv"


def run(config: dict) -> bool:
    """Call RQ2 script to complete rhetorical strategy extraction and frame classification.
    Return True = success, False = failure.

    Config fields (from config.yaml how section):
      no_gemini : bool  -- skip Gemini, use rule classifier only
      max_rows  : int   -- debug: process only first N rows of documents
    """
    how_cfg = config.get("how", {})

    cmd = [sys.executable, str(RQ2_SCRIPT)]

    if how_cfg.get("no_gemini", False):
        cmd.append("--no-gemini")

    max_rows = how_cfg.get("max_rows")
    if max_rows:
        cmd += ["--max-rows", str(max_rows)]

    logger.info(f"[HOW] Starting RQ2 script: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))

    if result.returncode != 0:
        logger.error(f"[HOW] RQ2 script exited with code {result.returncode}, check logs above")
        return False

    logger.info("[HOW] ✅ RQ2 completed")
    return True


def collect(config: dict) -> tuple[list[dict], list[dict]]:
    """Read RQ2 results, return (per-document record list, aggregated summary list).

    Per-document records: ExpressionRecord.to_dict() format
      topic / lang / text / predicate / context / target / frame_type / layer

    Aggregated summary: rq2_aggregated_summary.csv converted to dict list (for paper appendix)
    """
    records: list[dict] = []
    summary: list[dict] = []

    # ── per-document 结果 ──────────────────────────────────────────────────
    if not LABELED_CSV.exists():
        logger.warning(f"[HOW] 未找到 {LABELED_CSV.name}，跳过 collect")
        return records, summary

    logger.info(f"[HOW] 读取 {LABELED_CSV.name}")
    df = pd.read_csv(LABELED_CSV)

    # 过滤噪声（[noise] 标签）
    df = df[df.get("frame_type", pd.Series(dtype=str)) != "[noise]"]

    for _, row in df.iterrows():
        rec = {
            "topic":      int(row.get("topic", -1)),
            "lang":       str(row.get("lang", "")),
            "text":       str(row.get("text", "")),
            "predicate":  str(row.get("predicate", "")),
            "context":    str(row.get("context", "")),
            "target":     str(row.get("target", "")),
            "frame_type": str(row.get("frame_type", "other")),
            "layer":      str(row.get("layer", "")),
        }
        records.append(rec)

    logger.info(f"[HOW] ✅ 收集 {len(records)} 条修辞框架记录")

    # ── 聚合摘要 ───────────────────────────────────────────────────────────
    if SUMMARY_CSV.exists():
        logger.info(f"[HOW] 读取 {SUMMARY_CSV.name}")
        summary = pd.read_csv(SUMMARY_CSV).to_dict(orient="records")
        logger.info(f"[HOW] 摘要: {len(summary)} 行")
    else:
        logger.warning(f"[HOW] 未找到 {SUMMARY_CSV.name}，聚合摘要为空")

    return records, summary
