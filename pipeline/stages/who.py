"""
pipeline/stages/who.py — WHO Stage Wrapper (RQ1 Target Entity Identification)
=============================================================================
Reuses unsupervised_classification/RQ1/target_extraction_v3.py,
Calls via subprocess to maintain script integrity, avoiding module-level path initialization conflicts.

After execution, reads checkpoint CSV and converts results to unified schema.
"""

import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Repository root (two levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RQ1_SCRIPT   = PROJECT_ROOT / "unsupervised_classification" / "RQ1" / "target_extraction_v3.py"
RQ1_DATA_DIR = PROJECT_ROOT / "unsupervised_classification" / "RQ1" / "data"


def run(config: dict) -> bool:
    """Call RQ1 script to complete target entity identification.
    Return True = success, False = failure.

    Config fields (from config.yaml who section):
      stage          : layer12 / llm / full
      spacy_fallback : bool
      llm_type       : gemini / openai
      llm_model      : model name
      concurrency    : LLM concurrency
      top_n          : Top-N entities
      input          : input CSV path (document_topic_mapping.csv)
    """
    who_cfg  = config.get("who", {})
    stage    = who_cfg.get("stage", "full")
    input_p  = config.get("input", "")

    cmd = [
        sys.executable,
        str(RQ1_SCRIPT),
        "--stage", stage,
        "--input", str(input_p),
        "--output-dir", str(RQ1_DATA_DIR),
        "--top-n", str(who_cfg.get("top_n", 10)),
        "--llm-type", who_cfg.get("llm_type", "gemini"),
        "--llm-model", who_cfg.get("llm_model", "gemini-2.5-flash-lite"),
        "--concurrency", str(who_cfg.get("concurrency", 50)),
    ]
    if who_cfg.get("spacy_fallback", False):
        cmd.append("--spacy-fallback")

    logger.info(f"[WHO] Starting RQ1 script: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))

    if result.returncode != 0:
        logger.error(f"[WHO] RQ1 script exited with code {result.returncode}, check logs above")
        return False

    logger.info("[WHO] ✅ RQ1 completed")
    return True


def collect(config: dict) -> list[dict]:
    """Read RQ1 checkpoint CSV, convert to unified schema list.
    Each record corresponds to a (topic, lang) combination with Top entities.

    Return: list of TopicTargets.to_dict()
    """
    # Priority: LLM checkpoint > layer12 checkpoint > summarized rq1 file
    candidates = [
        RQ1_DATA_DIR / "checkpoint_llm.csv",
        RQ1_DATA_DIR / "checkpoint_layer12.csv",
        RQ1_DATA_DIR / "rq1_topic_targets_v3.csv",
    ]
    ckpt_path = next((p for p in candidates if p.exists()), None)

    if ckpt_path is None:
        logger.warning("[WHO] 未找到任何 RQ1 checkpoint，跳过 collect")
        return []

    logger.info(f"[WHO] 读取 checkpoint: {ckpt_path.name}")
    df = pd.read_csv(ckpt_path)

    # checkpoint 可能有不同的列名，统一处理
    results: list[dict] = []

    # 情况1：document_topic_mapping 格式（每行一条文档）
    if "topic" in df.columns and "lang" in df.columns:
        # 如果有 entities 列（layer12/llm checkpoint 格式）
        entity_col = next(
            (c for c in ["entities", "targets", "entity_list"] if c in df.columns),
            None
        )
        for (topic, lang), grp in df.groupby(["topic", "lang"]):
            if int(topic) == -1:
                continue
            targets_counter: dict[str, int] = {}
            if entity_col:
                for raw in grp[entity_col].dropna():
                    # 可能是 Python list repr 或逗号分隔
                    items = _parse_entity_list(raw)
                    for item in items:
                        if item:
                            targets_counter[item] = targets_counter.get(item, 0) + 1
            results.append({
                "topic": int(topic),
                "lang": str(lang),
                "targets": sorted(targets_counter, key=lambda k: -targets_counter[k])[:20],
                "target_counts": dict(
                    sorted(targets_counter.items(), key=lambda kv: -kv[1])[:20]
                ),
            })

    # 情况2：已经是 rq1 汇总格式（topic / entity / count）
    elif "entity" in df.columns and "count" in df.columns:
        for (topic, lang), grp in df.groupby(["topic", "lang"] if "lang" in df.columns else ["topic"]):
            lang_val = str(lang) if "lang" in df.columns else "unknown"
            top = grp.nlargest(20, "count")
            targets_counter = dict(zip(top["entity"], top["count"]))
            results.append({
                "topic": int(topic),
                "lang": lang_val,
                "targets": list(targets_counter.keys()),
                "target_counts": targets_counter,
            })

    logger.info(f"[WHO] ✅ 收集 {len(results)} 个 (topic, lang) 目标实体记录")
    return results


def _parse_entity_list(raw: Any) -> list[str]:
    """解析实体字段，支持 Python list repr 或逗号分隔字符串"""
    if isinstance(raw, list):
        return [str(x).strip() for x in raw]
    s = str(raw).strip()
    if s.startswith("["):
        import ast
        try:
            items = ast.literal_eval(s)
            return [str(x).strip() for x in items if x]
        except Exception:
            pass
    return [x.strip() for x in s.split(",") if x.strip()]
