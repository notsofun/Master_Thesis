"""
pipeline/orchestrator.py — Pipeline Orchestration Core
=======================================================
Unified scheduling of WHO → HOW → WHY three stages, collect results, write to unified schema.

Design Principles:
  - Each stage calls original script via subprocess, does not modify core research logic
  - Stage failures are logged, subsequent stages continue by default (strict mode for immediate termination)
  - Intermediate results written to each RQ's native output directory (preserve existing data flow)
  - Aggregated results written to pipeline/outputs/
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Add project root to sys.path, to import pipeline modules
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.stages.who import run as run_who, collect as collect_who
from pipeline.stages.how import run as run_how, collect as collect_how
from pipeline.stages.why import run as run_why, collect as collect_why
from pipeline.schema import PipelineResult
from pipeline.export import export_all

logger = logging.getLogger(__name__)


def run(config: dict, strict: bool = False) -> PipelineResult:
    """Execute complete WHO → HOW → WHY pipeline.

    Args:
        config  : Configuration dict parsed from config.yaml
        strict  : True = fail immediately on any stage failure; False = log error and continue

    Returns:
        PipelineResult: Contains three-layer results + error info + metadata
    """
    stages_cfg  = config.get("stages", {})
    output_dir  = Path(config.get("output", "pipeline/outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    result = PipelineResult(
        input_path    = str(config.get("input", "")),
        run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        stages_run    = [],
    )

    # ── 填充基础统计（读取输入文档） ──────────────────────────────────────
    input_path = Path(config.get("input", ""))
    _populate_doc_stats(result, input_path, config.get("lang", "auto"))

    # ══════════════════════════════════════════════════════════════════════
    # Stage 1: WHO — 目标实体识别 (RQ1)
    # ══════════════════════════════════════════════════════════════════════
    if stages_cfg.get("who", True):
        logger.info("=" * 60)
        logger.info("▶ WHO 阶段：目标实体识别 (RQ1)")
        logger.info("=" * 60)

        ok = run_who(config)
        if not ok:
            msg = "RQ1 (WHO) 脚本运行失败"
            result.add_error("who", msg)
            logger.error(f"[WHO] ❌ {msg}")
            if strict:
                raise RuntimeError(msg)
        else:
            result.who_results = collect_who(config)
            result.stages_run.append("who")
            logger.info(f"[WHO] ✅ 识别 {len(result.who_results)} 个 (topic, lang) 组合")
    else:
        logger.info("[WHO] ⏭ 已跳过（stages.who = false）")
        # 即使跳过执行，也尝试收集已有结果
        result.who_results = collect_who(config)
        if result.who_results:
            result.stages_run.append("who_cached")

    # ══════════════════════════════════════════════════════════════════════
    # Stage 2: HOW — 修辞策略提取 (RQ2)
    # ══════════════════════════════════════════════════════════════════════
    if stages_cfg.get("how", True):
        logger.info("=" * 60)
        logger.info("▶ HOW 阶段：修辞策略提取 (RQ2)")
        logger.info("=" * 60)

        ok = run_how(config)
        if not ok:
            msg = "RQ2 (HOW) 脚本运行失败"
            result.add_error("how", msg)
            logger.error(f"[HOW] ❌ {msg}")
            if strict:
                raise RuntimeError(msg)
        else:
            records, summary = collect_how(config)
            result.how_results  = records
            result.how_summary  = summary
            result.stages_run.append("how")
            logger.info(
                f"[HOW] ✅ {len(records)} 条修辞记录 | "
                f"框架分布: {_frame_counts(records)}"
            )
    else:
        logger.info("[HOW] ⏭ 已跳过（stages.how = false）")
        records, summary = collect_how(config)
        result.how_results = records
        result.how_summary = summary
        if records:
            result.stages_run.append("how_cached")

    # ══════════════════════════════════════════════════════════════════════
    # Stage 3: WHY — 道德动机投影 (RQ3)
    # ══════════════════════════════════════════════════════════════════════
    if stages_cfg.get("why", True):
        logger.info("=" * 60)
        logger.info("▶ WHY 阶段：道德动机投影 (RQ3)")
        logger.info("=" * 60)

        ok = run_why(config)
        if not ok:
            msg = "RQ3 (WHY) 脚本运行失败"
            result.add_error("why", msg)
            logger.error(f"[WHY] ❌ {msg}")
            if strict:
                raise RuntimeError(msg)
        else:
            records, summary = collect_why(config)
            result.why_results = records
            result.why_summary = summary
            result.stages_run.append("why")
            logger.info(
                f"[WHY] ✅ {len(records)} 条道德偏移记录 | "
                f"轴均值: {result.why_axis_means}"
            )
    else:
        logger.info("[WHY] ⏭ 已跳过（stages.why = false）")
        records, summary = collect_why(config)
        result.why_results = records
        result.why_summary = summary
        if records:
            result.stages_run.append("why_cached")

    # ══════════════════════════════════════════════════════════════════════
    # 导出
    # ══════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("▶ 导出结果")
    logger.info("=" * 60)

    export_formats = config.get("export", {}).get("formats", ["jsonl", "csv", "markdown"])
    exported_files = export_all(result, output_dir, formats=export_formats)

    logger.info("=" * 60)
    logger.info("✅ 管线全部完成")
    logger.info(f"   已运行阶段: {result.stages_run}")
    logger.info(f"   输出文件:")
    for f in exported_files:
        logger.info(f"     {f}")
    if result.errors:
        logger.warning(f"   ⚠ 有 {len(result.errors)} 个错误，详见输出 JSON")
    logger.info("=" * 60)

    return result


# ── 内部辅助函数 ──────────────────────────────────────────────────────────────

def _populate_doc_stats(result: PipelineResult, input_path: Path, lang_filter: str):
    """读取输入文档，填充 total_documents / language_counts / topic_count"""
    if not input_path.exists():
        logger.warning(f"[META] Input file not found: {input_path}")
        return
    try:
        import pandas as pd
        df = pd.read_csv(input_path)
        if lang_filter and lang_filter != "auto":
            df = df[df["lang"] == lang_filter]
        df = df[df["topic"] != -1]  # Exclude noise topic
        result.total_documents = len(df)
        result.language_counts = df["lang"].value_counts().to_dict() if "lang" in df.columns else {}
        result.topic_count = df["topic"].nunique() if "topic" in df.columns else 0
    except Exception as e:
        logger.warning(f"[META] Cannot read input file statistics: {e}")


def _frame_counts(records: list[dict]) -> dict[str, int]:
    """Count rhetorical frame frequency (for debug output)"""
    counts: dict[str, int] = {}
    for r in records:
        ft = r.get("frame_type", "?")
        counts[ft] = counts.get(ft, 0) + 1
    # Sort by frequency descending, keep only top-5
    return dict(sorted(counts.items(), key=lambda kv: -kv[1])[:5])
