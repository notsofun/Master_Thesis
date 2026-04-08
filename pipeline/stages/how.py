"""
pipeline/stages/how.py — HOW 阶段封装（RQ2 修辞策略提取）
===========================================================
复用 unsupervised_classification/RQ2/rq2_pipeline_v2.py。
通过 subprocess 调用，避免模块级路径初始化冲突。

collect() 读取 rq2_framing_labeled.csv 和 rq2_aggregated_summary.csv，
转换为统一 Schema。
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

# RQ2 产出的关键文件
LABELED_CSV  = RQ2_DATA_DIR / "rq2_framing_labeled.csv"
SUMMARY_CSV  = RQ2_DATA_DIR / "rq2_aggregated_summary.csv"


def run(config: dict) -> bool:
    """
    调用 RQ2 脚本完成修辞策略提取与框架分类。
    返回 True = 成功，False = 失败。

    config 字段（来自 config.yaml 的 how 节）：
      no_gemini : bool  — 跳过 Gemini，仅用规则分类器
      max_rows  : int   — 调试：仅处理前 N 行文档
    """
    how_cfg = config.get("how", {})

    cmd = [sys.executable, str(RQ2_SCRIPT)]

    if how_cfg.get("no_gemini", False):
        cmd.append("--no-gemini")

    max_rows = how_cfg.get("max_rows")
    if max_rows:
        cmd += ["--max-rows", str(max_rows)]

    logger.info(f"[HOW] 启动 RQ2 脚本: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))

    if result.returncode != 0:
        logger.error(f"[HOW] RQ2 脚本退出码 {result.returncode}，请检查上方日志")
        return False

    logger.info("[HOW] ✅ RQ2 完成")
    return True


def collect(config: dict) -> tuple[list[dict], list[dict]]:
    """
    读取 RQ2 输出结果，返回 (per-document 记录列表, 聚合摘要列表)。

    per-document 记录：ExpressionRecord.to_dict() 格式
      topic / lang / text / predicate / context / target / frame_type / layer

    聚合摘要：rq2_aggregated_summary.csv 原样转 dict 列表（论文附录用）
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
