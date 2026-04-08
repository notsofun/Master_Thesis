"""
pipeline/stages/why.py — WHY 阶段封装（RQ3 道德动机投影）
==========================================================
复用 unsupervised_classification/RQ3/main.py。
通过 subprocess 调用，避免模块级路径初始化冲突。

collect() 读取 rq3_bias_matrix.csv / rq3_anova_results.csv / rq3_summary.csv，
转换为统一 Schema。
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

# RQ3 产出的关键文件
BIAS_CSV  = RQ3_DATA_DIR / "rq3_bias_matrix.csv"
ANOVA_CSV = RQ3_DATA_DIR / "rq3_anova_results.csv"
SUMMARY_CSV = RQ3_DATA_DIR / "rq3_summary.csv"

# 7 个道德轴名称（与 RQ3/main.py 保持一致）
MORAL_AXES = ["Harm", "Fairness", "Loyalty", "Authority", "Sanctity", "RealThreat", "SymThreat"]


def run(config: dict) -> bool:
    """
    调用 RQ3 脚本完成道德动机投影。
    返回 True = 成功，False = 失败。

    config 字段（来自 config.yaml 的 why 节）：
      max_docs   : int  — 调试：仅处理前 N 条文档
      batch_size : int  — E5 编码批大小
      from_bias  : bool — 跳过轴构建+编码，从已有 bias_matrix 开始
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

    logger.info(f"[WHY] 启动 RQ3 脚本: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))

    if result.returncode != 0:
        logger.error(f"[WHY] RQ3 脚本退出码 {result.returncode}，请检查上方日志")
        return False

    logger.info("[WHY] ✅ RQ3 完成")
    return True


def collect(config: dict) -> tuple[list[dict], list[dict]]:
    """
    读取 RQ3 输出结果，返回 (per-document 道德偏移记录列表, 统计摘要列表)。

    per-document 记录：MoralBiasRecord.to_dict() 格式
      topic / lang / text / bias: {axis → float}

    统计摘要：rq3_summary.csv 原样转 dict 列表（论文用）
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
