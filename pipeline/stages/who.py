"""
pipeline/stages/who.py — WHO 阶段封装（RQ1 目标实体识别）
==========================================================
复用 unsupervised_classification/RQ1/target_extraction_v3.py，
通过 subprocess 调用保持脚本完整性，避免 module-level 路径初始化冲突。

运行后从 checkpoint CSV 读取结果，转换为统一 Schema。
"""

import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# 仓库根目录（相对于本文件向上两层）
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RQ1_SCRIPT   = PROJECT_ROOT / "unsupervised_classification" / "RQ1" / "target_extraction_v3.py"
RQ1_DATA_DIR = PROJECT_ROOT / "unsupervised_classification" / "RQ1" / "data"


def run(config: dict) -> bool:
    """
    调用 RQ1 脚本完成目标实体识别。
    返回 True = 成功，False = 失败。

    config 字段（来自 config.yaml 的 who 节）：
      stage          : layer12 / llm / full
      spacy_fallback : bool
      llm_type       : gemini / openai
      llm_model      : 模型名
      concurrency    : LLM 并发数
      top_n          : Top-N 实体
      input          : 输入 CSV 路径（document_topic_mapping.csv）
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

    logger.info(f"[WHO] 启动 RQ1 脚本: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))

    if result.returncode != 0:
        logger.error(f"[WHO] RQ1 脚本退出码 {result.returncode}，请检查上方日志")
        return False

    logger.info("[WHO] ✅ RQ1 完成")
    return True


def collect(config: dict) -> list[dict]:
    """
    从 RQ1 输出的 checkpoint CSV 读取结果，转换为统一 Schema 列表。
    每条记录对应一个 (topic, lang) 组合，包含该组合的 Top 实体。

    返回：list of TopicTargets.to_dict()
    """
    # 优先读 LLM checkpoint，其次 layer12 checkpoint，最后 rq1 汇总文件
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
