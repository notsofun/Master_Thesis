"""
pipeline/export.py — 结果导出工具
===================================
支持三种格式：
  1. JSONL  — 机器可读，每行一个 JSON 对象（per-document 记录）
  2. CSV    — 论文附录，各阶段分表输出（who/how/why/summary）
  3. Markdown — 论文 Discussion 章节可直接粘贴的汇总报告

所有导出均写入同一 output_dir，文件名带时间戳前缀（便于版本追踪）。
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 时间戳前缀（每次运行统一，由 export_all 设定）
_TS = datetime.now().strftime("%Y%m%d_%H%M%S")


def export_all(result, output_dir: Path, formats: list[str]) -> list[Path]:
    """
    根据 formats 列表导出所有格式。
    返回已生成的文件路径列表。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []

    if "jsonl" in formats:
        files += export_jsonl(result, output_dir)
    if "csv" in formats:
        files += export_csv(result, output_dir)
    if "markdown" in formats:
        files += export_markdown(result, output_dir)

    return files


# ══════════════════════════════════════════════════════════════════════════════
# 1. JSONL — 机器可读
# ══════════════════════════════════════════════════════════════════════════════

def export_jsonl(result, output_dir: Path) -> list[Path]:
    """
    导出两个 JSONL 文件：
      pipeline_results.jsonl  — per-document 三层合并记录（便于后处理）
      pipeline_meta.json      — 元信息 + 统计摘要（便于快速查阅）
    """
    files: list[Path] = []

    # ── per-document 三层合并 ─────────────────────────────────────────────
    doc_path = output_dir / f"{_TS}_pipeline_results.jsonl"
    with open(doc_path, "w", encoding="utf-8") as f:
        # HOW records 携带 text，以 text 为 key 关联 WHY bias
        why_index = _index_why(result.why_results)

        for rec in result.how_results:
            text_key = rec.get("text", "")[:200]
            merged = {
                "text":       text_key,
                "topic":      rec.get("topic"),
                "lang":       rec.get("lang"),
                # HOW
                "who": _who_for(rec.get("topic"), rec.get("lang"), result.who_results),
                # HOW
                "how": {
                    "predicate":  rec.get("predicate"),
                    "target":     rec.get("target"),
                    "frame_type": rec.get("frame_type"),
                    "layer":      rec.get("layer"),
                },
                # WHY
                "why": why_index.get(text_key, {}),
            }
            f.write(json.dumps(merged, ensure_ascii=False) + "\n")

    files.append(doc_path)
    logger.info(f"[EXPORT] JSONL → {doc_path.name}")

    # ── 元信息 + 统计摘要 ─────────────────────────────────────────────────
    meta_path = output_dir / f"{_TS}_pipeline_meta.json"
    meta = {
        "run_timestamp":    result.run_timestamp,
        "input_path":       result.input_path,
        "stages_run":       result.stages_run,
        "total_documents":  result.total_documents,
        "language_counts":  result.language_counts,
        "topic_count":      result.topic_count,
        "who_topic_count":  len(result.who_results),
        "how_record_count": len(result.how_results),
        "how_frame_counts": result.how_frame_counts,
        "why_record_count": len(result.why_results),
        "why_axis_means":   result.why_axis_means,
        "errors":           result.errors,
        "who_summary":      result.who_results,       # per-topic targets
        "how_agg_summary":  result.how_summary,       # aggregated frame table
        "why_agg_summary":  result.why_summary,       # ANOVA + bias means
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    files.append(meta_path)
    logger.info(f"[EXPORT] Meta JSON → {meta_path.name}")

    return files


# ══════════════════════════════════════════════════════════════════════════════
# 2. CSV — 论文附录用分表
# ══════════════════════════════════════════════════════════════════════════════

def export_csv(result, output_dir: Path) -> list[Path]:
    """
    导出四张 CSV：
      who_targets.csv      — 每个 (topic, lang) 的 Top 实体 + 频次
      how_expressions.csv  — 每条修辞框架记录（text / predicate / frame_type …）
      why_bias_matrix.csv  — 每条文档的道德轴偏移分
      pipeline_summary.csv — 跨三层的聚合统计（论文 Results 直接引用）
    """
    try:
        import pandas as pd
    except ImportError:
        logger.error("[EXPORT] pandas 未安装，跳过 CSV 导出")
        return []

    files: list[Path] = []

    # ── WHO: 实体频次表 ───────────────────────────────────────────────────
    who_rows = result.who_df_records
    if who_rows:
        p = output_dir / f"{_TS}_who_targets.csv"
        pd.DataFrame(who_rows).sort_values(["topic", "lang", "count"], ascending=[True, True, False])\
          .to_csv(p, index=False, encoding="utf-8-sig")
        files.append(p)
        logger.info(f"[EXPORT] WHO CSV → {p.name} ({len(who_rows)} 行)")

    # ── HOW: 修辞框架记录 ─────────────────────────────────────────────────
    if result.how_results:
        p = output_dir / f"{_TS}_how_expressions.csv"
        pd.DataFrame(result.how_results).to_csv(p, index=False, encoding="utf-8-sig")
        files.append(p)
        logger.info(f"[EXPORT] HOW CSV → {p.name} ({len(result.how_results)} 行)")

    # ── WHY: 道德轴偏移矩阵 ───────────────────────────────────────────────
    if result.why_results:
        # 展平 bias dict 到各列
        why_flat = []
        for r in result.why_results:
            row = {"topic": r["topic"], "lang": r["lang"], "text": r["text"]}
            row.update(r.get("bias", {}))
            why_flat.append(row)
        p = output_dir / f"{_TS}_why_bias_matrix.csv"
        pd.DataFrame(why_flat).to_csv(p, index=False, encoding="utf-8-sig")
        files.append(p)
        logger.info(f"[EXPORT] WHY CSV → {p.name} ({len(why_flat)} 行)")

    # ── 跨三层聚合摘要 ────────────────────────────────────────────────────
    summary_rows = _build_summary_rows(result)
    if summary_rows:
        p = output_dir / f"{_TS}_pipeline_summary.csv"
        pd.DataFrame(summary_rows).to_csv(p, index=False, encoding="utf-8-sig")
        files.append(p)
        logger.info(f"[EXPORT] Summary CSV → {p.name}")

    return files


def _build_summary_rows(result) -> list[dict]:
    """
    构建跨三层的聚合摘要行：
    每行 = 一个 (topic, lang) 组合，包含：
      - WHO: top targets
      - HOW: dominant frame type
      - WHY: mean bias scores
    """
    try:
        import pandas as pd
    except ImportError:
        return []

    rows: list[dict] = []

    # 构建 WHO 索引
    who_idx: dict[tuple, list] = {}
    for w in result.who_results:
        who_idx[(w["topic"], w["lang"])] = w.get("targets", [])[:5]

    # 构建 HOW 聚合（dominant frame per topic-lang）
    how_idx: dict[tuple, dict] = {}
    for h in result.how_results:
        key = (h.get("topic"), h.get("lang"))
        ft = h.get("frame_type", "other")
        if key not in how_idx:
            how_idx[key] = {}
        how_idx[key][ft] = how_idx[key].get(ft, 0) + 1

    # 构建 WHY 聚合（mean bias per topic-lang）
    why_sums: dict[tuple, dict] = {}
    why_cnts: dict[tuple, int] = {}
    for w in result.why_results:
        key = (w.get("topic"), w.get("lang"))
        if key not in why_sums:
            why_sums[key] = {}
            why_cnts[key] = 0
        for ax, val in w.get("bias", {}).items():
            why_sums[key][ax] = why_sums[key].get(ax, 0.0) + val
        why_cnts[key] += 1

    # 合并所有 (topic, lang) 组合
    all_keys = set(who_idx) | set(how_idx) | set(why_sums)
    for (topic, lang) in sorted(all_keys):
        row: dict = {"topic": topic, "lang": lang}

        # WHO
        row["who_top_targets"] = "; ".join(who_idx.get((topic, lang), []))

        # HOW
        frame_dist = how_idx.get((topic, lang), {})
        if frame_dist:
            dominant = max(frame_dist, key=lambda k: frame_dist[k])
            row["how_dominant_frame"] = dominant
            row["how_frame_count"] = sum(frame_dist.values())
        else:
            row["how_dominant_frame"] = ""
            row["how_frame_count"] = 0

        # WHY
        cnt = why_cnts.get((topic, lang), 0)
        for ax, total in why_sums.get((topic, lang), {}).items():
            row[f"why_{ax}_mean"] = round(total / cnt, 4) if cnt > 0 else None

        rows.append(row)

    return rows


# ══════════════════════════════════════════════════════════════════════════════
# 3. Markdown — 论文可直接引用的摘要报告
# ══════════════════════════════════════════════════════════════════════════════

def export_markdown(result, output_dir: Path) -> list[Path]:
    """
    导出一份 Markdown 报告，结构对应 who / how / why 三节。
    可直接粘贴到论文 Results / Discussion 章节。
    """
    lines = []
    ts = result.run_timestamp
    lines += [
        f"# Hate Speech Analysis Pipeline — Results Report",
        f"",
        f"> Generated: {ts}  ",
        f"> Input: `{result.input_path}`  ",
        f"> Stages run: {', '.join(result.stages_run)}  ",
        f"> Total documents (non-noise): **{result.total_documents}**  ",
        f"> Languages: {_fmt_dict(result.language_counts)}  ",
        f"> Topics: **{result.topic_count}**",
        f"",
        f"---",
        f"",
    ]

    # ── WHO ───────────────────────────────────────────────────────────────
    lines += [
        f"## WHO — Hate Target Identification (RQ1)",
        f"",
        f"Identified hate targets across **{len(result.who_results)}** topic-language groups "
        f"using a three-stage pipeline (spaCy NER + domain dictionary + LLM fallback).",
        f"",
    ]
    if result.who_results:
        lines.append("| Topic | Lang | Top Targets |")
        lines.append("|------:|------|-------------|")
        for w in sorted(result.who_results, key=lambda x: (x["topic"], x["lang"]))[:30]:
            targets_str = ", ".join(w.get("targets", [])[:5])
            lines.append(f"| {w['topic']} | {w['lang']} | {targets_str} |")
        if len(result.who_results) > 30:
            lines.append(f"| … | … | *(showing 30 of {len(result.who_results)})* |")
    lines.append("")

    # ── HOW ───────────────────────────────────────────────────────────────
    frame_counts = result.how_frame_counts
    total_how = sum(frame_counts.values())
    lines += [
        f"## HOW — Rhetorical Strategy Analysis (RQ2)",
        f"",
        f"Extracted **{total_how}** predicate-target expressions and classified them "
        f"into 10 rhetorical frame types using a rule-based + LLM hybrid classifier.",
        f"",
        f"### Frame Type Distribution",
        f"",
        f"| Frame Type | Count | % |",
        f"|------------|------:|--:|",
    ]
    for ft, cnt in sorted(frame_counts.items(), key=lambda kv: -kv[1]):
        pct = f"{100*cnt/total_how:.1f}" if total_how else "0.0"
        lines.append(f"| `{ft}` | {cnt} | {pct}% |")
    lines.append("")

    # HOW 聚合摘要（如有）
    if result.how_summary:
        lines += [
            f"### Aggregated Summary (first 10 rows)",
            f"",
        ]
        _md_table_from_dicts(lines, result.how_summary[:10])
        lines.append("")

    # ── WHY ───────────────────────────────────────────────────────────────
    axis_means = result.why_axis_means
    lines += [
        f"## WHY — Moral Motivation Analysis (RQ3)",
        f"",
        f"Projected **{len(result.why_results)}** documents onto 7 moral axes "
        f"(MFD 2.0 + Intergroup Threat Theory) using E5 multilingual embeddings.",
        f"",
        f"### Mean Moral Axis Bias (full corpus)",
        f"",
        f"Negative bias → documents lean toward the *harmful* pole of each axis.",
        f"",
        f"| Moral Axis | Mean Bias |",
        f"|------------|----------:|",
    ]
    for ax, mean in sorted(axis_means.items(), key=lambda kv: kv[1]):
        direction = "↓ harmful" if mean < 0 else "↑ positive"
        lines.append(f"| {ax} | {mean:.4f} {direction} |")
    lines.append("")

    # WHY 统计摘要（如有）
    why_anova = [r for r in result.why_summary if r.get("_type") == "anova"]
    if why_anova:
        lines += [
            f"### ANOVA Results (language differences)",
            f"",
        ]
        _md_table_from_dicts(lines, why_anova[:10])
        lines.append("")

    # ── 错误摘要 ──────────────────────────────────────────────────────────
    if result.errors:
        lines += [
            f"---",
            f"",
            f"## ⚠ Errors & Warnings",
            f"",
        ]
        for err in result.errors:
            lines.append(f"- **[{err['stage'].upper()}]** {err['detail']}")
        lines.append("")

    lines += [
        f"---",
        f"",
        f"*Report generated by `pipeline/run_pipeline.py` — "
        f"Minimal Reproducible Pipeline for Multilingual Hate Speech Analysis*",
    ]

    md_path = output_dir / f"{_TS}_report.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"[EXPORT] Markdown → {md_path.name}")
    return [md_path]


# ── 内部辅助 ──────────────────────────────────────────────────────────────────

def _index_why(why_results: list[dict]) -> dict[str, dict]:
    """以 text[:200] 为 key 建立 WHY 偏移索引"""
    idx: dict[str, dict] = {}
    for r in why_results:
        k = str(r.get("text", ""))[:200]
        idx[k] = r.get("bias", {})
    return idx


def _who_for(topic: Any, lang: Any, who_results: list[dict]) -> list[str]:
    """查找给定 (topic, lang) 对应的 Top 实体"""
    for w in who_results:
        if w.get("topic") == topic and w.get("lang") == lang:
            return w.get("targets", [])[:5]
    return []


def _fmt_dict(d: dict) -> str:
    return ", ".join(f"{k}: {v}" for k, v in d.items())


def _md_table_from_dicts(lines: list[str], rows: list[dict]):
    """将 dict 列表渲染为 Markdown 表格（追加到 lines）"""
    if not rows:
        return
    # 过滤掉 _type 内部字段
    keys = [k for k in rows[0].keys() if not k.startswith("_")]
    lines.append("| " + " | ".join(keys) + " |")
    lines.append("|" + "|".join(["---"] * len(keys)) + "|")
    for row in rows:
        vals = [str(row.get(k, "")) for k in keys]
        lines.append("| " + " | ".join(vals) + " |")
