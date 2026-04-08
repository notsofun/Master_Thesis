"""
pipeline/schema.py — 统一输出 Schema
=====================================
定义管线各阶段的标准化数据结构。
使用普通 dataclass（不引入 Pydantic），轻量、可序列化。

数据流：
  WHO  → WhoResult（per-topic 目标实体列表）
  HOW  → HowResult（per-document 修辞框架标签）
  WHY  → WhyResult（per-topic/lang 道德轴偏移分）
  合并 → DocumentRecord（per-document 三层结果）
  汇总 → PipelineResult（全量 + 统计摘要）
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any


# ── WHO / RQ1 ────────────────────────────────────────────────────────────────

@dataclass
class TopicTargets:
    """每个话题识别出的仇恨对象实体列表"""
    topic: int
    lang: str
    targets: list[str] = field(default_factory=list)      # Top-N 实体词
    target_counts: dict[str, int] = field(default_factory=dict)  # 实体频次

    def to_dict(self) -> dict:
        return asdict(self)


# ── HOW / RQ2 ────────────────────────────────────────────────────────────────

@dataclass
class ExpressionRecord:
    """单条 SVO / 谓词窗口提取结果"""
    topic: int
    lang: str
    text: str           # 原始文档
    predicate: str      # 谓词（动词短语）
    context: str        # 上下文窗口
    target: str         # 目标实体
    frame_type: str     # 修辞框架（10类）
    layer: str          # 提取层：svo / window / llm_fallback

    def to_dict(self) -> dict:
        return asdict(self)


# ── WHY / RQ3 ────────────────────────────────────────────────────────────────

MORAL_AXES = [
    "Harm", "Fairness", "Loyalty", "Authority", "Sanctity",
    "RealThreat", "SymThreat",
]

@dataclass
class MoralBiasRecord:
    """单条文档在7个道德轴上的偏移分"""
    topic: int
    lang: str
    text: str
    bias: dict[str, float] = field(default_factory=dict)  # axis_name → bias_score

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MoralAxisSummary:
    """按语言/话题聚合的道德轴均值"""
    group_key: str          # e.g. "lang=zh" or "topic=3"
    axis_means: dict[str, float] = field(default_factory=dict)
    anova_pvalues: dict[str, float] = field(default_factory=dict)  # axis → p-value

    def to_dict(self) -> dict:
        return asdict(self)


# ── 汇总级别 Schema ──────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    """整个管线的汇总结果，用于导出"""

    # 元信息
    input_path: str = ""
    run_timestamp: str = ""
    stages_run: list[str] = field(default_factory=list)
    total_documents: int = 0
    language_counts: dict[str, int] = field(default_factory=dict)
    topic_count: int = 0

    # WHO 结果：topic → TopicTargets
    who_results: list[dict] = field(default_factory=list)

    # HOW 结果：per-document 修辞框架
    how_results: list[dict] = field(default_factory=list)

    # HOW 聚合摘要：frame_type × (topic, lang) 频次
    how_summary: list[dict] = field(default_factory=list)

    # WHY 结果：per-document 道德轴偏移
    why_results: list[dict] = field(default_factory=list)

    # WHY 统计摘要：ANOVA + 语言/话题均值
    why_summary: list[dict] = field(default_factory=list)

    # 失败/跳过样本
    errors: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def add_error(self, stage: str, detail: str, text: str = ""):
        self.errors.append({"stage": stage, "detail": detail, "text": text[:200]})

    # ── 便捷访问 ────────────────────────────────────────────────────────────

    @property
    def who_df_records(self) -> list[dict]:
        """展平 who_results 供 CSV 导出"""
        rows = []
        for t in self.who_results:
            for entity, cnt in t.get("target_counts", {}).items():
                rows.append({
                    "topic": t["topic"],
                    "lang": t["lang"],
                    "target": entity,
                    "count": cnt,
                })
        return rows

    @property
    def how_frame_counts(self) -> dict[str, int]:
        """全量框架类型频次"""
        counts: dict[str, int] = {}
        for r in self.how_results:
            ft = r.get("frame_type", "unknown")
            counts[ft] = counts.get(ft, 0) + 1
        return counts

    @property
    def why_axis_means(self) -> dict[str, float]:
        """全量道德轴均值"""
        sums: dict[str, float] = {}
        cnts: dict[str, int] = {}
        for r in self.why_results:
            for axis, val in r.get("bias", {}).items():
                sums[axis] = sums.get(axis, 0.0) + val
                cnts[axis] = cnts.get(axis, 0) + 1
        return {k: round(sums[k] / cnts[k], 4) for k in sums if cnts[k] > 0}
