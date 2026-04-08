"""
pipeline/schema.py — Unified Output Schema
===========================================
Defines standardized data structures for each stage of the pipeline.
Uses simple dataclasses (no Pydantic), lightweight and serializable.

Data flow:
  WHO  → WhoResult (per-topic target entity list)
  HOW  → HowResult (per-document rhetorical frame label)
  WHY  → WhyResult (per-topic/lang moral axis bias scores)
  Merge → DocumentRecord (per-document three-layer results)
  Aggregate → PipelineResult (full + statistical summary)
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any


# -- WHO / RQ1

@dataclass
class TopicTargets:
    """List of identified hate target entities for each topic"""
    topic: int
    lang: str
    targets: list[str] = field(default_factory=list)      # Top-N entity words
    target_counts: dict[str, int] = field(default_factory=dict)  # Entity frequency

    def to_dict(self) -> dict:
        return asdict(self)


# -- HOW / RQ2

@dataclass
class ExpressionRecord:
    """Single SVO / predicate window extraction result"""
    topic: int
    lang: str
    text: str           # Original document
    predicate: str      # Predicate (verb phrase)
    context: str        # Context window
    target: str         # Target entity
    frame_type: str     # Rhetorical frame (10 types)
    layer: str          # Extraction layer: svo / window / llm_fallback

    def to_dict(self) -> dict:
        return asdict(self)


# -- WHY / RQ3

MORAL_AXES = [
    "Harm", "Fairness", "Loyalty", "Authority", "Sanctity",
    "RealThreat", "SymThreat",
]

@dataclass
class MoralBiasRecord:
    """Single document bias scores on 7 moral axes"""
    topic: int
    lang: str
    text: str
    bias: dict[str, float] = field(default_factory=dict)  # axis_name → bias_score

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MoralAxisSummary:
    """Aggregated mean moral axis scores per language/topic"""
    group_key: str          # e.g. "lang=zh" or "topic=3"
    axis_means: dict[str, float] = field(default_factory=dict)
    anova_pvalues: dict[str, float] = field(default_factory=dict)  # axis → p-value

    def to_dict(self) -> dict:
        return asdict(self)


# -- Aggregation-level Schema

@dataclass
class PipelineResult:
    """Aggregated pipeline results for export"""

    # Metadata
    input_path: str = ""
    run_timestamp: str = ""
    stages_run: list[str] = field(default_factory=list)
    total_documents: int = 0
    language_counts: dict[str, int] = field(default_factory=dict)
    topic_count: int = 0

    # WHO results: topic → TopicTargets
    who_results: list[dict] = field(default_factory=list)

    # HOW results: per-document rhetorical frame
    how_results: list[dict] = field(default_factory=list)

    # HOW aggregated summary: frame_type × (topic, lang) frequency
    how_summary: list[dict] = field(default_factory=list)

    # WHY results: per-document moral axis bias
    why_results: list[dict] = field(default_factory=list)

    # WHY statistical summary: ANOVA + language/topic means
    why_summary: list[dict] = field(default_factory=list)

    # Failed/skipped samples
    errors: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def add_error(self, stage: str, detail: str, text: str = ""):
        self.errors.append({"stage": stage, "detail": detail, "text": text[:200]})

    # -- Convenient accessors

    @property
    def who_df_records(self) -> list[dict]:
        """Flatten who_results for CSV export"""
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
        """Full frame type frequency"""
        counts: dict[str, int] = {}
        for r in self.how_results:
            ft = r.get("frame_type", "unknown")
            counts[ft] = counts.get(ft, 0) + 1
        return counts

    @property
    def why_axis_means(self) -> dict[str, float]:
        """Full moral axis means"""
        sums: dict[str, float] = {}
        cnts: dict[str, int] = {}
        for r in self.why_results:
            for axis, val in r.get("bias", {}).items():
                sums[axis] = sums.get(axis, 0.0) + val
                cnts[axis] = cnts.get(axis, 0) + 1
        return {k: round(sums[k] / cnts[k], 4) for k in sums if cnts[k] > 0}
