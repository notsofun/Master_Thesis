"""
pipeline/smoke_test.py — 最小烟雾测试
======================================
不调用任何外部 API，不依赖 GPU，在 < 30 秒内验证：
  1. 配置加载 ✓
  2. Schema 创建与序列化 ✓
  3. 导出函数（JSONL / CSV / Markdown）✓
  4. 阶段包装器 import 和 collect()（需要 checkpoint 文件存在）
  5. Orchestrator 在 skip-all + collect-only 模式下的调度逻辑 ✓

运行方式：
  python pipeline/smoke_test.py
  python pipeline/smoke_test.py --verbose

不需要安装任何额外依赖（pandas、PyYAML 除外，与主管线共享）。
"""

import json
import sys
import tempfile
import traceback
from pathlib import Path

# ── 项目根 ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

PASS  = "✅"
FAIL  = "❌"
SKIP  = "⏭"

results: list[tuple[str, bool, str]] = []   # (test_name, ok, detail)


def test(name: str):
    """装饰器：捕获异常，记录结果"""
    def decorator(fn):
        def wrapper(*args, **kwargs):
            try:
                fn(*args, **kwargs)
                results.append((name, True, ""))
                print(f"  {PASS} {name}")
            except Exception as e:
                detail = traceback.format_exc()
                results.append((name, False, detail))
                print(f"  {FAIL} {name}")
                print(f"     {e}")
        return wrapper
    return decorator


# ══════════════════════════════════════════════════════════════════════════════
# Tests
# ══════════════════════════════════════════════════════════════════════════════

@test("1. Schema — PipelineResult 创建与 to_dict()")
def test_schema():
    from pipeline.schema import PipelineResult, TopicTargets, ExpressionRecord, MoralBiasRecord

    r = PipelineResult(
        input_path="test.csv",
        run_timestamp="2026-01-01 00:00:00",
        stages_run=["who", "how", "why"],
        total_documents=100,
        language_counts={"zh": 40, "en": 35, "ja": 25},
        topic_count=5,
        who_results=[
            {"topic": 0, "lang": "zh", "targets": ["基督教", "外来者"], "target_counts": {"基督教": 10}},
        ],
        how_results=[
            {"topic": 0, "lang": "zh", "text": "test", "predicate": "渗透", "context": "...",
             "target": "基督教", "frame_type": "dehumanization", "layer": "svo"},
        ],
        why_results=[
            {"topic": 0, "lang": "zh", "text": "test",
             "bias": {"Harm": -0.42, "Sanctity": -0.38}},
        ],
    )
    d = r.to_dict()
    assert "who_results" in d
    assert "how_results" in d
    assert "why_results" in d
    assert r.how_frame_counts == {"dehumanization": 1}
    assert "Harm" in r.why_axis_means


@test("2. Schema — add_error() 与 errors 字段")
def test_schema_errors():
    from pipeline.schema import PipelineResult
    r = PipelineResult()
    r.add_error("who", "Test error", "sample text")
    assert len(r.errors) == 1
    assert r.errors[0]["stage"] == "who"


@test("3. Export — JSONL 导出")
def test_export_jsonl():
    from pipeline.schema import PipelineResult
    from pipeline.export import export_jsonl

    r = _make_test_result()
    with tempfile.TemporaryDirectory() as tmp:
        files = export_jsonl(r, Path(tmp))
        assert len(files) == 2
        # 验证 JSONL 可解析
        jsonl_file = next(f for f in files if f.suffix == ".jsonl")
        lines = jsonl_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) > 0
        obj = json.loads(lines[0])
        assert "text" in obj and "who" in obj and "how" in obj and "why" in obj


@test("4. Export — CSV 导出")
def test_export_csv():
    from pipeline.schema import PipelineResult
    from pipeline.export import export_csv

    r = _make_test_result()
    with tempfile.TemporaryDirectory() as tmp:
        files = export_csv(r, Path(tmp))
        # 至少应该有 who/how/why/summary 中的几个
        assert len(files) >= 2
        for f in files:
            assert f.exists()
            assert f.stat().st_size > 0


@test("5. Export — Markdown 导出")
def test_export_markdown():
    from pipeline.schema import PipelineResult
    from pipeline.export import export_markdown

    r = _make_test_result()
    with tempfile.TemporaryDirectory() as tmp:
        files = export_markdown(r, Path(tmp))
        assert len(files) == 1
        content = files[0].read_text(encoding="utf-8")
        assert "WHO" in content
        assert "HOW" in content
        assert "WHY" in content
        assert "dehumanization" in content


@test("6. Export — export_all() 三格式全跑")
def test_export_all():
    from pipeline.schema import PipelineResult
    from pipeline.export import export_all

    r = _make_test_result()
    with tempfile.TemporaryDirectory() as tmp:
        files = export_all(r, Path(tmp), formats=["jsonl", "csv", "markdown"])
        assert len(files) >= 3


@test("7. Config — run_pipeline.py load_config()")
def test_load_config():
    from pipeline.run_pipeline import load_config
    config_path = PROJECT_ROOT / "pipeline" / "config.yaml"
    if config_path.exists():
        cfg = load_config(config_path)
        assert isinstance(cfg, dict)
        assert "input" in cfg or len(cfg) == 0  # 空也接受
    else:
        # 如果配置文件不存在，load_config 应返回空 dict
        cfg = load_config(Path("/nonexistent/path.yaml"))
        assert cfg == {}


@test("8. Config — merge_args_into_config() 覆盖逻辑")
def test_merge_args():
    from pipeline.run_pipeline import merge_args_into_config, build_parser

    # 构造一个 args namespace
    parser = build_parser()
    args = parser.parse_args([
        "--output", "/tmp/test_out",
        "--no-gemini",
        "--max-rows", "50",
        "--skip", "why",
    ])
    base_cfg = {
        "input": "data/input.csv",
        "output": "old_output",
        "stages": {"who": True, "how": True, "why": True},
    }
    merged = merge_args_into_config(base_cfg, args)
    assert merged["output"] == "/tmp/test_out"
    assert merged["stages"]["why"] == False
    assert merged["how"]["no_gemini"] == True
    assert merged["how"]["max_rows"] == 50


@test("9. Stages — who.py import 正常")
def test_who_import():
    from pipeline.stages.who import run, collect, _parse_entity_list
    # 测试辅助函数
    assert _parse_entity_list('["A", "B"]') == ["A", "B"]
    assert _parse_entity_list("A, B, C") == ["A", "B", "C"]
    assert _parse_entity_list(["X", "Y"]) == ["X", "Y"]


@test("10. Stages — how.py import 正常")
def test_how_import():
    from pipeline.stages.how import run, collect, LABELED_CSV, SUMMARY_CSV
    # 仅检查模块能加载，路径常量存在
    assert LABELED_CSV is not None
    assert SUMMARY_CSV is not None


@test("11. Stages — why.py import 正常")
def test_why_import():
    from pipeline.stages.why import run, collect, BIAS_CSV, MORAL_AXES
    assert len(MORAL_AXES) == 7


@test("12. Orchestrator — 跳过全部阶段，仅收集现有结果")
def test_orchestrator_skip_all():
    from pipeline.orchestrator import run as run_pipeline

    with tempfile.TemporaryDirectory() as tmp:
        # 使用不存在的输入路径 + 跳过全部执行阶段
        # 预期：collect() 均返回空列表，但不崩溃
        cfg = {
            "input": str(PROJECT_ROOT / "unsupervised_classification"
                         / "topic_modeling_results" / "sixth" / "data"
                         / "document_topic_mapping.csv"),
            "output": tmp,
            "lang": "auto",
            "stages": {"who": False, "how": False, "why": False},
            "export": {"formats": ["jsonl", "csv", "markdown"]},
        }
        result = run_pipeline(cfg, strict=False)
        # 仅验证不崩溃，不验证结果内容
        assert result is not None
        assert hasattr(result, "stages_run")


# ══════════════════════════════════════════════════════════════════════════════
# 辅助：构造测试用的 PipelineResult
# ══════════════════════════════════════════════════════════════════════════════

def _make_test_result():
    from pipeline.schema import PipelineResult
    return PipelineResult(
        input_path="test_input.csv",
        run_timestamp="2026-01-01 12:00:00",
        stages_run=["who", "how", "why"],
        total_documents=200,
        language_counts={"zh": 80, "en": 70, "ja": 50},
        topic_count=5,
        who_results=[
            {"topic": 0, "lang": "zh", "targets": ["基督教徒", "外来移民"], "target_counts": {"基督教徒": 25, "外来移民": 15}},
            {"topic": 1, "lang": "en", "targets": ["Christians", "immigrants"], "target_counts": {"Christians": 20, "immigrants": 12}},
            {"topic": 2, "lang": "ja", "targets": ["クリスチャン", "外国人"], "target_counts": {"クリスチャン": 18, "外国人": 10}},
        ],
        how_results=[
            {"topic": 0, "lang": "zh", "text": "他们在渗透我们的社区",
             "predicate": "渗透", "context": "他们在渗透我们的社区", "target": "基督教徒",
             "frame_type": "dehumanization", "layer": "svo"},
            {"topic": 1, "lang": "en", "text": "They are stealing our jobs",
             "predicate": "stealing", "context": "They are stealing our jobs", "target": "immigrants",
             "frame_type": "economic_exploitation", "layer": "window"},
            {"topic": 2, "lang": "ja", "text": "外国人が文化を壊している",
             "predicate": "壊している", "context": "外国人が文化を壊している", "target": "外国人",
             "frame_type": "social_exclusion", "layer": "svo"},
        ],
        how_summary=[
            {"frame_type": "dehumanization", "count": 45, "pct": 22.5},
            {"frame_type": "economic_exploitation", "count": 38, "pct": 19.0},
        ],
        why_results=[
            {"topic": 0, "lang": "zh", "text": "他们在渗透我们的社区",
             "bias": {"Harm": -0.42, "Fairness": -0.31, "Loyalty": 0.15,
                      "Authority": 0.08, "Sanctity": -0.38, "RealThreat": -0.45, "SymThreat": -0.52}},
            {"topic": 1, "lang": "en", "text": "They are stealing our jobs",
             "bias": {"Harm": -0.35, "Fairness": -0.48, "Loyalty": 0.22,
                      "Authority": 0.12, "Sanctity": -0.25, "RealThreat": -0.55, "SymThreat": -0.30}},
        ],
        why_summary=[
            {"group_key": "lang=zh", "Harm_mean": -0.42, "Sanctity_mean": -0.38},
            {"_type": "anova", "axis": "Harm", "F": 12.5, "p": 0.001},
        ],
    )


# ══════════════════════════════════════════════════════════════════════════════
# 运行
# ══════════════════════════════════════════════════════════════════════════════

def main():
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    print()
    print("=" * 60)
    print("Pipeline Smoke Test")
    print("=" * 60)
    print()

    # 运行所有测试函数（按定义顺序）
    test_schema()
    test_schema_errors()
    test_export_jsonl()
    test_export_csv()
    test_export_markdown()
    test_export_all()
    test_load_config()
    test_merge_args()
    test_who_import()
    test_how_import()
    test_why_import()
    test_orchestrator_skip_all()

    print()
    print("=" * 60)
    passed = sum(1 for _, ok, _ in results if ok)
    failed = sum(1 for _, ok, _ in results if not ok)
    print(f"结果: {passed} 通过 / {failed} 失败 / {len(results)} 总计")
    print("=" * 60)

    if verbose and failed:
        print()
        print("失败详情:")
        for name, ok, detail in results:
            if not ok:
                print(f"\n── {name} ──")
                print(detail)

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
