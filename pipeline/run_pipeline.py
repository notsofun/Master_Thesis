"""
pipeline/run_pipeline.py — 统一命令行入口
==========================================
一条命令完成 WHO → HOW → WHY 三阶段分析，导出结构化结果。

用法（从仓库根目录运行）：

  # 完整流程（使用默认配置）
  python pipeline/run_pipeline.py

  # 指定输入 / 输出
  python pipeline/run_pipeline.py \\
      --input unsupervised_classification/topic_modeling_results/sixth/data/document_topic_mapping.csv \\
      --output pipeline/outputs/

  # 仅跑某阶段（跳过其余）
  python pipeline/run_pipeline.py --only who
  python pipeline/run_pipeline.py --only how
  python pipeline/run_pipeline.py --only why

  # 跳过某阶段
  python pipeline/run_pipeline.py --skip why

  # 调试模式（不调 Gemini，限制行数）
  python pipeline/run_pipeline.py --no-gemini --max-rows 100

  # 只重跑可视化（各 RQ 已有 checkpoint）
  python pipeline/run_pipeline.py --viz-only

  # 使用自定义配置文件
  python pipeline/run_pipeline.py --config pipeline/config.yaml

作者: Zhidian  |  日期: 2026-04
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# ── 确保项目根在 sys.path ───────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── 日志（使用项目统一的 set_logger） ──────────────────────────────────────
from scripts.set_logger import setup_logging

# ── 加载 .env ───────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)
except ImportError:
    pass


def load_config(config_path: Path) -> dict:
    """读取 YAML 配置文件，返回 dict。不存在则返回空 dict。"""
    if not config_path.exists():
        return {}
    try:
        import yaml
        with open(config_path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        # PyYAML 未安装时使用简单解析（仅支持 key: value 格式）
        cfg: dict = {}
        for line in config_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and ":" in line:
                k, _, v = line.partition(":")
                cfg[k.strip()] = v.strip()
        return cfg


def merge_args_into_config(cfg: dict, args: argparse.Namespace) -> dict:
    """
    命令行参数覆盖配置文件：
    优先级：命令行 > config.yaml > 代码默认值
    """
    import copy
    cfg = copy.deepcopy(cfg)

    if args.input:
        cfg["input"] = args.input
    if args.output:
        cfg["output"] = args.output
    if args.lang:
        cfg["lang"] = args.lang

    # stages 开关
    stages = cfg.setdefault("stages", {"who": True, "how": True, "why": True})
    if args.only:
        for k in stages:
            stages[k] = (k == args.only)
    if args.skip:
        for s in args.skip:
            stages[s] = False

    # WHO 参数
    who = cfg.setdefault("who", {})
    if args.who_stage:
        who["stage"] = args.who_stage
    if args.spacy_fallback:
        who["spacy_fallback"] = True

    # HOW 参数
    how = cfg.setdefault("how", {})
    if args.no_gemini:
        how["no_gemini"] = True
    if args.max_rows is not None:
        how["max_rows"] = args.max_rows

    # WHY 参数
    why = cfg.setdefault("why", {})
    if args.max_docs is not None:
        why["max_docs"] = args.max_docs
    if args.from_bias:
        why["from_bias"] = True

    # viz-only 模式：覆盖各阶段为 viz/cached
    if args.viz_only:
        who["stage"] = "viz"
        how["viz_only"] = True
        why["viz_only"] = True

    return cfg


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python pipeline/run_pipeline.py",
        description=(
            "Multilingual Hate Speech Analysis Pipeline\n"
            "WHO (target identification) → HOW (rhetorical strategies) → WHY (moral motivation)\n"
            "\n"
            "完整流程（默认）：\n"
            "  python pipeline/run_pipeline.py\n"
            "\n"
            "快速调试（不调 API，限制行数）：\n"
            "  python pipeline/run_pipeline.py --no-gemini --max-rows 50\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # 基础参数
    parser.add_argument(
        "--input", type=str, default=None,
        metavar="PATH",
        help=(
            "输入 CSV（需含 text / lang / topic 列）\n"
            "默认：config.yaml 中 input 字段"
        ),
    )
    parser.add_argument(
        "--output", type=str, default=None,
        metavar="DIR",
        help="输出目录（默认：pipeline/outputs/）",
    )
    parser.add_argument(
        "--lang", type=str, default=None,
        choices=["zh", "en", "ja", "auto"],
        help="语言过滤（auto = 不过滤，默认 auto）",
    )
    parser.add_argument(
        "--config", type=str,
        default=str(PROJECT_ROOT / "pipeline" / "config.yaml"),
        metavar="PATH",
        help="配置文件路径（默认：pipeline/config.yaml）",
    )

    # 阶段控制
    stage_group = parser.add_mutually_exclusive_group()
    stage_group.add_argument(
        "--only", type=str, choices=["who", "how", "why"],
        help="只运行指定阶段（与 --skip 互斥）",
    )
    stage_group.add_argument(
        "--skip", type=str, nargs="+", choices=["who", "how", "why"],
        metavar="STAGE",
        help="跳过指定阶段，可多选（e.g. --skip who why）",
    )

    # 通用模式开关
    parser.add_argument(
        "--viz-only", action="store_true",
        help="只重跑可视化（各 RQ 须已有 checkpoint）",
    )
    parser.add_argument(
        "--no-gemini", action="store_true",
        help="跳过所有 Gemini 调用，使用规则分类器（离线/调试）",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="任意阶段失败即终止（默认：记录错误后继续）",
    )

    # WHO 参数
    who_group = parser.add_argument_group("WHO (RQ1) 参数")
    who_group.add_argument(
        "--who-stage", choices=["layer12", "llm", "viz", "full"], default=None,
        help="RQ1 运行阶段（默认：full）",
    )
    who_group.add_argument(
        "--spacy-fallback", action="store_true",
        help="使用轻量 spaCy（显存不足时）",
    )

    # HOW 参数
    how_group = parser.add_argument_group("HOW (RQ2) 参数")
    how_group.add_argument(
        "--max-rows", type=int, default=None, metavar="N",
        help="调试：HOW 阶段仅处理前 N 条文档",
    )

    # WHY 参数
    why_group = parser.add_argument_group("WHY (RQ3) 参数")
    why_group.add_argument(
        "--max-docs", type=int, default=None, metavar="N",
        help="调试：WHY 阶段仅处理前 N 条文档",
    )
    why_group.add_argument(
        "--from-bias", action="store_true",
        help="WHY 阶段跳过轴构建+编码，从已有 bias_matrix.csv 开始",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # ── 初始化日志 ─────────────────────────────────────────────────────────
    log, log_path = setup_logging("pipeline")
    log.info("=" * 60)
    log.info("Thesis Analysis Pipeline 启动")
    log.info("=" * 60)

    # ── 加载配置 ───────────────────────────────────────────────────────────
    config_path = Path(args.config)
    cfg = load_config(config_path)
    if not cfg:
        log.warning(f"配置文件未找到或为空: {config_path}，使用内置默认值")

    # 设置必要的默认值
    cfg.setdefault("input", str(
        PROJECT_ROOT / "unsupervised_classification"
        / "topic_modeling_results" / "sixth" / "data"
        / "document_topic_mapping.csv"
    ))
    cfg.setdefault("output", str(PROJECT_ROOT / "pipeline" / "outputs"))
    cfg.setdefault("lang", "auto")
    cfg.setdefault("stages", {"who": True, "how": True, "why": True})

    # 命令行参数覆盖配置文件
    cfg = merge_args_into_config(cfg, args)

    # ── 前置检查 ───────────────────────────────────────────────────────────
    input_path = Path(cfg["input"])
    if not input_path.exists():
        log.error(
            f"输入文件不存在: {input_path}\n"
            f"请检查 --input 参数或 config.yaml 中的 input 字段。\n"
            f"预期格式：含 text / lang / topic 列的 CSV 文件。"
        )
        sys.exit(1)

    log.info(f"配置文件  : {config_path}")
    log.info(f"输入文件  : {input_path}")
    log.info(f"输出目录  : {cfg['output']}")
    log.info(f"语言过滤  : {cfg.get('lang', 'auto')}")
    log.info(f"运行阶段  : {cfg.get('stages', {})}")
    log.info(f"日志文件  : {log_path}")

    # ── 运行管线 ───────────────────────────────────────────────────────────
    from pipeline.orchestrator import run as run_pipeline

    try:
        result = run_pipeline(cfg, strict=args.strict)
    except RuntimeError as e:
        log.error(f"管线在 strict 模式下提前终止: {e}")
        sys.exit(2)
    except KeyboardInterrupt:
        log.warning("用户中断（Ctrl+C）")
        sys.exit(130)
    except Exception as e:
        log.exception(f"管线发生未预期错误: {e}")
        sys.exit(3)

    # ── 结果摘要 ───────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("✅ 管线完成")
    log.info(f"   阶段: {result.stages_run}")
    log.info(f"   文档数: {result.total_documents}")
    log.info(f"   语言: {result.language_counts}")
    log.info(f"   WHO 目标组合: {len(result.who_results)}")
    log.info(f"   HOW 修辞记录: {len(result.how_results)}")
    log.info(f"   WHY 偏移记录: {len(result.why_results)}")
    if result.errors:
        log.warning(f"   ⚠ 错误: {len(result.errors)} 条（见输出 JSON）")
    log.info(f"   输出目录: {cfg['output']}")
    log.info("=" * 60)

    sys.exit(0)


if __name__ == "__main__":
    main()
