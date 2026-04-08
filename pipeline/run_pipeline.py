"""
pipeline/run_pipeline.py — Unified CLI Entry Point
===================================================
Complete WHO → HOW → WHY three-stage analysis with one command, export structured results.

Usage (run from project root):

  # Full workflow (using default config)
  python pipeline/run_pipeline.py

  # Specify input / output
  python pipeline/run_pipeline.py \\
      --input unsupervised_classification/topic_modeling_results/sixth/data/document_topic_mapping.csv \\
      --output pipeline/outputs/

  # Run single stage only (skip rest)
  python pipeline/run_pipeline.py --only who
  python pipeline/run_pipeline.py --only how
  python pipeline/run_pipeline.py --only why

  # Skip stage
  python pipeline/run_pipeline.py --skip why

  # Debug mode (skip Gemini, limit rows)
  python pipeline/run_pipeline.py --no-gemini --max-rows 100

  # Visualization only (requires checkpoints)
  python pipeline/run_pipeline.py --viz-only

  # Use custom config file
  python pipeline/run_pipeline.py --config pipeline/config.yaml

Author: Zhidian | Date: 2026-04
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# ── 确保项目根在 sys.path ───────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging using project unified set_logger
from scripts.set_logger import setup_logging

# Load .env file
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)
except ImportError:
    pass


def load_config(config_path: Path) -> dict:
    """Load YAML config file, return dict; return empty dict if not found."""
    if not config_path.exists():
        return {}
    try:
        import yaml
        with open(config_path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        # Simple fallback parsing if PyYAML not installed (key: value format only)
        cfg: dict = {}
        for line in config_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and ":" in line:
                k, _, v = line.partition(":")
                cfg[k.strip()] = v.strip()
        return cfg


def merge_args_into_config(cfg: dict, args: argparse.Namespace) -> dict:
    """CLI args override config file. Priority: CLI > config.yaml > code defaults."""
    import copy
    cfg = copy.deepcopy(cfg)

    if args.input:
        cfg["input"] = args.input
    if args.output:
        cfg["output"] = args.output
    if args.lang:
        cfg["lang"] = args.lang

    # Stage switches
    stages = cfg.setdefault("stages", {"who": True, "how": True, "why": True})
    if args.only:
        for k in stages:
            stages[k] = (k == args.only)
    if args.skip:
        for s in args.skip:
            stages[s] = False

    # WHO parameters
    who = cfg.setdefault("who", {})
    if args.who_stage:
        who["stage"] = args.who_stage
    if args.spacy_fallback:
        who["spacy_fallback"] = True

    # HOW parameters
    how = cfg.setdefault("how", {})
    if args.no_gemini:
        how["no_gemini"] = True
    if args.max_rows is not None:
        how["max_rows"] = args.max_rows

    # WHY parameters
    why = cfg.setdefault("why", {})
    if args.max_docs is not None:
        why["max_docs"] = args.max_docs
    if args.from_bias:
        why["from_bias"] = True

    # Viz-only mode: override stages to viz/cached
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
            "Full workflow (default):\n"
            "  python pipeline/run_pipeline.py\n"
            "\n"
            "Quick debug (skip API, limit rows):\n"
            "  python pipeline/run_pipeline.py --no-gemini --max-rows 50\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Basic parameters
    parser.add_argument(
        "--input", type=str, default=None,
        metavar="PATH",
        help=(
            "Input CSV (must contain text / lang / topic columns)\n"
            "Default: input field in config.yaml"
        ),
    )
    parser.add_argument(
        "--output", type=str, default=None,
        metavar="DIR",
        help="Output directory (default: pipeline/outputs/)",
    )
    parser.add_argument(
        "--lang", type=str, default=None,
        choices=["zh", "en", "ja", "auto"],
        help="Language filter (auto = no filter, default auto)",
    )
    parser.add_argument(
        "--config", type=str,
        default=str(PROJECT_ROOT / "pipeline" / "config.yaml"),
        metavar="PATH",
        help="Config file path (default: pipeline/config.yaml)",
    )

    # Stage control
    stage_group = parser.add_mutually_exclusive_group()
    stage_group.add_argument(
        "--only", type=str, choices=["who", "how", "why"],
        help="Run only specified stage (mutually exclusive with --skip)",
    )
    stage_group.add_argument(
        "--skip", type=str, nargs="+", choices=["who", "how", "why"],
        metavar="STAGE",
        help="Skip specified stages, multiple allowed (e.g. --skip who why)",
    )

    # General mode switches
    parser.add_argument(
        "--viz-only", action="store_true",
        help="Visualization only (requires existing checkpoints)",
    )
    parser.add_argument(
        "--no-gemini", action="store_true",
        help="Skip all Gemini calls, use rule classifier (offline/debug)",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Fail immediately on any stage failure (default: log and continue)",
    )

    # WHO parameters
    who_group = parser.add_argument_group("WHO (RQ1) Parameters")
    who_group.add_argument(
        "--who-stage", choices=["layer12", "llm", "viz", "full"], default=None,
        help="RQ1 execution stage (default: full)",
    )
    who_group.add_argument(
        "--spacy-fallback", action="store_true",
        help="Use lightweight spaCy (when low on VRAM)",
    )

    # HOW parameters
    how_group = parser.add_argument_group("HOW (RQ2) Parameters")
    how_group.add_argument(
        "--max-rows", type=int, default=None, metavar="N",
        help="Debug: process only first N documents in HOW stage",
    )

    # WHY parameters
    why_group = parser.add_argument_group("WHY (RQ3) Parameters")
    why_group.add_argument(
        "--max-docs", type=int, default=None, metavar="N",
        help="Debug: process only first N documents in WHY stage",
    )
    why_group.add_argument(
        "--from-bias", action="store_true",
        help="Skip axis building+encoding, start from existing bias_matrix.csv",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Initialize logging
    log, log_path = setup_logging("pipeline")
    log.info("=" * 60)
    log.info("Thesis Analysis Pipeline starting")
    log.info("=" * 60)

    # Load configuration
    config_path = Path(args.config)
    cfg = load_config(config_path)
    if not cfg:
        log.warning(f"Config file not found or empty: {config_path}, using built-in defaults")

    # Set necessary default values
    cfg.setdefault("input", str(
        PROJECT_ROOT / "unsupervised_classification"
        / "topic_modeling_results" / "sixth" / "data"
        / "document_topic_mapping.csv"
    ))
    cfg.setdefault("output", str(PROJECT_ROOT / "pipeline" / "outputs"))
    cfg.setdefault("lang", "auto")
    cfg.setdefault("stages", {"who": True, "how": True, "why": True})

    # CLI args override config file
    cfg = merge_args_into_config(cfg, args)

    # Pre-execution checks
    input_path = Path(cfg["input"])
    if not input_path.exists():
        log.error(
            f"Input file not found: {input_path}\n"
            f"Check --input parameter or input field in config.yaml.\n"
            f"Expected format: CSV file with text / lang / topic columns."
        )
        sys.exit(1)

    log.info(f"Config file: {config_path}")
    log.info(f"Input file : {input_path}")
    log.info(f"Output dir : {cfg['output']}")
    log.info(f"Lang filter: {cfg.get('lang', 'auto')}")
    log.info(f"Stages     : {cfg.get('stages', {})}")
    log.info(f"Log file   : {log_path}")

    # Run pipeline
    from pipeline.orchestrator import run as run_pipeline

    try:
        result = run_pipeline(cfg, strict=args.strict)
    except RuntimeError as e:
        log.error(f"Pipeline terminated early in strict mode: {e}")
        sys.exit(2)
    except KeyboardInterrupt:
        log.warning("User interrupt (Ctrl+C)")
        sys.exit(130)
    except Exception as e:
        log.exception(f"Pipeline encountered unexpected error: {e}")
        sys.exit(3)

    # Result summary
    log.info("=" * 60)
    log.info("✅ Pipeline completed")
    log.info(f"   Stages: {result.stages_run}")
    log.info(f"   Documents: {result.total_documents}")
    log.info(f"   Languages: {result.language_counts}")
    log.info(f"   WHO targets: {len(result.who_results)}")
    log.info(f"   HOW records: {len(result.how_results)}")
    log.info(f"   WHY records: {len(result.why_results)}")
    if result.errors:
        log.warning(f"   ⚠ Errors: {len(result.errors)} (see output JSON)")
    log.info(f"   Output dir: {cfg['output']}")
    log.info("=" * 60)

    sys.exit(0)


if __name__ == "__main__":
    main()
