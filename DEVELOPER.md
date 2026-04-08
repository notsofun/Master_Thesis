# Developer Guide

**中文版本请参考 [DEVELOPER.zh.md](DEVELOPER.zh.md)**

## Architecture Overview

The pipeline orchestrates three independent research stages:

```
Input (document_topic_mapping.csv)
        ↓
    [WHO Stage — RQ1]  ← unsupervised_classification/RQ1/target_extraction_v3.py
        ↓
    [HOW Stage — RQ2] ← unsupervised_classification/RQ2/rq2_pipeline_v2.py
        ↓
    [WHY Stage — RQ3] ← unsupervised_classification/RQ3/main.py
        ↓
    [Aggregation & Export]
        ↓
    [JSONL / CSV / Markdown]
```

### Design Principles

1. **Non-invasive**: Each stage calls the original RQ script via subprocess without modifying core research logic
2. **Fault-tolerant**: Stage failures are logged; pipeline continues unless `--strict` mode
3. **Resumable**: Each stage reads/writes checkpoints for recovery from failures
4. **Modular**: Each RQ's outputs remain in their original directories; aggregation happens separately

---

## Module Structure

### Code Comments

All Python source files in the pipeline contain comments in Chinese (原始中文注释) with English translations. For a comprehensive reference of all Chinese comments and their English equivalents, see [BILINGUAL_CODE_COMMENTS.md](BILINGUAL_CODE_COMMENTS.md).

**When writing new code**, follow this pattern:
```python
# English comment | 中文注释
def new_function():
    """English docstring.
    中文文档字符串。"""
```

### `pipeline/run_pipeline.py`

**Entry point** — Parses CLI arguments, loads configuration, manages logging.

Key functions:
- `load_config(path)` — Read YAML config with fallback parsing
- `merge_args_into_config(cfg, args)` — CLI args override config file
- `build_parser()` — argparse setup

Run from project root:
```bash
python pipeline/run_pipeline.py [options]
```

### `pipeline/orchestrator.py`

**Orchestration core** — Schedules WHO → HOW → WHY stages, collects results, writes unified schema.

Key function:
- `run(config, strict=False)` — Execute pipeline, return PipelineResult

Each stage:
1. Calls subprocess with original RQ script
2. Logs success/failure
3. Collects results into unified Schema
4. Continues or stops based on `strict` mode

### `pipeline/schema.py`

**Data schema definitions** — Dataclasses for type-safe output.

Key classes:
- `TopicTargets` — WHO results (per-topic entities)
- `ExpressionRecord` — HOW results (per-document rhetorical frames)
- `MoralBiasRecord` — WHY results (per-document moral axis biases)
- `PipelineResult` — Aggregated result container

### `pipeline/export.py`

**Export logic** — Converts PipelineResult to JSONL, CSV, Markdown.

Key functions:
- `export_all(result, output_dir, formats)` — Route to format-specific exporters
- `export_jsonl(result, output_dir)` — Per-document + metadata JSONL
- `export_csv(result, output_dir)` — Four CSV tables
- `export_markdown(result, output_dir)` — Paper-ready report

### `pipeline/stages/who.py`

**RQ1 wrapper** — Calls `unsupervised_classification/RQ1/target_extraction_v3.py` via subprocess.

Key functions:
- `run(config)` — Execute RQ1 script
- `collect(config)` — Read checkpoint, parse entities, return TopicTargets list

### `pipeline/stages/how.py`

**RQ2 wrapper** — Calls `unsupervised_classification/RQ2/rq2_pipeline_v2.py`.

Key functions:
- `run(config)` — Execute RQ2 script
- `collect(config)` — Read labeled CSV, return ExpressionRecord list + summary

### `pipeline/stages/why.py`

**RQ3 wrapper** — Calls `unsupervised_classification/RQ3/main.py`.

Key functions:
- `run(config)` — Execute RQ3 script
- `collect(config)` — Read bias matrix, return MoralBiasRecord list + ANOVA summary

### `pipeline/smoke_test.py`

**Sanity check** — < 30 seconds, no API calls, validates schema & export logic.

Run:
```bash
python pipeline/smoke_test.py           # Minimal output
python pipeline/smoke_test.py --verbose # Full traceback on failure
```

---

## Configuration System

**File:** `pipeline/config.yaml`

```yaml
input:  "path/to/document_topic_mapping.csv"
output: "pipeline/outputs"
lang:   "auto"  # Filter by language (zh/en/ja/auto)

stages:
  who:  true   # Enable/disable each stage
  how:  true
  why:  true

who:
  stage:          "full"       # layer12 | llm | full | viz
  spacy_fallback: false        # Lightweight spaCy
  llm_type:       "gemini"     # gemini | openai
  llm_model:      "gemini-2.5-flash-lite"
  concurrency:    50           # Parallel LLM calls
  top_n:          10           # Top-N entities per group

how:
  no_gemini:      false        # Skip Gemini, use rules only
  max_rows:       null         # Debug: process first N rows

why:
  max_docs:       null
  batch_size:     64           # E5 encoding batch
  from_bias:      false        # Skip encoding, start from bias_matrix

export:
  formats:        [jsonl, csv, markdown]
```

**Priority:** CLI args > config.yaml > code defaults

---

## Extending Individual Stages

### Adding a New Processing Module

Suppose you want to add a Stage 4 for "WHAT" (linguistic features):

1. **Create wrapper** in `pipeline/stages/what.py`:

```python
def run(config: dict) -> bool:
    """Call external script"""
    cmd = [sys.executable, "/path/to/rq4_main.py"]
    # Add config-driven arguments
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode == 0

def collect(config: dict) -> list[dict]:
    """Parse output, return standardized records"""
    records = []
    # Read RQ4 output files
    return records
```

2. **Define schema** in `pipeline/schema.py`:

```python
@dataclass
class LinguisticRecord:
    topic: int
    lang: str
    text: str
    features: dict[str, float]
```

3. **Update orchestrator** in `pipeline/orchestrator.py`:

```python
from pipeline.stages.what import run as run_what, collect as collect_what

# In run():
if stages_cfg.get("what", True):
    logger.info("▶ WHAT stage")
    ok = run_what(config)
    if not ok:
        # Error handling
    result.what_results = collect_what(config)
```

4. **Update output schema** in `pipeline/schema.py`:

```python
@dataclass
class PipelineResult:
    what_results: list[dict] = field(default_factory=list)
```

5. **Update export** in `pipeline/export.py`:

Add CSV export for `what_results`:

```python
def export_csv(result, output_dir: Path):
    # ... existing code ...
    if result.what_results:
        p = output_dir / f"{_TS}_what_features.csv"
        pd.DataFrame(result.what_results).to_csv(p, index=False)
```

---

## Development Workflow

### 1. Local Testing

```bash
# Test individual stage
python unsupervised_classification/RQ1/target_extraction_v3.py --help

# Run pipeline with debug flags
python pipeline/run_pipeline.py --no-gemini --max-rows 10

# Test specific stage only
python pipeline/run_pipeline.py --only who --max-rows 50

# Quick smoke test
python pipeline/smoke_test.py --verbose
```

### 2. Adding Logging

Use Python's `logging` module:

```python
import logging
logger = logging.getLogger(__name__)

logger.info("Processing started")
logger.warning("Potential issue")
logger.error("Critical failure")
```

Logs are written to `logs/` directory by `scripts/set_logger.py`.

### 3. Error Handling

Wrap subprocess calls:

```python
try:
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        logger.error(f"Script failed with code {result.returncode}")
        return False
except Exception as e:
    logger.exception(f"Unexpected error: {e}")
    return False
```

---

## Testing

### Unit Tests

```bash
# Run smoke test
python pipeline/smoke_test.py

# Run specific tests
python -m pytest pipeline/tests/ -v
```

### Integration Tests

```bash
# Full pipeline with sample data (can take 10+ minutes)
python pipeline/run_pipeline.py \
    --input sample_data.csv \
    --output test_output/ \
    --max-rows 50
```

### Validation Checklist

- [ ] Input CSV has required columns (text, lang, topic)
- [ ] spaCy models downloaded for all languages
- [ ] .env file has GEMINI_API_KEY
- [ ] Output files generated (JSONL, CSV, Markdown)
- [ ] No errors in `result.errors`
- [ ] Checkpoint files readable

---

## Debugging

### Common Issues

**TypeError in JSON serialization:**
```bash
# Ensure all dict values are JSON-serializable
# Use asdict() from dataclasses
from dataclasses import asdict
record_dict = asdict(record)
```

**subprocess hangs:**
```bash
# Set timeout
result = subprocess.run(cmd, timeout=3600, cwd=PROJECT_ROOT)
```

**Memory leak in E5 encoding (WHY stage):**
```bash
# Reduce batch_size in config.yaml
# or use --from-bias to skip encoding
python pipeline/run_pipeline.py --from-bias
```

### Enable Verbose Logging

```bash
# Set environment variable
export LOGLEVEL=DEBUG
python pipeline/run_pipeline.py
```

---

## Performance Optimization

### Parallel Processing

- **WHO LLM calls**: Use `concurrency` setting in config.yaml
- **HOW SVO extraction**: No parallelization (depends on spaCy)  
- **WHY encoding**: Batch-based (adjust `batch_size` in config.yaml)

### Checkpoint Resumption

If pipeline crashes mid-execution:

```bash
# Resume from HOW (WHO already complete)
python pipeline/run_pipeline.py --skip who
```

Each stage's checkpoint is created incrementally.

### Memory Management

For large datasets (> 100k documents):

1. Reduce spaCy model (use `-sm` models)
2. Lower E5 batch_size from 64 to 16
3. Process in chunks (use `--max-rows` for testing)

---

## CI/CD Integration

Example GitHub Actions workflow (`.github/workflows/pipeline.yml`):

```yaml
name: Pipeline Test

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - run: pip install -r requirements.txt
      - run: python -m spacy download en_core_web_sm
      - run: python -m spacy download zh_core_web_sm
      - run: python -m spacy download ja_core_news_sm
      
      - run: python pipeline/smoke_test.py
      - run: python pipeline/run_pipeline.py --no-gemini --max-rows 50
```

---

## Contributing

1. **Write tests** for new functionality
2. **Update schema** if adding output fields
3. **Document** configuration options in comments
4. **Run smoke_test** before submitting PR
5. **Add logging** for debugging

---

## Code Style

- **Python 3.10+** type hints
- **Docstrings** for all public functions
- **Logging** instead of print()
- **Path** from pathlib, not os.path
- **UTF-8** encoding explicit in file operations

---

## References

- [Python subprocess](https://docs.python.org/3/library/subprocess.html)
- [dataclasses](https://docs.python.org/3/library/dataclasses.html)
- [pathlib](https://docs.python.org/3/library/pathlib.html)
- [logging](https://docs.python.org/3/library/logging.html)

---

*Last updated: April 2026*
