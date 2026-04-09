# Multilingual Hate Speech Analysis Pipeline

**For detailed documentation in Chinese, see [README.zh.md](README.zh.md)**

## Overview

This project provides a complete pipeline for analyzing hate speech in multilingual contexts (English, Chinese, Japanese). It answers three research questions:

| Research Question | Stage | Script | Key Method |
|---|---|---|---|
| **WHO** — Who is hate speech targeting? | RQ1 | `unsupervised_classification/RQ1/` | spaCy NER + domain dictionary + Gemini LLM fallback |
| **HOW** — How is hate expressed? | RQ2 | `unsupervised_classification/RQ2/` | SVO dependency parsing + predicate window + 10 rhetorical frames |
| **WHY** — What motivates hate speech? | RQ3 | `unsupervised_classification/RQ3/` | Moral Foundations Dictionary 2.0 + threat lexicon + E5 vectors |

---

## Dataset Guide

This section explains where to find and how to use the Chinese, Japanese, and English religious hate speech datasets in this project.

### Overview

The project curates multilingual hate speech data through three approaches:
1. **English**: Aggregated from existing public datasets (see below)
2. **Chinese & Japanese**: Systematically collected from social media platforms, filtered with active learning, and annotated with human review
3. **LLM-augmented**: Synthetic data generated for minority classes to address data imbalance

### English Dataset

English hate speech data was collected by filtering and merging from existing annotated datasets:
- **HateXplain** ([paper](https://arxiv.org/abs/2012.15606))
- **Jigsaw Multilingual Toxic Comments** (Kaggle)
- Multiple Misogyny & Hate Speech datasets (MLMA, Measuring Hate Speech, etc.)
- **Location**: `data_collection/English_Existing/`

These provide pre-annotated hate speech labels, which we map to our binary classification schema.

---

### Chinese Dataset (Construction Pipeline)

#### **Stage 1: Raw Data Collection** 📥

| Source | Records | Location | Pipeline |
|--------|---------|----------|----------|
| Tieba (Chinese forum) | ~46,640 | `data_collection/Tieba/all_search_posts.csv` | `data_collection/Tieba/main.py` |
| HuggingFace (Zhihu, Weibo, etc.) | ~19,820 | Merged in processing | [Datasets: liyucheng/zhihu_26k, vilarin/weibo-2014, m4rque2/weibo_automobile, Giacinta/weibo] |
| **Total raw** | **~66,460** | — | — |

**How to reproduce**: Edit the keyword list in `data_collection/Tieba/main.py` and run the scraper.

#### **Stage 2: Preprocessing & Cleaning** 🧹

After merging Tieba + HuggingFace data:
```bash
python scripts/combine.py
```
**Output**: `data_collection/Tieba/final_cleaned_data.csv` (~15,181 records)

#### **Stage 3: Active Learning Selection** 🎯

Using ensemble voting (LaBSE + XLM-R) + confidence thresholding, we select ~4,000 samples where the baseline model is most uncertain:

```bash
python data_detect/run_pipeline.py  # or
python data_detect/Chinese/run_pipeline.py
```
This creates a high-quality annotation set using hierarchical sampling:
- **Consensus samples** → High precision
- **Conflicting samples** → Model edge cases

#### **Stage 4: Human Annotation & Model Fine-tuning** ✅

Once labeled, we fine-tune the baseline detection model on this 4,000-sample dataset using:
- Focal Loss (to handle class imbalance)
- Back-translation augmentation
- Multi-task learning (hate speech + religious relevance)

**Fine-tuned model**: `model_train/classifier/Chinese/thu_best_multitask_model_back_translated_both_focal_loss.pt`

#### **Stage 5: Final Filtering** 🔬

Use the fine-tuned model on the full raw dataset:

```bash
python data_detect/run_pipeline.py --input data_collection/Tieba/final_cleaned_data.csv
```

**Result**: `data_detect/finetuned_detection/chinese_predictions.csv`

**Statistics** (as of March 2026):
- Religious relevance: **1,687 hate records** out of 15,181 (11.11%)
- Total processed: 15,181 records

**For downstream analysis**: Use the filtered records where `hate_label == 1`

---

### Japanese Dataset (Construction Pipeline)

#### **Stage 1: Raw Data Collection** 📥

| Source | Records | Location | Pipeline |
|--------|---------|----------|----------|
| 5ch (Japanese BBS) | ~26,737 | `data_collection/5ch/` | `data_collection/5ch/main.py` |
| Common Crawl (web scrape) | ~38,303 | `data_collection/common_crawl/` | `data_collection/common_crawl/special_Ja.py` |
| **Total raw** | **~65,040** | — | — |

**How to reproduce**: Edit keyword lists and run the respective pipelines.

#### **Stage 2: Preprocessing & Cleaning** 🧹

After merging 5ch + Common Crawl data:
```bash
python scripts/combine.py
```
**Output**: `data_collection/5ch/raw_religious_ja.csv` (~43,102 records)

#### **Stage 3–5: Active Learning → Fine-tuning → Filtering** 🎯✅🔬

Same process as Chinese (see above).

**Fine-tuned model**: Located in `model_train/classifier/Japanese/`

**Final filtering**:
```bash
python data_detect/run_pipeline.py --input data_collection/5ch/raw_religious_ja.csv
```

**Result**: `data_detect/finetuned_detection/japanese_predictions.csv`

**Statistics** (as of March 2026):
- Religious hate speech: **1,610 records** out of 43,102 (3.74%)
- Total processed: 43,102 records

---

### LLM-Augmented Data

For minority classes, we generate synthetic hate speech using LLMs to achieve better class balance:

**Location**: `data_augmentation/` (all stages) and `data_collection/common_crawl/`

**Process**:
- LLM-based synthesis with redundancy checks (see `data_augmentation/LLM/generated_texts`)
- Back-translation for multilingual augmentation (see `data_augmentation/back_translation/data`)

---

### Quick Reference: File Locations

| Category | Language | Raw Data | Cleaned Data | Predictions |
|----------|----------|----------|--------------|-------------|
| **Natural** | Chinese | `Tieba/all_search_posts.csv` | `Tieba/final_cleaned_data.csv` | `data_detect/finetuned_detection/chinese_predictions.csv` |
| **Natural** | Japanese | `5ch/` + `common_crawl/` | `5ch/raw_religious_ja.csv` | `data_detect/finetuned_detection/japanese_predictions.csv` |
| **Natural** | English | `English_Existing/` (Kaggle, HateXplain, etc.) | `English_Existing/merged_deduped.csv` | — |
| **Augmented** | Chinese/Japanese | — | `data_augmentation/` | — |

---

## Quick Start (Users)

### 1. Environment Setup

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy language models
python -m spacy download en_core_web_trf
python -m spacy download zh_core_web_trf
python -m spacy download ja_core_news_trf

# For low memory: use lightweight models instead
python -m spacy download en_core_web_sm
python -m spacy download zh_core_web_sm
python -m spacy download ja_core_news_sm
```

### 2. Configure API Keys

Create a `.env` file in the project root:

```bash
GEMINI_API_KEY=your_api_key_here
# OPENAI_API_KEY=your_key_here  # Optional
```

### 3. Run the Pipeline

Use your prepared CSV file with columns: `text`, `lang` (zh/en/ja), `topic` (integer)

```bash
# Full workflow (default)
python pipeline/run_pipeline.py

# With custom input/output
python pipeline/run_pipeline.py --input my_data.csv --output my_output/

# Debug mode (skip API calls, limit rows)
python pipeline/run_pipeline.py --no-gemini --max-rows 100

# Run single stage
python pipeline/run_pipeline.py --only who   # Target identification only
python pipeline/run_pipeline.py --only how   # Rhetoric analysis only
python pipeline/run_pipeline.py --only why   # Motivation analysis only

# Skip stages (collect existing results)
python pipeline/run_pipeline.py --skip who how

# Visualization only (requires existing checkpoints)
python pipeline/run_pipeline.py --viz-only

# Strict mode (fail immediately on any error)
python pipeline/run_pipeline.py --strict
```

---

## Output Format

The pipeline generates structured results in multiple formats in `pipeline/outputs/`:

### 1. **JSONL** — Machine-readable Format

**`YYYYMMDD_HHMMSS_pipeline_results.jsonl`** (per-document merged records)

```json
{
  "text": "Christians are destroying our culture",
  "topic": 3,
  "lang": "en",
  "who": ["Christians", "foreigners"],
  "how": {
    "predicate": "destroying",
    "target": "Christians",
    "frame_type": "dehumanization",
    "layer": "svo"
  },
  "why": {
    "Harm": -0.42,
    "Fairness": -0.31,
    "Loyalty": 0.15,
    "Authority": 0.08,
    "Sanctity": -0.38,
    "RealThreat": -0.45,
    "SymThreat": -0.52
  }
}
```

**`YYYYMMDD_HHMMSS_pipeline_meta.json`** (aggregated metadata & summary)

### 2. **CSV** — Appendix Tables

| File | Content |
|------|---------|
| `*_who_targets.csv` | Target entities per topic/language with frequencies |
| `*_how_expressions.csv` | Rhetorical frame records (predicate, target, frame type) |
| `*_why_bias_matrix.csv` | Moral axis displacement per document |
| `*_pipeline_summary.csv` | Cross-stage summary (one row per topic-language combination) |

**Summary table columns:**

| Column | Meaning |
|--------|---------|
| `topic` | Topic ID |
| `lang` | Language (zh/en/ja) |
| `who_top_targets` | Top-5 hate targets (semicolon-separated) |
| `how_dominant_frame` | Most common rhetorical frame |
| `how_frame_count` | Total rhetorical records for this group |
| `why_Harm_mean` | Mean Harm axis bias (negative = harm-focused) |
| `why_Sanctity_mean` | Mean Sanctity axis bias |
| ... | Remaining moral axes |

### 3. **Markdown** — Paper-ready Report

**`YYYYMMDD_HHMMSS_report.md`** — Structured report ready for thesis Results section

- WHO section: Target entities by topic/language
- HOW section: Rhetorical frame frequency tables
- WHY section: Moral axis means + ANOVA significance tests

---

## Input Format

Your CSV must contain:

| Column | Type | Example |
|--------|------|---------|
| `text` | str | "基督徒正在破坏我们的文化" |
| `lang` | str | "zh" (or "en", "ja") |
| `topic` | int | 3 (use -1 for noise, automatically filtered) |

**Example:**
```csv
text,lang,topic
"基督徒是社会的毒瘤",zh,3
"Christians are ruining our culture",en,1
"外国人が我々の文化を破壊している",ja,5
```

*This CSV is typically generated by `unsupervised_classification/bertopic_hate.py`*

---

## Configuration File (`pipeline/config.yaml`)

```yaml
input:  "unsupervised_classification/topic_modeling_results/sixth/data/document_topic_mapping.csv"
output: "pipeline/outputs"
lang:   "auto"  # or "zh", "en", "ja"

stages:
  who: true
  how: true
  why: true

who:
  stage:          "full"    # layer12 / llm / full / viz
  spacy_fallback: false     # Use lightweight spaCy if memory-constrained
  llm_type:       "gemini"
  concurrency:    50        # LLM concurrency
  top_n:          10        # Top-N targets per group

how:
  no_gemini: false
  max_rows:  null           # null = process all, or specify number for debug

why:
  max_docs:   null
  batch_size: 64
  from_bias:  false         # Skip encoding, start from existing bias_matrix

export:
  formats: [jsonl, csv, markdown]
```

---

## Troubleshooting

### FileNotFoundError: document_topic_mapping.csv
Run topic modeling first:
```bash
python unsupervised_classification/bertopic_hate.py
```

### ModuleNotFoundError
Install dependencies:
```bash
pip install -r requirements.txt
```

### GEMINI_API_KEY not found
Add to `.env`:
```
GEMINI_API_KEY=your_key_here
```
Or use offline mode:
```bash
python pipeline/run_pipeline.py --no-gemini
```

### spaCy model not found
```bash
python -m spacy download zh_core_web_sm  # Or desired model
python pipeline/run_pipeline.py --spacy-fallback
```

### Out of memory (spaCy encoding)
Reduce batch size in `config.yaml`:
```yaml
why:
  batch_size: 16  # default 64
```

Or skip to existing results:
```bash
python pipeline/run_pipeline.py --from-bias
```

### Smoke Test (< 30 seconds)
Verify pipeline logic without API calls:
```bash
python pipeline/smoke_test.py
python pipeline/smoke_test.py --verbose  # Show full traceback on failure
```

---

## Checkpoint System (Resumable)

All stages save checkpoints. If the pipeline crashes, resume from the last successful stage:

| Stage | Checkpoint | Location |
|-------|------------|----------|
| WHO | checkpoint_layer12.csv | `RQ1/data/` |
| WHO | checkpoint_llm.csv | `RQ1/data/` |
| HOW | rq2_raw_extractions.csv | `RQ2/data/` |
| HOW | rq2_framing_cache.json | `RQ2/data/` |
| WHY | rq3_axis_vectors.npz | `RQ3/data/` |
| WHY | rq3_bias_matrix.csv | `RQ3/data/` |

Resume:
```bash
# WHO complete, resume from HOW
python pipeline/run_pipeline.py --skip who
```

---

## Project Structure

```
.
├── pipeline/                      # Pipeline orchestration
│   ├── run_pipeline.py           # CLI entry point
│   ├── orchestrator.py           # Core scheduling logic
│   ├── schema.py                 # Unified output schema
│   ├── export.py                 # Export to JSONL/CSV/Markdown
│   ├── config.yaml               # Configuration
│   ├── smoke_test.py             # Quick validation
│   └── stages/
│       ├── who.py                # RQ1 wrapper
│       ├── how.py                # RQ2 wrapper
│       └── why.py                # RQ3 wrapper
│
├── unsupervised_classification/
│   ├── RQ1/                      # Target identification
│   ├── RQ2/                      # Rhetoric analysis
│   └── RQ3/                      # Motivation analysis
│
├── data_collection/              # Data collection modules
├── data_detect/                  # Hate detection models
├── data_preanalysis/             # Pre-analysis tools
├── model_train/                  # Model training
├── model_eval/                   # Model evaluation
├── scripts/                      # Utilities
│
├── requirements.txt
├── README.md                     # This file (English)
├── README.zh.md                  # Chinese version
└── DEVELOPER.md                  # Developer guide (English)
```

---

## For Developers

See [DEVELOPER.md](DEVELOPER.md) for:
- Architecture and design decisions
- How to extend each stage
- Testing and debugging
- CI/CD pipeline setup

See [README.zh.md](README.zh.md) for complete Chinese documentation.

---

*Master's Thesis — Multilingual Hate Speech Analysis*  
*Last updated: April 2026*