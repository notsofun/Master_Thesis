# Thesis Analysis Pipeline

**最小可复现分析管线（Minimal Reproducible Pipeline）**  
将 WHO → HOW → WHY 三阶段仇恨言论分析封装为一条命令。

---

## 概览

本管线编排以下三个研究问题的分析脚本：

| 阶段 | 研究问题 | 脚本 | 核心方法 |
|------|----------|------|----------|
| **WHO** | 仇恨言论针对谁？ | `RQ1/target_extraction_v3.py` | spaCy NER + 领域词典 + Gemini LLM 兜底 |
| **HOW** | 如何表达仇恨？ | `RQ2/rq2_pipeline_v2.py` | SVO依存分析 + 谓词窗口 + 10类修辞框架 |
| **WHY** | 动机是什么？ | `RQ3/main.py` | MFD 2.0 + 群际威胁词典 + E5 向量投影 |

管线不修改任何核心研究逻辑，仅做编排、收集与格式化导出。

---

## 环境安装

```bash
# 1. 从仓库根目录安装依赖
pip install -r requirements.txt

# 2. 配置 API 密钥（Gemini / OpenAI）
cp .env.example .env        # 如有模板
# 或直接编辑 .env，填入：
#   GEMINI_API_KEY=your_key_here
#   OPENAI_API_KEY=your_key_here（可选）

# 3. 安装 spaCy 语言模型
python -m spacy download en_core_web_trf
python -m spacy download zh_core_web_trf
python -m spacy download ja_core_news_trf
# 如显存不足，改用轻量模型：
python -m spacy download en_core_web_sm
python -m spacy download zh_core_web_sm
python -m spacy download ja_core_news_sm
```

---

## 快速开始

所有命令均从**仓库根目录**运行。

### 完整流程（推荐）

```bash
python pipeline/run_pipeline.py
```

使用 `pipeline/config.yaml` 中的默认配置，输出到 `pipeline/outputs/`。

### 指定输入/输出

```bash
python pipeline/run_pipeline.py \
    --input unsupervised_classification/topic_modeling_results/sixth/data/document_topic_mapping.csv \
    --output pipeline/outputs/
```

### 调试模式（不调 API，限制行数，快速验证）

```bash
python pipeline/run_pipeline.py --no-gemini --max-rows 100 --max-docs 200
```

### 只跑某一阶段

```bash
python pipeline/run_pipeline.py --only who   # 仅运行 WHO (RQ1)
python pipeline/run_pipeline.py --only how   # 仅运行 HOW (RQ2)
python pipeline/run_pipeline.py --only why   # 仅运行 WHY (RQ3)
```

### 跳过某些阶段（收集已有 checkpoint）

```bash
# 只跑 WHY，WHO/HOW 从已有结果收集
python pipeline/run_pipeline.py --skip who how
```

### 只重跑可视化（不重新调 API）

```bash
python pipeline/run_pipeline.py --viz-only
```

### 严格模式（任意阶段失败即终止）

```bash
python pipeline/run_pipeline.py --strict
```

### 烟雾测试（验证管线基础逻辑，< 30 秒）

```bash
python pipeline/smoke_test.py
python pipeline/smoke_test.py --verbose  # 失败时显示详细 traceback
```

---

## 命令行参数一览

```
python pipeline/run_pipeline.py [选项]

基础参数：
  --input PATH        输入 CSV（含 text/lang/topic 列）
  --output DIR        输出目录（默认 pipeline/outputs/）
  --lang {zh,en,ja,auto}  语言过滤（默认 auto = 不过滤）
  --config PATH       配置文件（默认 pipeline/config.yaml）

阶段控制（互斥）：
  --only {who,how,why}   只运行指定阶段
  --skip STAGE [...]     跳过指定阶段（可多选）

通用开关：
  --viz-only          只重跑可视化（各 RQ 须已有 checkpoint）
  --no-gemini         跳过 Gemini，用规则分类器（离线/调试）
  --strict            任意阶段失败即终止

WHO (RQ1) 参数：
  --who-stage {layer12,llm,viz,full}  RQ1 运行阶段（默认 full）
  --spacy-fallback    使用轻量 spaCy（显存不足时）

HOW (RQ2) 参数：
  --max-rows N        调试：HOW 阶段仅处理前 N 条文档

WHY (RQ3) 参数：
  --max-docs N        调试：WHY 阶段仅处理前 N 条文档
  --from-bias         WHY 跳过编码，从已有 bias_matrix.csv 开始
```

---

## 输入格式

输入文件为 CSV，必须包含以下列：

| 列名 | 类型 | 说明 |
|------|------|------|
| `text` | str | 原始文本 |
| `lang` | str | 语言代码（`zh` / `en` / `ja`） |
| `topic` | int | BERTopic 话题编号（`-1` = 噪声，自动过滤） |

示例行：
```csv
text,lang,topic
"基督徒是社会的毒瘤",zh,3
"Christians are ruining our culture",en,1
"外国人が我々の文化を破壊している",ja,5
```

> 该文件由 `unsupervised_classification/bertopic_hate.py` 生成，
> 位于 `topic_modeling_results/sixth/data/document_topic_mapping.csv`。

---

## 输出格式说明

所有输出写入 `pipeline/outputs/`（或 `--output` 指定目录），文件名含时间戳前缀。

### JSONL — 机器可读（每行一个文档）

**`YYYYMMDD_HHMMSS_pipeline_results.jsonl`**

每行结构：
```json
{
  "text": "原始文本（前200字）",
  "topic": 3,
  "lang": "zh",
  "who": ["基督徒", "外来者"],
  "how": {
    "predicate": "渗透",
    "target": "基督徒",
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

**`YYYYMMDD_HHMMSS_pipeline_meta.json`**

元信息 + 各阶段聚合摘要，包含：
- 运行时间戳、输入路径、已运行阶段
- 文档总数、语言分布、话题数量
- WHO 各话题目标实体
- HOW 修辞框架频次
- WHY 道德轴均值 + ANOVA 结果

### CSV — 论文附录（各阶段分表）

| 文件 | 内容 |
|------|------|
| `*_who_targets.csv` | topic / lang / target / count |
| `*_how_expressions.csv` | topic / lang / text / predicate / frame_type / layer |
| `*_why_bias_matrix.csv` | topic / lang / text / Harm / Fairness / … |
| `*_pipeline_summary.csv` | 跨三层聚合（每行一个 topic-lang 组合） |

`pipeline_summary.csv` 列说明：

| 列名 | 说明 |
|------|------|
| `topic` | 话题编号 |
| `lang` | 语言 |
| `who_top_targets` | Top-5 仇恨目标（分号分隔） |
| `how_dominant_frame` | 最主要的修辞框架 |
| `how_frame_count` | 该话题-语言组合的修辞记录总数 |
| `why_Harm_mean` | Harm 轴平均偏移（负值 → 偏向伤害）|
| `why_Sanctity_mean` | Sanctity 轴平均偏移 |
| … | 其余5个轴同上 |

### Markdown — 论文可直接引用的报告

**`YYYYMMDD_HHMMSS_report.md`**

结构对应论文 Results 章节：
- WHO 节：按话题-语言列举 Top 实体
- HOW 节：修辞框架频次分布表 + 聚合摘要
- WHY 节：道德轴均值表 + ANOVA 显著性

### 可视化（各 RQ 原生输出，不在管线汇总目录）

| 位置 | 文件 |
|------|------|
| `RQ1/visualizations/` | 话题-语言目标分布图（PNG） |
| `RQ2/data/` | `rq2_A_topic_frame_heatmap.html` 等 5 张交互图 |
| `RQ3/data/` | `rq3_A_topic_axis_heatmap.html` 等 6 张交互图 |

---

## 中间 Checkpoint（断点续传）

各阶段均有 checkpoint，支持在任意步骤中断后从断点重跑：

| 阶段 | Checkpoint 文件 | 说明 |
|------|----------------|------|
| WHO | `RQ1/data/checkpoint_layer12.csv` | spaCy+词典层结果 |
| WHO | `RQ1/data/checkpoint_llm.csv` | +LLM 兜底结果 |
| HOW | `RQ2/data/rq2_raw_extractions.csv` | 原始 SVO 提取 |
| HOW | `RQ2/data/rq2_framing_labeled.csv` | +框架分类标签 |
| HOW | `RQ2/data/rq2_framing_cache.json` | Gemini 调用缓存 |
| WHY | `RQ3/data/rq3_axis_vectors.npz` | 道德轴向量 |
| WHY | `RQ3/data/rq3_bias_matrix.csv` | 文档偏移矩阵 |

---

## 常见错误排查

### `FileNotFoundError: document_topic_mapping.csv`
BERTopic 结果文件不存在，需先运行话题建模：
```bash
python unsupervised_classification/bertopic_hate.py
```

### `ModuleNotFoundError: No module named 'spacy'`
安装依赖：
```bash
pip install -r requirements.txt
```

### `GEMINI_API_KEY not found`
在 `.env` 文件中添加：
```
GEMINI_API_KEY=your_key_here
```
或使用离线模式跳过 API 调用：
```bash
python pipeline/run_pipeline.py --no-gemini
```

### WHO 阶段：spaCy 模型未下载
```bash
python -m spacy download zh_core_web_trf  # 或 zh_core_web_sm（轻量）
# 然后用 --spacy-fallback 指定轻量模型
python pipeline/run_pipeline.py --spacy-fallback
```

### WHY 阶段：显存不足 / 编码太慢
减小批大小（在 config.yaml 中设置）：
```yaml
why:
  batch_size: 16
```
或从已有 bias_matrix 直接开始：
```bash
python pipeline/run_pipeline.py --from-bias
```

### 管线中途崩溃
各阶段均有 checkpoint，可用 `--skip` 跳过已完成阶段：
```bash
# WHO 已完成，从 HOW 开始
python pipeline/run_pipeline.py --skip who
```

### 验证配置正确性
```bash
python pipeline/smoke_test.py --verbose
```

---

## 配置文件（`pipeline/config.yaml`）

```yaml
input:  "unsupervised_classification/topic_modeling_results/sixth/data/document_topic_mapping.csv"
output: "pipeline/outputs"
lang:   "auto"

stages:
  who: true
  how: true
  why: true

who:
  stage:          "full"    # layer12 / llm / full
  spacy_fallback: false
  llm_type:       "gemini"
  concurrency:    50
  top_n:          10

how:
  no_gemini: false
  max_rows:  null

why:
  max_docs:   null
  batch_size: 64
  from_bias:  false

export:
  formats: [jsonl, csv, markdown]
```

---

## 文件结构

```
pipeline/
├── run_pipeline.py     # CLI 入口（此文件）
├── config.yaml         # 主配置文件
├── orchestrator.py     # 管线编排逻辑
├── schema.py           # 统一输出数据结构
├── export.py           # JSONL / CSV / Markdown 导出
├── smoke_test.py       # 烟雾测试（< 30 秒，不调 API）
├── stages/
│   ├── __init__.py
│   ├── who.py          # RQ1 封装（目标识别）
│   ├── how.py          # RQ2 封装（修辞策略）
│   └── why.py          # RQ3 封装（道德动机）
└── outputs/            # 所有导出结果（自动创建）
    ├── *.jsonl
    ├── *.json
    ├── *.csv
    └── *.md
```

---

*Zhidian Huang · Master's Thesis · 2026*
