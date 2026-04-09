# 多语言仇恨言论分析管线

**English version: [README.md](README.md)**

## 项目概览

本项目提供完整的多语言仇恨言论分析管线（英文、中文、日文），通过三个研究问题进行系统分析：

| 研究问题 | 阶段 | 脚本位置 | 核心方法 |
|----------|------|--------|--------|
| **谁** (WHO) — 仇恨言论针对谁？ | RQ1 | `unsupervised_classification/RQ1/` | spaCy NER + 领域词典 + Gemini LLM 兜底 |
| **如何** (HOW) — 如何表达仇恨？ | RQ2 | `unsupervised_classification/RQ2/` | SVO 依存分析 + 谓词窗口 + 10 类修辞框架 |
| **为什么** (WHY) — 动机是什么？ | RQ3 | `unsupervised_classification/RQ3/` | Moral Foundations Dictionary 2.0 + 群际威胁词典 + E5 向量投影 |

---

## 数据集指南

本节说明如何在本项目中查找和使用中日英三语宗教仇恨言论数据集。

### 概览

项目通过三种方式收集多语言仇恨言论数据：
1. **英文**：从现有公开数据集筛选合并（见下文）
2. **中文 & 日文**：从社交媒体平台系统爬取 → 主动学习筛选 → 人工标注微调
3. **LLM增强**：对少数类用大型语言模型生成合成数据以平衡类分布

### 英文数据集

英文仇恨言论数据来自现有注释数据集的筛选合并：
- **HateXplain** ([论文](https://arxiv.org/abs/2012.15606))
- **Jigsaw 多语言有毒评论** (Kaggle)
- 多个女性仇恨 & 仇恨言论数据集（MLMA、Measuring Hate Speech 等）
- **位置**：`data_collection/English_Existing/`

这些数据已标注仇恨标签，我们统一映射到二分类模式。

---

### 中文数据集（构建管线）

#### **第1阶段：原始数据收集** 📥

| 数据源 | 记录数 | 位置 | 爬取脚本 |
|--------|--------|------|--------|
| Tieba（中文论坛） | ~46,640 条 | `data_collection/Tieba/all_search_posts.csv` | `data_collection/Tieba/main.py` |
| HuggingFace（知乎、微博等） | ~19,820 条 | 从以下数据集合并 | [liyucheng/zhihu_26k, vilarin/weibo-2014, m4rque2/weibo_automobile, Giacinta/weibo] |
| **合计原始数据** | **~66,460 条** | — | — |

**复现方法**：编辑 `data_collection/Tieba/main.py` 中的关键词列表，运行爬虫。

#### **第2阶段：预处理 & 清理** 🧹

将 Tieba + HuggingFace 数据合并并清理：
```bash
python scripts/combine.py
```
**输出**：`data_collection/Tieba/final_cleaned_data.csv` (~15,181 条)

#### **第3阶段：主动学习选样** 🎯

采用集成学习投票（LaBSE + XLM-R）+ 置信度阈值，从基座模型最不确定的样本中选出 ~4,000 条：

```bash
python data_detect/run_pipeline.py  # 或
python data_detect/Chinese/run_pipeline.py
```
采用分层抽样策略：
- **共识样本** → 高精度
- **冲突样本** → 模型边界情况

#### **第4阶段：人工标注 & 模型微调** ✅

在 4,000 条标注数据上微调基座检测模型，采用：
- Focal Loss（处理类不平衡）
- 回译增强
- 多任务学习（仇恨言论 + 宗教相关性）

**微调模型**：`model_train/classifier/Chinese/thu_best_multitask_model_back_translated_both_focal_loss.pt`

#### **第5阶段：最终筛选** 🔬

在完整原始数据上应用微调模型进行推理：

```bash
python data_detect/run_pipeline.py --input data_collection/Tieba/final_cleaned_data.csv
```

**输出**：`data_detect/finetuned_detection/chinese_predictions.csv`

**统计数据**（截至 2026 年 3 月）：
- 仇恨言论：**1,687 条** / 15,181 条原始数据（11.11%）
- 处理总量：15,181 条

**下游分析**：使用 `hate_label == 1` 的筛选结果

---

### 日文数据集（构建管线）

#### **第1阶段：原始数据收集** 📥

| 数据源 | 记录数 | 位置 | 爬取脚本 |
|--------|--------|------|--------|
| 5ch（日本 BBS） | ~26,737 条 | `data_collection/5ch/` | `data_collection/5ch/main.py` |
| Common Crawl（网页爬取） | ~38,303 条 | `data_collection/common_crawl/` | `data_collection/common_crawl/special_Ja.py` |
| **合计原始数据** | **~65,040 条** | — | — |

**复现方法**：编辑对应脚本中的关键词列表，运行爬虫。

#### **第2阶段：预处理 & 清理** 🧹

将 5ch + Common Crawl 数据合并并清理：
```bash
python scripts/combine.py
```
**输出**：`data_collection/5ch/raw_religious_ja.csv` (~43,102 条)

#### **第3–5阶段：主动学习 → 微调 → 最终筛选** 🎯✅🔬

流程同中文（见上文）。

**微调模型**：位于 `model_train/classifier/Japanese/`

**最终筛选**：
```bash
python data_detect/run_pipeline.py --input data_collection/5ch/raw_religious_ja.csv
```

**输出**：`data_detect/finetuned_detection/japanese_predictions.csv`

**统计数据**（截至 2026 年 3 月）：
- 宗教相关仇恨言论：**1,610 条** / 43,102 条原始数据（3.74%）
- 处理总量：43,102 条

---

### LLM 增强数据

对于少数类，使用大型语言模型生成合成仇恨言论以实现类平衡：

**位置**：`data_augmentation/` （全部阶段）以及 `data_collection/common_crawl/`

**流程**：
- LLM API 合成 + 冗余检查（见 `data_augmentation/LLM/generated_texts`）
- 回译多语言增强（见 `data_augmentation/back_translation/data`）

---

### 快速查询表：文件位置

| 数据类别 | 语言 | 原始数据 | 清理数据 | 预测结果 |
|----------|------|---------|---------|----------|
| **自然数据** | 中文 | `Tieba/all_search_posts.csv` | `Tieba/final_cleaned_data.csv` | `data_detect/finetuned_detection/chinese_predictions.csv` |
| **自然数据** | 日文 | `5ch/` + `common_crawl/` | `5ch/raw_religious_ja.csv` | `data_detect/finetuned_detection/japanese_predictions.csv` |
| **自然数据** | 英文 | `English_Existing/` (Kaggle、HateXplain 等) | `English_Existing/merged_deduped.csv` | — |
| **增强数据** | 中日文 | — | `data_augmentation/` | — |

---

## 快速开始（用户指南）

### 1. 环境配置

```bash
# 创建并激活虚拟环境
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 下载 spaCy 语言模型
python -m spacy download en_core_web_trf
python -m spacy download zh_core_web_trf
python -m spacy download ja_core_news_trf

# 低显存选项：使用轻量级模型
python -m spacy download en_core_web_sm
python -m spacy download zh_core_web_sm
python -m spacy download ja_core_news_sm
```

### 2. 配置 API 密钥

在项目根目录创建 `.env` 文件：

```bash
GEMINI_API_KEY=你的密钥
# OPENAI_API_KEY=你的密钥  # 可选
```

### 3. 运行管线

准备 CSV 文件，包含列：`text`（文本）、`lang`（语言代码：zh/en/ja）、`topic`（整数话题编号）

```bash
# 完整流程（默认）
python pipeline/run_pipeline.py

# 指定输入/输出
python pipeline/run_pipeline.py --input my_data.csv --output my_output/

# 调试模式（跳过 API，限制行数）
python pipeline/run_pipeline.py --no-gemini --max-rows 100

# 只运行某一阶段
python pipeline/run_pipeline.py --only who   # 仅目标识别
python pipeline/run_pipeline.py --only how   # 仅修辞分析
python pipeline/run_pipeline.py --only why   # 仅動機分析

# 跳过某些阶段（收集已有结果）
python pipeline/run_pipeline.py --skip who how

# 仅重跑可视化（需要已有 checkpoint）
python pipeline/run_pipeline.py --viz-only

# 严格模式（任意失败即终止）
python pipeline/run_pipeline.py --strict
```

---

## 输出格式说明

管线生成多格式结果，保存在 `pipeline/outputs/` 目录：

### 1. **JSONL** 格式 — 机器可读

**`YYYYMMDD_HHMMSS_pipeline_results.jsonl`** （每条文档的三层合并记录）

```json
{
  "text": "基督教正在摧毁我们的文化",
  "topic": 3,
  "lang": "zh",
  "who": ["基督教", "外来者"],
  "how": {
    "predicate": "摧毁",
    "target": "基督教",
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

**`YYYYMMDD_HHMMSS_pipeline_meta.json`** （聚合元数据 & 摘要）

### 2. **CSV** 格式 — 论文附录表格

| 文件 | 内容 |
|------|------|
| `*_who_targets.csv` | 各话题/语言的仇恨对象及频次 |
| `*_how_expressions.csv` | 修辞框架记录（谓词、目标、框架类型） |
| `*_why_bias_matrix.csv` | 每条文档的道德轴偏移分 |
| `*_pipeline_summary.csv` | 跨三层聚合汇总（每行一个话题-语言组合） |

**汇总表列说明：**

| 列名 | 含义 |
|------|------|
| `topic` | 话题编号 |
| `lang` | 语言（zh/en/ja） |
| `who_top_targets` | Top-5 仇恨目标（分号分隔） |
| `how_dominant_frame` | 最主要修辞框架 |
| `how_frame_count` | 此分组的修辞记录总数 |
| `why_Harm_mean` | Harm 轴平均偏移（负值 = 伤害倾向） |
| `why_Sanctity_mean` | Sanctity 轴平均偏移 |
| ... | 其余道德轴 |

### 3. **Markdown** 格式 — 论文就绪报告

**`YYYYMMDD_HHMMSS_report.md`** — 可直接用于学位论文 Results 章节

- WHO 段：按话题/语言列举仇恨对象
- HOW 段：修辞框架频次分布表
- WHY 段：道德轴均值表 + ANOVA 显著性检验

---

## 输入数据格式

CSV 文件必须包含以下列：

| 列名 | 类型 | 示例 |
|------|------|------|
| `text` | 字符串 | "基督徒是社会的毒瘤" |
| `lang` | 字符串 | "zh"（或 "en", "ja"） |
| `topic` | 整数 | 3（-1 表示噪声，自动过滤） |

**完整示例：**
```csv
text,lang,topic
"基督徒是社会的毒瘤",zh,3
"Christians are ruining our culture",en,1
"外国人が我々の文化を破壊している",ja,5
```

*此 CSV 通常由 `unsupervised_classification/bertopic_hate.py` 生成*

---

## 配置文件（`pipeline/config.yaml`）

```yaml
input:  "unsupervised_classification/topic_modeling_results/sixth/data/document_topic_mapping.csv"
output: "pipeline/outputs"
lang:   "auto"  # 或指定 "zh", "en", "ja"

stages:
  who: true    # 各阶段启用/禁用
  how: true
  why: true

who:
  stage:          "full"    # layer12 / llm / full / viz
  spacy_fallback: false     # 低显存时使用轻量 spaCy
  llm_type:       "gemini"
  concurrency:    50        # LLM 并发数
  top_n:          10        # 每组 Top-N 目标

how:
  no_gemini: false
  max_rows:  null           # null = 全部, 或指定数字用于调试

why:
  max_docs:   null
  batch_size: 64            # E5 编码批大小
  from_bias:  false         # 跳过编码，从已有 bias_matrix 开始

export:
  formats: [jsonl, csv, markdown]
```

**参数优先级：** CLI 参数 > config.yaml > 代码默认值

---

## 故障排查

### FileNotFoundError: document_topic_mapping.csv
先运行话题建模：
```bash
python unsupervised_classification/bertopic_hate.py
```

### ModuleNotFoundError
安装依赖：
```bash
pip install -r requirements.txt
```

### GEMINI_API_KEY not found
编辑 `.env`：
```
GEMINI_API_KEY=你的密钥
```
或使用离线模式：
```bash
python pipeline/run_pipeline.py --no-gemini
```

### spaCy 模型未找到
```bash
python -m spacy download zh_core_web_sm  # 或其他模型
python pipeline/run_pipeline.py --spacy-fallback
```

### 显存不足（E5 编码）
在 `config.yaml` 中减少批大小：
```yaml
why:
  batch_size: 16  # 默认 64
```

或跳到已有结果：
```bash
python pipeline/run_pipeline.py --from-bias
```

### 烟雾测试（快速验证，< 30 秒）
验证管线基础逻辑，无需调用 API：
```bash
python pipeline/smoke_test.py
python pipeline/smoke_test.py --verbose  # 失败时显示完整错误
```

---

## 检查点系统（支持断点续传）

所有阶段均有检查点。管线中途崩溃可从最后成功阶段恢复：

| 阶段 | 检查点文件 | 位置 |
|------|-----------|------|
| WHO | checkpoint_layer12.csv | `RQ1/data/` |
| WHO | checkpoint_llm.csv | `RQ1/data/` |
| HOW | rq2_raw_extractions.csv | `RQ2/data/` |
| HOW | rq2_framing_cache.json | `RQ2/data/` |
| WHY | rq3_axis_vectors.npz | `RQ3/data/` |
| WHY | rq3_bias_matrix.csv | `RQ3/data/` |

恢复执行：
```bash
# WHO 已完成，从 HOW 开始
python pipeline/run_pipeline.py --skip who
```

---

## 项目结构

```
.
├── pipeline/                      # 管线编排核心
│   ├── run_pipeline.py           # CLI 入口
│   ├── orchestrator.py           # 任务调度逻辑
│   ├── schema.py                 # 统一输出 Schema
│   ├── export.py                 # JSONL/CSV/Markdown 导出
│   ├── config.yaml               # 配置文件
│   ├── smoke_test.py             # 快速烟雾测试
│   └── stages/
│       ├── who.py                # RQ1 包装器
│       ├── how.py                # RQ2 包装器
│       └── why.py                # RQ3 包装器
│
├── unsupervised_classification/  # 无监督分类（三个 RQ）
│   ├── RQ1/                      # 目标识别
│   ├── RQ2/                      # 修辞分析
│   └── RQ3/                      # 动机分析
│
├── data_collection/              # 数据收集模块
├── data_detect/                  # 仇恨言论检测模型
├── data_preanalysis/             # 预分析工具
├── model_train/                  # 模型训练
├── model_eval/                   # 模型评估
├── scripts/                      # 杂项工具
│
├── requirements.txt
├── README.md                     # 英文主文档
├── README.zh.md                  # 本文件（中文）
├── DEVELOPER.md                  # 开发者指南（英文）
└── DEVELOPER.zh.md               # 开发者指南（中文）
```

---

## 开发者信息

详见 [DEVELOPER.zh.md](DEVELOPER.zh.md)，包含：
- 架构设计和模块说明
- 如何扩展各阶段
- 测试和调试方法
- CI/CD 流程设置

---

*硕士学位论文 — 多语言仇恨言论分析*  
*最后更新：2026 年 4 月*
