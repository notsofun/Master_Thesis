# 开发者指南

**English version: [DEVELOPER.md](DEVELOPER.md)**

## 架构概述

管线通过 subprocess 编排三个独立的研究阶段：

```
输入 (document_topic_mapping.csv)
        ↓
    [WHO 阶段 — RQ1]  ← unsupervised_classification/RQ1/target_extraction_v3.py
        ↓
    [HOW 阶段 — RQ2] ← unsupervised_classification/RQ2/rq2_pipeline_v2.py
        ↓
    [WHY 阶段 — RQ3] ← unsupervised_classification/RQ3/main.py
        ↓
    [结果聚合与导出]
        ↓
    [JSONL / CSV / Markdown]
```

### 设计原则

1. **非侵入性**：各阶段通过 subprocess 调用原始 RQ 脚本，不修改核心研究逻辑
2. **容错性**：阶段失败时记录错误，默认继续（可配置严格模式立即终止）
3. **可恢复性**：各阶段读写检查点，支持从中断处恢复
4. **模块化**：各 RQ 输出保留在原目录，汇总导出在 pipeline/ 中

---

## 模块结构

### 代码注释

管线中所有 Python 源文件包含中文注释及其英文翻译。关于所有中文注释及其英文等价物的完整参考，请参见 [BILINGUAL_CODE_COMMENTS.md](BILINGUAL_CODE_COMMENTS.md)。

**编写新代码时**，遵循此模式：
```python
# English comment | 中文注释
def new_function():
    """English docstring.
    中文文档字符串。"""
```

### `pipeline/run_pipeline.py`

**CLI 入口点** — 解析命令行参数，加载配置，管理日志。

关键函数：
- `load_config(path)` — 读取 YAML 配置，支持回退解析
- `merge_args_into_config(cfg, args)` — CLI 参数覆盖配置文件
- `build_parser()` — argparse 设置

从项目根目录运行：
```bash
python pipeline/run_pipeline.py [选项]
```

### `pipeline/orchestrator.py`

**编排核心** — 调度 WHO → HOW → WHY 三阶段，收集结果，写入统一 Schema。

关键函数：
- `run(config, strict=False)` — 执行管线，返回 PipelineResult

各阶段流程：
1. 调用 subprocess 运行原始 RQ 脚本
2. 记录成功/失败
3. 将结果收集到统一 Schema
4. 根据 strict 模式决定继续或终止

### `pipeline/schema.py`

**数据 Schema 定义** — 类型安全的 dataclass。

关键类：
- `TopicTargets` — WHO 结果（每话题的实体）
- `ExpressionRecord` — HOW 结果（每文档的修辞框架）
- `MoralBiasRecord` — WHY 结果（每文档的道德轴偏移）
- `PipelineResult` — 聚合结果容器

### `pipeline/export.py`

**导出逻辑** — 将 PipelineResult 转换为 JSONL、CSV、Markdown。

关键函数：
- `export_all(result, output_dir, formats)` — 路由到各格式导出器
- `export_jsonl(result, output_dir)` — 每文档 + 元数据 JSONL
- `export_csv(result, output_dir)` — 四张 CSV 表
- `export_markdown(result, output_dir)` — 论文就绪的报告

### `pipeline/stages/who.py`

**RQ1 包装器** — 通过 subprocess 调用 `unsupervised_classification/RQ1/target_extraction_v3.py`。

关键函数：
- `run(config)` — 执行 RQ1 脚本
- `collect(config)` — 读取检查点，解析实体，返回 TopicTargets 列表

### `pipeline/stages/how.py`

**RQ2 包装器** — 通过 subprocess 调用 `unsupervised_classification/RQ2/rq2_pipeline_v2.py`。

关键函数：
- `run(config)` — 执行 RQ2 脚本
- `collect(config)` — 读取标注 CSV，返回 ExpressionRecord 列表 + 摘要

### `pipeline/stages/why.py`

**RQ3 包装器** — 通过 subprocess 调用 `unsupervised_classification/RQ3/main.py`。

关键函数：
- `run(config)` — 执行 RQ3 脚本
- `collect(config)` — 读取偏移矩阵，返回 MoralBiasRecord 列表 + ANOVA 摘要

### `pipeline/smoke_test.py`

**烟雾测试** — < 30 秒，无 API 调用，验证 Schema 和导出逻辑。

运行：
```bash
python pipeline/smoke_test.py           # 最小输出
python pipeline/smoke_test.py --verbose # 失败时显示完整错误
```

---

## 配置系统

**文件：** `pipeline/config.yaml`

```yaml
input:  "path/to/document_topic_mapping.csv"
output: "pipeline/outputs"
lang:   "auto"  # 按语言过滤 (zh/en/ja/auto)

stages:
  who:  true   # 启用/禁用各阶段
  how:  true
  why:  true

who:
  stage:          "full"       # layer12 | llm | full | viz
  spacy_fallback: false        # 轻量 spaCy
  llm_type:       "gemini"     # gemini | openai
  llm_model:      "gemini-2.5-flash-lite"
  concurrency:    50           # 并行 LLM 调用数
  top_n:          10           # 每组 Top-N 实体

how:
  no_gemini:      false        # 跳过 Gemini，仅规则
  max_rows:       null         # 调试：处理前 N 行

why:
  max_docs:       null
  batch_size:     64           # E5 编码批大小
  from_bias:      false        # 跳过编码，从 bias_matrix 开始

export:
  formats:        [jsonl, csv, markdown]
```

**优先级：** CLI 参数 > config.yaml > 代码默认值

---

## 扩展各阶段

### 添加新的处理模块

假设要添加第 4 阶段"WHAT"（语言特征）：

1. **创建包装器** `pipeline/stages/what.py`：

```python
def run(config: dict) -> bool:
    """调用外部脚本"""
    cmd = [sys.executable, "/path/to/rq4_main.py"]
    # 添加配置驱动的参数
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode == 0

def collect(config: dict) -> list[dict]:
    """解析输出，返回标准化记录"""
    records = []
    # 读取 RQ4 输出文件
    return records
```

2. **定义 Schema** `pipeline/schema.py`：

```python
@dataclass
class LinguisticRecord:
    topic: int
    lang: str
    text: str
    features: dict[str, float]
```

3. **更新编排器** `pipeline/orchestrator.py`：

```python
from pipeline.stages.what import run as run_what, collect as collect_what

# 在 run() 中：
if stages_cfg.get("what", True):
    logger.info("▶ WHAT 阶段")
    ok = run_what(config)
    if not ok:
        # 错误处理
    result.what_results = collect_what(config)
```

4. **更新输出 Schema** `pipeline/schema.py`：

```python
@dataclass
class PipelineResult:
    what_results: list[dict] = field(default_factory=list)
```

5. **更新导出逻辑** `pipeline/export.py`：

添加 `what_results` 的 CSV 导出：

```python
def export_csv(result, output_dir: Path):
    # ... 现有代码 ...
    if result.what_results:
        p = output_dir / f"{_TS}_what_features.csv"
        pd.DataFrame(result.what_results).to_csv(p, index=False)
```

---

## 开发工作流

### 1. 本地测试

```bash
# 测试单个阶段
python unsupervised_classification/RQ1/target_extraction_v3.py --help

# 运行管线，启用调试模式
python pipeline/run_pipeline.py --no-gemini --max-rows 10

# 仅测试特定阶段
python pipeline/run_pipeline.py --only who --max-rows 50

# 快速烟雾测试
python pipeline/smoke_test.py --verbose
```

### 2. 添加日志

使用 Python 标准库 logging：

```python
import logging
logger = logging.getLogger(__name__)

logger.info("处理开始")
logger.warning("潜在问题")
logger.error("关键错误")
```

日志写入 `logs/` 目录（由 `scripts/set_logger.py` 管理）。

### 3. 错误处理

包装 subprocess 调用：

```python
try:
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        logger.error(f"脚本失败，退出码 {result.returncode}")
        return False
except Exception as e:
    logger.exception(f"意外错误：{e}")
    return False
```

---

## 测试

### 单元测试

```bash
# 运行烟雾测试
python pipeline/smoke_test.py

# 运行特定测试
python -m pytest pipeline/tests/ -v
```

### 集成测试

```bash
# 完整管线，使用样本数据（可能耗时 10+ 分钟）
python pipeline/run_pipeline.py \
    --input sample_data.csv \
    --output test_output/ \
    --max-rows 50
```

### 验证清单

- [ ] 输入 CSV 有必需列（text, lang, topic）
- [ ] 所有语言的 spaCy 模型已下载
- [ ] .env 文件包含 GEMINI_API_KEY
- [ ] 输出文件已生成（JSONL, CSV, Markdown）
- [ ] result.errors 为空
- [ ] 检查点文件可读

---

## 调试

### 常见问题

**JSON 序列化 TypeError：**
```bash
# 确保所有 dict 值都能序列化为 JSON
# 使用 dataclasses 的 asdict()
from dataclasses import asdict
record_dict = asdict(record)
```

**subprocess 挂起：**
```bash
# 设置超时
result = subprocess.run(cmd, timeout=3600, cwd=PROJECT_ROOT)
```

**E5 编码内存泄漏（WHY 阶段）：**
```bash
# 减少 config.yaml 中的 batch_size
# 或使用 --from-bias 跳过编码
python pipeline/run_pipeline.py --from-bias
```

### 启用详细日志

```bash
# 设置环境变量
export LOGLEVEL=DEBUG
python pipeline/run_pipeline.py
```

---

## 性能优化

### 并行处理

- **WHO LLM 调用**：使用 config.yaml 中的 `concurrency` 参数
- **HOW SVO 提取**：无并行化（依赖 spaCy）
- **WHY 编码**：基于批处理（调整 config.yaml 中的 `batch_size`）

### 从检查点恢复

若管线中途崩溃：

```bash
# 恢复执行（WHO 已完成，从 HOW 开始）
python pipeline/run_pipeline.py --skip who
```

各阶段的检查点逐步创建。

### 内存管理

处理大数据集（> 10 万文档）：

1. 使用轻量 spaCy 模型（`-sm` 版本）
2. 将 E5 batch_size 从 64 降至 16
3. 分块处理（使用 `--max-rows` 进行测试）

---

## CI/CD 集成

GitHub Actions 示例流程（`.github/workflows/pipeline.yml`）：

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

## 贡献指南

1. **编写测试** for 新功能
2. **更新 Schema** 如是否添加输出字段
3. **文档化** 配置选项在注释中
4. **运行 smoke_test** 在提交 PR 前
5. **添加日志** 便于调试

---

## 代码风格

- **Python 3.10+** 类型注解
- **文档字符串** 对所有公共函数
- **日志记录** 而非 print()
- **pathlib.Path** 而非 os.path
- **文件操作** 明确指定 UTF-8 编码

---

## 参考资源

- [Python subprocess](https://docs.python.org/3/library/subprocess.html)
- [dataclasses](https://docs.python.org/3/library/dataclasses.html)
- [pathlib](https://docs.python.org/3/library/pathlib.html)
- [logging](https://docs.python.org/3/library/logging.html)

---

*最后更新：2026 年 4 月*
