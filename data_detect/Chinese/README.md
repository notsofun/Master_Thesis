# 中文仇恨言论检测模型集成

## 概述

已将 Notebook 中提到的三个中文模型（THU-COAI、Davidcliao、Morit）封装成 wrapper，兼容现有的 Pipeline。

## 模型详情

### 1. THU-COAI (RoBERTa-Cold)
- **类型**: 微调 RoBERTa 二分类模型
- **输出**: 0 (非仇恨) / 1 (仇恨)
- **文件**: `data_detect/Chinese/models/thucoai.py`
- **特点**: 快速轻量级，CPU 友好

### 2. Davidcliao (Flair-based)
- **类型**: 基于 Flair 的政治仇恨言论检测
- **输出**: 0 (非仇恨) / 1 (仇恨)
- **文件**: `data_detect/Chinese/models/davidcliao.py`
- **特点**: 针对台湾政治仇恨言论优化
- **需求**: 需手动下载模型文件 `best-model.pt`

### 3. Morit (零样本分类)
- **类型**: XLM-RoBERTa 零样本分类
- **输出**: 0 (正常言论) / 1 (仇恨言论)
- **文件**: `data_detect/Chinese/models/morit.py`
- **特点**: 无需微调，灵活定义分类标签

## 使用方法

### 快速开始

```python
from data_detect.pipeline import HatePipeline
from data_detect.Chinese.constants import ChineseModelName
from data_detect.Chinese.factory import ChineseModelFactory
import logging

logger = logging.getLogger(__name__)

# 指定要使用的模型
models = [ChineseModelName.THUCOAI, ChineseModelName.MORIT]

# 创建 Pipeline
pipeline = HatePipeline(
    logger=logger,
    input_csv="data/input.csv",  # 需包含 'text' 或 'main_content' 列
    models=models,
    model_factory=ChineseModelFactory,
    device="cpu"  # 或 "cuda:0"
)

# 运行检测
result = pipeline.run_detection(total_annotation_n=4000)
```

### 运行脚本

```bash
cd /Users/noriaki/Documents/GitHub/Master_Thesis
python -m data_detect.Chinese.run_pipeline
```

## 配置文件

### `data_detect/Chinese/constants.py`
定义了三个模型的配置信息 (ChineseModelName 枚举)

### `data_detect/Chinese/config.py`
Pipeline 参数配置:
- `DEFAULT_INPUT_CSV`: 输入 CSV 路径
- `OUTPUT_DIR`: 输出目录
- `DEFAULT_POPULATION`: 总体规模 (100000)
- `DEFAULT_MARGIN`: 误差范围 (5%)

## 特殊说明

### Davidcliao 模型需要额外步骤

需要手动下载模型文件：

```bash
# 在项目根目录或 Chinese/ 目录下
wget https://github.com/davidycliao/taiwan-political-hatespeech-detection/raw/main/ch-hs-model/best-model.pt
```

然后修改 `run_pipeline.py` 中的模型列表：

```python
models_to_test = [
    ChineseModelName.THUCOAI, 
    ChineseModelName.DAVIDCLIAO,  # 添加这行
    ChineseModelName.MORIT
]
```

### 输入数据格式

CSV 文件必须包含以下其中之一:
- `text` 列 (英文/日文数据)
- `main_content` 列 (中文数据，如天涯帖子数据)

Pipeline 会自动检测并使用合适的列。

### 输出格式

生成的 annotation CSV 包含:
- `text`: 原始文本
- `model_votes`: 各模型投票结果 (如 "2/0" 表示 2 个投票支持 1 个不支持)
- `average_prob`: 平均概率
- `sampling_strategy_tag`: 采样策略标签

## 集成说明

三个模型都继承自 `BaseModel`，实现了 `score(text) -> int` 方法：

```python
class BaseModel(ABC):
    @abstractmethod
    def score(self, text: str) -> int:
        """返回预测标签: 0 或 1"""
        pass
    
    def predict(self, texts: List[str]) -> List[Dict]:
        """批量预测"""
        ...
```

这确保了与现有 Pipeline 和 Factory 的完全兼容。

## 依赖安装

```bash
# 基础依赖已有
pip install flair transformers

# 如需 GPU 支持
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 问题排查

### 内存不足
使用 `device="cpu"` 或减少 `total_annotation_n` 参数

### 模型下载失败
检查网络连接，或手动下载模型到本地

### 中文编码错误
确保 Python 文件编码为 UTF-8，CSV 文件编码正确
