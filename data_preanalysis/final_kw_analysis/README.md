# Final Keyword Analysis

对中文、英文和日文三种语言的最终宗教仇恨言论文本进行关键词分析。

## 功能

- **词频分析**：提取各语言Top N高频词汇
- **共现分析**：基于文档级共现计算PMI和T-score
- **多语言支持**：
  - 中文：使用jieba分词
  - 英文：使用TweetTokenizer
  - 日文：使用janome分词

## 使用方法

### 基础运行
```bash
cd data_preanalysis/final_kw_analysis
python analyze_final.py
```

### 输出文件结构
```
output/
├── zh/                              # 中文结果
│   ├── zh_summary.json              # 摘要信息
│   ├── zh_top_terms.csv             # Top 300词汇（按词频）
│   ├── zh_cooccurrence_pmi.csv      # 共现对（按PMI排序）
│   └── zh_cooccurrence_tscore.csv   # 共现对（按T-score排序）
├── en/                              # 英文结果
│   ├── en_summary.json
│   ├── en_top_terms.csv
│   ├── en_cooccurrence_pmi.csv
│   └── en_cooccurrence_tscore.csv
└── jp/                              # 日文结果
    ├── jp_summary.json
    ├── jp_top_terms.csv
    ├── jp_cooccurrence_pmi.csv
    └── jp_cooccurrence_tscore.csv
```

## 配置参数

在 `analyze_final.py` 顶部修改以下参数：

- `MIN_TERM_FREQ = 5`：最小词频阈值（过滤低频词）
- `MIN_CO = 3`：最小共现次数（避免虚高PMI）
- `TOPK_TERMS = 300`：输出词汇数量
- `TOPK_PAIRS_PER_CORE = 50`：每个核心词的Top K共现词

## 输出文件说明

### 词频文件 (`*_top_terms.csv`)
| 列名 | 说明 |
|-----|------|
| term | 词汇 |
| freq | 出现频次 |

### 共现文件 (`*_cooccurrence_*.csv`)
| 列名 | 说明 |
|-----|------|
| term_a | 词汇A |
| term_b | 词汇B |
| co_doc_count | 共现文档数 |
| df_a | A的文档频次 |
| df_b | B的文档频次 |
| pmi | 点互信息 |
| tscore | T-score统计量 |
| E | 期望共现次数 |
| \*_norm | 归一化分数 |
| score | 综合分数（0.5*pmi_norm + 0.5*tscore_norm） |

### 摘要文件 (`*_summary.json`)
包含：
- 文档总数
- 唯一词汇数
- 共现词对数

## 分析指标说明

### PMI (Pointwise Mutual Information)
衡量两个词汇共现的紧密程度，值越高表示关联越强。
```
PMI = log(P(A,B) / (P(A)*P(B)))
```

### T-score
统计显著性指标，考虑了共现的观察频次与期望频次。
```
T-score = (observed - expected) / sqrt(observed)
```

## 数据源

- **中文**：`data_detect/finetuned_detection/chinese_final_religious_hate.csv`
- **英文**：`data_collection/English_Existing/merged_deduped.csv`
- **日文**：`data_detect/finetuned_detection/japanese_final_religious_hate.csv`

## 参考

- 中文分析：基于 `data_preanalysis/Chinese/analyze.py`
- 日文分析：基于 `data_preanalysis/Japanese/analyze.py`
- 英文分析：基于 `data_preanalysis/English/keywords_extraction.py`
