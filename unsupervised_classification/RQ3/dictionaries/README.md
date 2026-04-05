# RQ3 词典目录

## 使用的词典及下载说明

### 1. MFD2.0.csv（需手动下载）
- **来源**: Frimer et al. (2019), OSF: https://osf.io/xakyw/
- **格式要求**: 第一列 `word`，第二列 `category`（如 `harm.virtue`、`harm.vice`、`fairness.virtue` 等）
- **或者**: 使用 FrameAxis 项目的整理版 https://github.com/negar-mokhberian/Moral_Foundation_FrameAxis/blob/master/MFD2.0.csv
  - 格式: `word, foundation, polarity`（polarity = virtue/vice）
- **涵盖的基础**: harm, fairness, loyalty(ingroup), authority, sanctity(purity) × virtue/vice

### 2. j-mfd.csv（需手动下载，可选）
- **来源**: 日本語 MFD, Matsuo et al. (2019) PLOS ONE
  - GitHub: https://github.com/soramame0518/j-mfd
- **格式要求**: `word, foundation, polarity`
- **说明**: 如不下载，代码会跳过日文官方词典，仅用 `intergroup_threat_custom.csv` 中的日文词汇

### 3. CMFD.csv（需手动下载，可选）
- **来源**: Chinese Moral Foundations Dictionary
  - GitHub: https://github.com/CivicTechLab/CMFD
- **格式要求**: `word, foundation, polarity`
- **说明**: 如不下载，代码会跳过中文官方词典

### 4. intergroup_threat_custom.csv（已内置）
- **来源**: 本研究自定义（无官方词典）
- **内容**: 基于 Intergroup Threat Theory (Stephan & Renfro, 2002) 的现实威胁轴和象征威胁轴
- **语言**: 中英日三语对齐
- **格式**: `word, foundation, polarity, lang, note`

## 词典融合策略

代码加载顺序：
1. 官方英文 MFD 2.0（harm/fairness/loyalty/authority/sanctity × virtue/vice）
2. 官方 J-MFD（日文，如存在）
3. 官方 CMFD（中文，如存在）
4. 自定义扩充词汇（`intergroup_threat_custom.csv`，含 realistic_threat 和 symbolic_threat 两个新轴）

**去重原则**: 同一词在同一轴同一极性下只保留一条（官方优先）
