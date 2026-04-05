"""
RQ3 词典加载模块 — dict_loader.py
===================================
将三个官方词典文件 + 一个自定义扩充文件解析为统一格式，
供 main.py 的 Step1 (build_axis_vectors) 使用。

支持的文件格式：
  1. mfd2.0.dic      — 英文 MFD 2.0，LIWC .dic 格式
                       类别：care/fairness/loyalty/authority/sanctity × virtue/vice
  2. J-MFD_2018r1.dic — 日文 J-MFD，LIWC .dic 格式（带 * 通配符词条）
                       类别：HarmVirtue/HarmVice/Fairness.../Ingroup.../Authority.../Purity...
  3. cmfd_civictech.csv — 中文 CMFD，CSV 格式（无 polarity 列，按「负向词汇特征」规则推断）
  4. intergroup_threat_custom.csv — 自定义群际威胁轴（中英日），CSV 格式（含 polarity 列）

输出格式（供 main.py 消费）：
  {
    "harm":             {"pos": [...英文词...], "neg": [...英文词...]},
    "fairness":         {"pos": [...], "neg": [...]},
    "loyalty":          {"pos": [...], "neg": [...]},
    "authority":        {"pos": [...], "neg": [...]},
    "sanctity":         {"pos": [...], "neg": [...]},
    "realistic_threat": {"pos": [...], "neg": [...]},
    "symbolic_threat":  {"pos": [...], "neg": [...]},
  }
  其中 pos = virtue/safety/cohesion 极，neg = vice/threat/degradation 极。

MFD 轴映射（MFD 原始名 → RQ3 轴名）：
  care      → harm         (care.virtue=pos, care.vice=neg)
  fairness  → fairness
  loyalty   → loyalty      (MFD 叫 loyalty/ingroup，J-MFD 叫 Ingroup)
  authority → authority
  sanctity  → sanctity     (J-MFD 叫 Purity)
  realistic_threat / symbolic_threat → 来自 intergroup_threat_custom.csv（无官方词典）

CMFD 特殊说明：
  CMFD 的 foundation 列没有 polarity，混合了 virtue 和 vice。
  本模块使用「前缀规则」做软分类：
    不以「不/非/无/反/失/错/恶/劣」开头 → pos（virtue）
    以上述否定前缀开头，或明确负面词 → neg（vice）
  这是近似处理。官方 CMFD 项目本身也没有严格 virtue/vice 区分。
  实际影响：CMFD 中文词被加入轴质心计算，偏差方向与英日对齐。

作者: Zhidian  |  日期: 2026-04
"""

import re
import logging
from pathlib import Path
from collections import defaultdict

import pandas as pd

log = logging.getLogger(__name__)

# ── 词典文件路径（相对于本文件所在目录） ─────────────────────────────────────
_THIS_DIR = Path(__file__).resolve().parent
DICT_DIR  = _THIS_DIR / "dictionaries"

MFD2_PATH        = DICT_DIR / "mfd2.0.dic"
JMFD_PATH        = DICT_DIR / "J-MFD_2018r1.dic"
CMFD_PATH        = DICT_DIR / "cmfd_civictech.csv"
THREAT_PATH      = DICT_DIR / "intergroup_threat_custom.csv"

# ── MFD 轴名映射（官方名 → RQ3 轴名） ────────────────────────────────────────
# MFD 2.0 英文
_MFD2_AXIS_MAP = {
    "care":      "harm",
    "fairness":  "fairness",
    "loyalty":   "loyalty",
    "authority": "authority",
    "sanctity":  "sanctity",
}

# J-MFD 日文（类别名 → RQ3 轴名 + virtue/vice）
_JMFD_AXIS_MAP = {
    "HarmVirtue":      ("harm",      "pos"),
    "HarmVice":        ("harm",      "neg"),
    "FairnessVirtue":  ("fairness",  "pos"),
    "FairnessVice":    ("fairness",  "neg"),
    "IngroupVirtue":   ("loyalty",   "pos"),
    "IngroupVice":     ("loyalty",   "neg"),
    "AuthorityVirtue": ("authority", "pos"),
    "AuthorityVice":   ("authority", "neg"),
    "PurityVirtue":    ("sanctity",  "pos"),
    "PurityVice":      ("sanctity",  "neg"),
    # MoralityGeneral 无明确极性，跳过
}

# CMFD 基础名 → RQ3 轴名（CMFD 无 virtue/vice，用规则推断）
_CMFD_AXIS_MAP = {
    "care":  "harm",
    "fair":  "fairness",
    "loya":  "loyalty",
    "auth":  "authority",
    "sanc":  "sanctity",
    # altr/dili/resi/mode/wast/libe/general 超出 MFD 五基础，跳过
}

# CMFD 负向词汇前缀规则（用于推断 vice）
_CMFD_VICE_PREFIXES = (
    "不", "非", "无", "反", "失", "错", "恶", "劣", "坏", "假", "伪",
    "歹", "邪", "凶", "残", "毒", "腐", "丑", "贪", "乱",
)
# 高频明确负面词（直接判 vice，不依赖前缀）
_CMFD_EXPLICIT_VICE = {
    "罪", "罪恶", "罪行", "犯罪", "杀", "杀害", "杀人", "杀戮",
    "伤害", "虐待", "欺骗", "欺诈", "腐败", "腐化", "堕落",
    "叛国", "叛逆", "背叛", "反叛", "作乱", "出卖",
    "混乱", "动乱", "动荡", "暴乱", "暴力", "恐怖",
    "污秽", "下流", "淫乱", "猥亵", "色情",
    "毒害", "毒品", "中毒", "寄生", "寄生虫",
}


# ═══════════════════════════════════════════════════════════════════════════════
# 解析器：各文件格式
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_liwc_dic(path: Path) -> dict[str, str]:
    """
    解析 LIWC .dic 格式文件。

    格式：
      %
      1<TAB>category_name
      2<TAB>category_name
      ...
      %
      word<TAB>cat_id1<TAB>cat_id2...
      ...

    返回：{word: [cat_name1, cat_name2, ...]}
    日文词典词条可能带 * 通配符（匹配任意后缀），保留原始词条（调用方决定是否去通配）。
    """
    content = path.read_text(encoding="utf-8", errors="replace")
    parts = content.split("%")
    if len(parts) < 3:
        log.warning(f"[DICT] {path.name}: 无法找到 %...% 类别区，跳过")
        return {}

    # 解析类别定义
    cat_map: dict[str, str] = {}
    for line in parts[1].strip().split("\n"):
        line = line.strip()
        if "\t" in line:
            num, name = line.split("\t", 1)
            cat_map[num.strip()] = name.strip()

    # 解析词条
    word_cats: dict[str, list[str]] = defaultdict(list)
    for line in parts[2].strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        tokens = line.split("\t")
        word = tokens[0].strip()
        if not word:
            continue
        for cat_id in tokens[1:]:
            cat_id = cat_id.strip()
            if cat_id in cat_map:
                word_cats[word].append(cat_map[cat_id])

    log.debug(f"[DICT] {path.name}: 解析 {len(word_cats)} 个词条，{len(cat_map)} 个类别")
    return dict(word_cats)


def _cmfd_infer_polarity(word: str) -> str:
    """
    对 CMFD 中文词汇推断极性（virtue=pos / vice=neg）。
    规则：
      1. 在明确负面词集合里 → vice
      2. 以否定前缀开头 → vice
      3. 其余 → virtue（保守默认）
    """
    if word in _CMFD_EXPLICIT_VICE:
        return "neg"
    for prefix in _CMFD_VICE_PREFIXES:
        if word.startswith(prefix) and len(word) > len(prefix):
            return "neg"
    return "pos"


# ═══════════════════════════════════════════════════════════════════════════════
# 主加载函数
# ═══════════════════════════════════════════════════════════════════════════════

def load_moral_axes() -> dict[str, dict[str, list[str]]]:
    """
    加载并合并所有词典文件，返回 7 个道德轴的词汇列表。

    合并策略：
      - 优先保留官方词典词汇
      - 自定义扩充词汇（intergroup_threat_custom.csv）追加到末尾
      - 同一轴同一极性内去重（保持顺序，官方先进入）

    返回格式：
      {
        "harm":      {"pos": [词...], "neg": [词...]},
        "fairness":  {"pos": [...],   "neg": [...]},
        ...（共 7 个轴）
      }
    """
    # 初始化 7 个轴的词汇集合（用 list + seen_set 保持顺序且去重）
    axis_keys = [
        "harm", "fairness", "loyalty", "authority", "sanctity",
        "realistic_threat", "symbolic_threat",
    ]
    result: dict[str, dict[str, list[str]]] = {
        k: {"pos": [], "neg": []} for k in axis_keys
    }
    seen: dict[str, dict[str, set[str]]] = {
        k: {"pos": set(), "neg": set()} for k in axis_keys
    }

    def _add(axis: str, polarity: str, word: str):
        """去重后添加词汇。"""
        w = word.strip()
        if not w or len(w) < 2:
            return
        # 去除通配符 * 后作为去重键
        key = w.rstrip("*").strip()
        if not key or key in seen[axis][polarity]:
            return
        seen[axis][polarity].add(key)
        result[axis][polarity].append(w)   # 保留原始形式（带*）供显示，E5编码时再去*

    # ── 1. MFD 2.0 英文 ───────────────────────────────────────────────────────
    if MFD2_PATH.exists():
        word_cats = _parse_liwc_dic(MFD2_PATH)
        mfd_added = 0
        for word, cat_names in word_cats.items():
            for cat_name in cat_names:
                if "." not in cat_name:
                    continue
                axis_raw, polarity_raw = cat_name.rsplit(".", 1)
                axis = _MFD2_AXIS_MAP.get(axis_raw)
                polarity = "pos" if polarity_raw == "virtue" else "neg"
                if axis:
                    _add(axis, polarity, word)
                    mfd_added += 1
        log.info(f"[DICT] MFD 2.0 英文: 加载 {mfd_added} 条词汇映射")
    else:
        log.warning(f"[DICT] 未找到 MFD 2.0 词典: {MFD2_PATH}")

    # ── 2. J-MFD 日文 ─────────────────────────────────────────────────────────
    if JMFD_PATH.exists():
        word_cats = _parse_liwc_dic(JMFD_PATH)
        jmfd_added = 0
        for word, cat_names in word_cats.items():
            for cat_name in cat_names:
                mapping = _JMFD_AXIS_MAP.get(cat_name)
                if mapping:
                    axis, polarity = mapping
                    _add(axis, polarity, word)
                    jmfd_added += 1
        log.info(f"[DICT] J-MFD 日文: 加载 {jmfd_added} 条词汇映射")
    else:
        log.warning(f"[DICT] 未找到 J-MFD 词典: {JMFD_PATH}")

    # ── 3. CMFD 中文 ──────────────────────────────────────────────────────────
    if CMFD_PATH.exists():
        cmfd_df = pd.read_csv(CMFD_PATH)
        cmfd_added = 0
        # 列名兼容：chinese 或 word
        word_col = "chinese" if "chinese" in cmfd_df.columns else "word"
        for _, row in cmfd_df.iterrows():
            word = str(row[word_col]).strip()
            foundation = str(row["foundation"]).strip()
            axis = _CMFD_AXIS_MAP.get(foundation)
            if not axis:
                continue
            polarity = _cmfd_infer_polarity(word)
            _add(axis, polarity, word)
            cmfd_added += 1
        log.info(f"[DICT] CMFD 中文: 加载 {cmfd_added} 条词汇（极性由规则推断）")
    else:
        log.warning(f"[DICT] 未找到 CMFD 词典: {CMFD_PATH}")

    # ── 4. 自定义群际威胁轴（中英日） ─────────────────────────────────────────
    if THREAT_PATH.exists():
        threat_df = pd.read_csv(THREAT_PATH)
        threat_added = 0
        for _, row in threat_df.iterrows():
            word       = str(row["word"]).strip()
            foundation = str(row["foundation"]).strip()
            polarity   = "pos" if str(row["polarity"]).strip() == "virtue" else "neg"
            if foundation in axis_keys:
                _add(foundation, polarity, word)
                threat_added += 1
        log.info(f"[DICT] 群际威胁自定义词典: 加载 {threat_added} 条")
    else:
        log.warning(f"[DICT] 未找到自定义词典: {THREAT_PATH}")

    # ── 汇总日志 ──────────────────────────────────────────────────────────────
    log.info("[DICT] ✅ 词典合并完成：")
    for axis in axis_keys:
        n_pos = len(result[axis]["pos"])
        n_neg = len(result[axis]["neg"])
        log.info(f"       [{axis:20s}] pos={n_pos:4d}词  neg={n_neg:4d}词")

    return result


def get_axis_words(axis_key: str, polarity: str,
                   axes: dict | None = None) -> list[str]:
    """
    获取指定轴的词汇列表，自动去除 LIWC 通配符 * 后缀。
    polarity: 'pos' 或 'neg'
    """
    if axes is None:
        axes = load_moral_axes()
    words_raw = axes.get(axis_key, {}).get(polarity, [])
    # 去除 * 通配符（E5 编码时需要干净词汇）
    return [w.rstrip("*").strip() for w in words_raw if w.rstrip("*").strip()]


def print_axes_summary(axes: dict):
    """终端打印词典摘要（调试用）。"""
    print("\n" + "=" * 60)
    print("道德轴词典摘要")
    print("=" * 60)
    for axis, pols in axes.items():
        print(f"\n[{axis}]")
        for pol, words in pols.items():
            sample = [w.rstrip("*") for w in words[:5]]
            print(f"  {pol} ({len(words)}词): {sample} ...")
    print("=" * 60)


# ── 单独运行时输出摘要 ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    axes = load_moral_axes()
    print_axes_summary(axes)
