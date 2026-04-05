"""
python unsupervised_classification/RQ3/main.py
python unsupervised_classification/RQ3/main.py --viz-only
python unsupervised_classification/RQ3/main.py --max-docs 500

RQ3 分析管线 — 词典轴道德动机投影 (FrameAxis)
==============================================

研究问题：为什么中英日三种语言针对同类宗教目标，攻击动机却截然不同？

方法论：Lexicon-Augmented Vector Projection（词典增强向量投影法）
─────────────────────────────────────────────────────────────────
用权威词典（MFD 2.0 + 群际威胁词典）的两极词汇构建"理论轴"，
将文档的 E5 多语言向量投影到这些轴上，计算语义偏移（Bias），
无需 LLM，100% 数学可解释。

词典来源（由 dict_loader.py 统一加载）：
  1. mfd2.0.dic        — 英文 MFD 2.0 (Frimer et al., 2019)，2104 词条
  2. J-MFD_2018r1.dic  — 日文 J-MFD (Matsuo et al., 2019)，725 词条
  3. cmfd_civictech.csv — 中文 CMFD (CivicTechLab)，6138 词条
  4. intergroup_threat_custom.csv — 自定义群际威胁轴，131 词条

理论轴（MFD 五基础 + 群际威胁理论两轴）：
  1. Harm      ← Care (关爱) vs. Harm (伤害/虐待)
  2. Fairness  ← Fairness (公平) vs. Cheating (欺骗/剥削)
  3. Loyalty   ← Loyalty (忠诚/爱国) vs. Betrayal (背叛/渗透)
  4. Authority ← Authority (权威/法律) vs. Subversion (颠覆/叛乱)
  5. Sanctity  ← Sanctity (神圣/纯洁) vs. Degradation (堕落/病理化)
  6. RealThreat ← Safety (安全/稳定) vs. Realistic Threat (资源抢夺/政治控制)
  7. SymThreat  ← Cultural Cohesion (文化凝聚) vs. Symbolic Threat (文化入侵/洗脑)

计算流程：
  Axis_k      = Centroid(E5(D_k+)) − Centroid(E5(D_k−))
  Bias(d, k)  = cos_sim(E5(d), Axis_k)

正值 → 文本语义偏向正极（关爱、公平、忠诚…）
负值 → 文本语义偏向负极（伤害、欺骗、背叛…）→ 攻击性道德动机

管线设计（解耦，每步输出 checkpoint，任意步可独立重跑）：
  Step 1: 构建道德轴（词典词汇 → E5 向量 → 轴向量）
          checkpoint: rq3_axis_vectors.npz
  Step 2: 文档向量投影（E5 编码文档 → Bias 矩阵）
          checkpoint: rq3_bias_matrix.csv
  Step 3: ANOVA 统计检验（三语言在各道德轴的差异显著性）
          checkpoint: rq3_anova_results.csv
  Step 4: 聚合可视化（5 张图表 + 1 张网络图）
          output: rq3_A_topic_axis_heatmap.html   — Topic×Axis 热力图
                  rq3_B_lang_axis_bar.html         — 三语言×轴均值条形图（核心）
                  rq3_C_lang_radar.html            — 语言道德动机雷达图
                  rq3_D_topic_lang_scatter.html    — 关键轴散点图
                  rq3_E_topic_network.html         — Topic 共现网络（新增）
                  rq3_F_lang_bias_violin.html      — 语言×轴 Bias 小提琴图（新增）
                  rq3_summary.csv

日志：
  - 使用项目统一的 scripts/set_logger.py → setup_logging()
  - 自动在脚本同级 logs/ 目录创建日志文件

运行模式：
  完整运行           python unsupervised_classification/RQ3/main.py
  跳过轴构建         python unsupervised_classification/RQ3/main.py --from-bias             # 已有 bias_matrix.csv
  只重跑可视化       python unsupervised_classification/RQ3/main.py --viz-only              # 已有 bias_matrix.csv
  调试（小数据）     python unsupervised_classification/RQ3/main.py --max-docs 200

作者: Zhidian  |  日期: 2026-04
"""

# ── 标准库 ────────────────────────────────────────────────────────────────────
import argparse
import os
import sys
import warnings
from pathlib import Path

# ── 第三方库 ──────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

# ── 路径配置 ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
THIS_DIR     = Path(__file__).resolve().parent
DATA_DIR     = THIS_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 输入数据路径
DOC_PATH = (
    PROJECT_ROOT
    / "unsupervised_classification"
    / "topic_modeling_results" / "sixth" / "data"
    / "document_topic_mapping.csv"
)

# 输出文件路径（checkpoint & 最终产物）
CKPT_AXIS   = DATA_DIR / "rq3_axis_vectors.npz"
CKPT_BIAS   = DATA_DIR / "rq3_bias_matrix.csv"
ANOVA_CSV   = DATA_DIR / "rq3_anova_results.csv"
SUMMARY_CSV = DATA_DIR / "rq3_summary.csv"
TOPIC_SUMMARY_CSV = DATA_DIR / "rq3_topic_summary.csv"

# ── 日志：使用项目统一的 set_logger ──────────────────────────────────────────
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.set_logger import setup_logging

load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)

# 模块级 logger 占位（由 __main__ 初始化后替换）
import logging as _logging
log: _logging.Logger = _logging.getLogger(__name__)

# ── 词典加载器（从文件读取，替代硬编码） ──────────────────────────────────────
from dict_loader import load_moral_axes, get_axis_words


# ═══════════════════════════════════════════════════════════════════════════════
# 一、轴元数据（标签、描述）— 词汇本身由 dict_loader 动态加载
# ═══════════════════════════════════════════════════════════════════════════════

# 轴元数据（标签仅用于图表显示，词汇来自 dict_loader.py）
AXIS_LABELS: dict[str, dict[str, str]] = {
    "harm":             {"en": "Harm",            "zh": "伤害轴",     "desc": "Care(+) vs. Harm(−)"},
    "fairness":         {"en": "Fairness",         "zh": "公平轴",     "desc": "Fairness(+) vs. Cheating(−)"},
    "loyalty":          {"en": "Loyalty",          "zh": "忠诚轴",     "desc": "Loyalty(+) vs. Betrayal(−)"},
    "authority":        {"en": "Authority",        "zh": "权威轴",     "desc": "Authority(+) vs. Subversion(−)"},
    "sanctity":         {"en": "Sanctity",         "zh": "圣洁轴",     "desc": "Sanctity(+) vs. Degradation(−)"},
    "realistic_threat": {"en": "Realistic Threat", "zh": "现实威胁轴", "desc": "Safety(+) vs. Realistic Threat(−)"},
    "symbolic_threat":  {"en": "Symbolic Threat",  "zh": "象征威胁轴", "desc": "Cultural Cohesion(+) vs. Symbolic Threat(−)"},
}

# 轴名列表（保持顺序，与可视化顺序一致）
AXIS_KEYS = list(AXIS_LABELS.keys())

# 快捷访问映射（用于图表轴标签）
AXIS_LABEL_EN   = {k: v["en"]   for k, v in AXIS_LABELS.items()}
AXIS_LABEL_ZH   = {k: v["zh"]   for k, v in AXIS_LABELS.items()}
AXIS_LABEL_DESC = {k: v["desc"] for k, v in AXIS_LABELS.items()}

# Topic 英文标签（来自 datamap_plot.py）
TOPIC_LABELS: dict[int, str] = {
    -1: "Inclusion and Reform Challenges",
     0: "Separation of Church & State / Japan",
     1: "Christians and Trump's Politics",
     2: "Women and Ordination",
     3: "LGBTQ Issues and Youth Alienation",
     4: "Religion, Sexuality, and Violence",
     5: "Catholic-Lutheran Communion Disputes",
     6: "Church Abuse Victims",
     7: "Religion and Natural Disasters",
     8: "Pro-Life and Women's Autonomy",
     9: "Divine-Human Nature / Jesus",
    10: "Conflicts and Reflections on the Bible",
    11: "2017 Pakistan Religious Violence",
    12: "Communion for Divorced and Remarried",
    13: "Religion in Public vs. Catholic Schools",
    14: "Public Prayer and Religious Freedom",
    15: "Authenticity and Faith in Catholicism",
    16: "Canadian Immigration and Cultural Disputes",
    17: "Pagan-Christian Conflicts",
    18: "Internal Discussions in Catholicism",
    19: "Doctrines of Religion and Salvation",
    20: "Nuns and Religious Culture",
    21: "Adventist Church",
    22: "Social Observations and Religious Critique",
    23: "Historical Critiques of Christian Violence",
    24: "Reforms and Divisions in Catholicism",
    25: "Social Division and Racial Antagonism",
    26: "Priests and Complexities of Faith",
    27: "Bakery and Religious Freedom Legal Conflicts",
    28: "Burke vs. Pope Francis",
    29: "Christian Principles and Criticisms",
    30: "Christian Development / Denominational Differences",
    31: "Antisemitism and Religious Persecution",
    32: "Church Doctrine and Sacramental Validity",
    33: "Healthcare Rights and Institutions",
    34: "Church-State Relations and Charities",
    35: "The Poor, Wealth, and Relief",
    36: "Internal Divisions in Catholicism",
    37: "Religious Beliefs and Social Phenomena",
    38: "Opposition and Impact of Religious Beliefs",
    39: "Religious Controversies and Social Critiques",
    40: "Critique of Money and Religious Manipulation",
    41: "Catholic Stereotypes and Theology",
    42: "Faith and Heresy Debates",
    43: "Missionaries and Confucianism",
    44: "Pope Francis: Conflicts and Mercy",
    45: "Life and Responsibility",
    46: "Religious Beliefs and Social Controversies",
}


LANG_LABEL = {"en": "English 🇬🇧", "zh": "中文 🇨🇳", "jp": "日本語 🇯🇵"}
LANG_COLOR = {"en": "#4C78A8", "zh": "#F58518",  "jp": "#54A24B"}
AXIS_COLOR = {
    "harm": "#E45756",
    "fairness": "#F58518",
    "loyalty": "#72B7B2",
    "authority": "#4C78A8",
    "sanctity": "#54A24B",
    "realistic_threat": "#B279A2",
    "symbolic_threat": "#FF9DA6",
}

# 中日文字体注入（与 RQ2 完全一致）
_FONT_CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;700&family=Noto+Sans+JP:wght@400;700&display=swap');
  body, .plotly-graph-div, .gtitle, .xtitle, .ytitle, text {
    font-family: 'Noto Sans SC', 'Noto Sans JP', 'Hiragino Sans', 'Microsoft YaHei',
                 'Meiryo', Arial, sans-serif !important;
  }
</style>
"""

_LAYOUT_BASE = dict(
    font=dict(
        family="'Noto Sans SC','Noto Sans JP','Hiragino Sans','Microsoft YaHei',Arial,sans-serif",
        size=12,
    ),
    paper_bgcolor="white",
    plot_bgcolor="white",
)


# ═══════════════════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════════════════

def inject_font(html_path: Path):
    """向已生成的 HTML 文件注入中日文字体 CSS（与 RQ2 完全一致）。"""
    content = html_path.read_text(encoding="utf-8")
    if "Noto Sans SC" not in content:
        content = content.replace("<head>", f"<head>{_FONT_CSS}", 1)
        html_path.write_text(content, encoding="utf-8")
        log.debug(f"[FONT] 字体注入完成: {html_path.name}")


def axis_bilingual_label(key: str) -> str:
    """返回双语轴标签，格式：'English Label<br><sup>(中文)</sup>'"""
    en = AXIS_LABEL_EN.get(key, key)
    zh = AXIS_LABEL_ZH.get(key, "")
    desc = AXIS_LABEL_DESC.get(key, "")
    if zh:
        return f"{en}<br><sup>({zh})</sup>"
    return en


def normalize_lang(lang: str) -> str:
    """规范化语言代码：jp → ja，与 RQ1 保持一致。"""
    return "ja" if str(lang).lower() in ("ja", "jp") else str(lang).lower()


def load_encoder():
    """
    懒加载 multilingual-e5-large 编码器。
    返回 (model, tokenizer) 或在不可用时抛出 ImportError。
    """
    try:
        from sentence_transformers import SentenceTransformer
        log.info("[E5] 加载 multilingual-e5-large 编码器...")
        model = SentenceTransformer("intfloat/multilingual-e5-large")
        log.info("[E5] 编码器加载完成")
        return model
    except ImportError:
        log.error("[E5] 需要安装 sentence-transformers: pip install sentence-transformers")
        raise
    except Exception as e:
        log.error(f"[E5] 编码器加载失败: {e}")
        raise


def encode_texts(model, texts: list[str], batch_size: int = 64,
                 desc: str = "编码") -> np.ndarray:
    """
    使用 SentenceTransformer 批量编码文本，带进度条。
    multilingual-e5-large 建议加 'query: ' 前缀用于文档编码。
    """
    # e5 模型: passage 前缀用于文档，query 前缀用于检索
    prefixed = [f"passage: {t}" for t in texts]
    all_embs = []
    for i in tqdm(range(0, len(prefixed), batch_size), desc=desc, unit="batch"):
        batch = prefixed[i: i + batch_size]
        embs = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        all_embs.append(embs)
    return np.vstack(all_embs)


def cosine_sim_batch(doc_vecs: np.ndarray, axis_vec: np.ndarray) -> np.ndarray:
    """
    计算文档向量组 doc_vecs (N×D) 与单个轴向量 axis_vec (D,) 的余弦相似度。
    由于 encode_texts 已 normalize_embeddings=True，doc_vecs 已单位化，
    axis_vec 也在 Step1 中单位化，所以直接点积即可。
    """
    axis_norm = axis_vec / (np.linalg.norm(axis_vec) + 1e-12)
    return doc_vecs @ axis_norm  # shape: (N,)


# ═══════════════════════════════════════════════════════════════════════════════
# Step 1: 构建道德轴向量
# ═══════════════════════════════════════════════════════════════════════════════

def build_axis_vectors(model) -> dict[str, np.ndarray]:
    """
    对词典每个轴，分别编码正极/负极词汇，
    计算质心差值 axis_vec = centroid(pos) − centroid(neg)，
    返回 {axis_key: axis_vector (D,)} 的字典。

    轴向量的方向含义：
      正方向 → 关爱、公平、忠诚……（道德正极）
      负方向 → 伤害、欺骗、背叛……（攻击动机）

    Bias = cos_sim(doc, axis_vec)：
      越负 → 文本在该道德维度上越偏向攻击性/负向动机
    """
    log.info("[STEP1] 开始构建道德轴向量...")
    log.info(f"[STEP1] 共 {len(AXIS_KEYS)} 个轴: {AXIS_KEYS}")

    # 从文件词典加载各轴词汇（MFD 2.0 + J-MFD + CMFD + 自定义威胁词典）
    moral_axes_vocab = load_moral_axes()

    axis_vectors: dict[str, np.ndarray] = {}

    for key in tqdm(AXIS_KEYS, desc="构建轴向量", unit="轴"):
        pos_words = get_axis_words(key, "pos", moral_axes_vocab)
        neg_words = get_axis_words(key, "neg", moral_axes_vocab)

        if not pos_words or not neg_words:
            log.warning(f"[STEP1]   轴 [{key}] 词汇为空（pos={len(pos_words)}, neg={len(neg_words)}），跳过")
            continue

        log.info(f"[STEP1]   轴 [{key}]: 正极 {len(pos_words)} 词, 负极 {len(neg_words)} 词")

        # 编码正极词汇（用 query 前缀，因为是短词汇）
        pos_prefixed = [f"query: {w}" for w in pos_words]
        neg_prefixed = [f"query: {w}" for w in neg_words]

        pos_embs = model.encode(pos_prefixed, normalize_embeddings=True,
                                show_progress_bar=False)
        neg_embs = model.encode(neg_prefixed, normalize_embeddings=True,
                                show_progress_bar=False)

        # 计算质心
        pos_centroid = pos_embs.mean(axis=0)
        neg_centroid = neg_embs.mean(axis=0)

        # 轴向量 = 正极质心 − 负极质心
        axis_vec = pos_centroid - neg_centroid

        # 记录轴向量模长（越大说明两极语义分离越清晰）
        norm = np.linalg.norm(axis_vec)
        log.info(f"[STEP1]   轴 [{key}] 向量模长: {norm:.4f}")

        axis_vectors[key] = axis_vec

    # 保存 checkpoint
    np.savez(CKPT_AXIS, **axis_vectors)
    log.info(f"[STEP1] ✅ 轴向量已保存: {CKPT_AXIS}")

    return axis_vectors


def load_axis_vectors() -> dict[str, np.ndarray]:
    """从 checkpoint 加载轴向量。"""
    data = np.load(CKPT_AXIS)
    axis_vectors = {k: data[k] for k in data.files}
    log.info(f"[STEP1] ✅ 从 checkpoint 加载轴向量: {list(axis_vectors.keys())}")
    return axis_vectors


# ═══════════════════════════════════════════════════════════════════════════════
# Step 2: 文档投影 → Bias 矩阵
# ═══════════════════════════════════════════════════════════════════════════════

def compute_bias_matrix(
    doc_df: pd.DataFrame,
    axis_vectors: dict[str, np.ndarray],
    model,
    batch_size: int = 64,
) -> pd.DataFrame:
    """
    对所有文档编码后，计算每个文档在每个道德轴上的 Bias 值。

    返回 DataFrame，列为：
      doc_id, topic, lang, text(截断), bias_{axis_key} × 7

    Bias 计算：
      Bias(doc, axis_k) = cos_sim(E5(doc), axis_vec_k)
    其中 axis_vec_k 已单位化（在此函数内完成），E5(doc) 已 L2 归一化。
    """
    log.info(f"[STEP2] 开始文档投影，共 {len(doc_df)} 条文档...")

    texts = doc_df["text"].astype(str).tolist()

    # 批量编码文档（带进度条）
    doc_embs = encode_texts(model, texts, batch_size=batch_size,
                             desc="编码文档 (E5)")
    log.info(f"[STEP2] 文档编码完成，矩阵形状: {doc_embs.shape}")

    # 对每个轴计算 Bias
    bias_cols: dict[str, np.ndarray] = {}
    for key in tqdm(AXIS_KEYS, desc="投影到道德轴", unit="轴"):
        axis_vec = axis_vectors[key]
        bias_vals = cosine_sim_batch(doc_embs, axis_vec)
        bias_cols[f"bias_{key}"] = bias_vals
        log.info(
            f"[STEP2]   轴 [{key}]: "
            f"均值={bias_vals.mean():.4f}, "
            f"std={bias_vals.std():.4f}, "
            f"min={bias_vals.min():.4f}, "
            f"max={bias_vals.max():.4f}"
        )

    # 组装结果 DataFrame
    result_df = doc_df[["topic", "lang"]].copy().reset_index(drop=True)
    result_df.insert(0, "doc_id", range(len(result_df)))
    result_df["text_preview"] = (
        doc_df["text"].astype(str).str[:80].reset_index(drop=True)
    )
    for key, vals in bias_cols.items():
        result_df[key] = vals.round(6)

    result_df.to_csv(CKPT_BIAS, index=False, encoding="utf-8-sig")
    log.info(f"[STEP2] ✅ Bias 矩阵已保存: {CKPT_BIAS} ({len(result_df)} 行)")

    return result_df


def load_bias_matrix() -> pd.DataFrame:
    """从 checkpoint 加载 Bias 矩阵。"""
    df = pd.read_csv(CKPT_BIAS)
    log.info(f"[STEP2] ✅ 从 checkpoint 加载 Bias 矩阵: {len(df)} 行")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Step 3: ANOVA 统计检验
# ═══════════════════════════════════════════════════════════════════════════════

def run_anova(bias_df: pd.DataFrame) -> pd.DataFrame:
    """
    对每个道德轴，检验中英日三语言的 Bias 均值差异是否显著（单因素 ANOVA）。

    附加 effect size (eta-squared, η²) 和 Tukey HSD 两两比较（Bonferroni 校正）。

    返回结果 DataFrame，列为：
      axis, F_stat, p_value, eta_squared, significant,
      mean_en, mean_zh, mean_jp,
      std_en, std_zh, std_jp,
      n_en, n_zh, n_jp,
      tukey_en_zh_p, tukey_en_jp_p, tukey_zh_jp_p
    """
    log.info("[STEP3] ANOVA 统计检验开始...")
    try:
        from scipy import stats as scipy_stats
    except ImportError as e:
        log.error("[STEP3] 需要 scipy 才能运行 ANOVA：pip install scipy")
        raise ImportError("scipy is required for run_anova()") from e

    # 语言代码规范化
    bias_df = bias_df.copy()
    bias_df["lang"] = bias_df["lang"].apply(normalize_lang)

    # 标准化语言码（RQ2 用 jp，RQ1 用 ja，此处统一）
    lang_map = {"ja": "jp"}
    bias_df["lang"] = bias_df["lang"].replace(lang_map)

    results = []
    for key in tqdm(AXIS_KEYS, desc="ANOVA检验", unit="轴"):
        col = f"bias_{key}"
        if col not in bias_df.columns:
            log.warning(f"[STEP3] 列 {col} 不存在，跳过")
            continue

        groups: dict[str, np.ndarray] = {}
        for lang in ["en", "zh", "jp"]:
            mask = bias_df["lang"] == lang
            vals = bias_df.loc[mask, col].dropna().values
            groups[lang] = vals

        # 跳过样本量不足的轴
        if any(len(g) < 10 for g in groups.values()):
            log.warning(
                f"[STEP3] 轴 [{key}] 某语言样本不足10，跳过: "
                f"en={len(groups['en'])}, zh={len(groups['zh'])}, jp={len(groups['jp'])}"
            )
            continue

        # 单因素 ANOVA
        f_stat, p_val = scipy_stats.f_oneway(
            groups["en"], groups["zh"], groups["jp"]
        )

        # Effect size: η² = SS_between / SS_total
        all_vals = np.concatenate(list(groups.values()))
        grand_mean = all_vals.mean()
        ss_between = sum(
            len(g) * (g.mean() - grand_mean) ** 2 for g in groups.values()
        )
        ss_total = sum((v - grand_mean) ** 2 for v in all_vals)
        eta_sq = ss_between / ss_total if ss_total > 0 else 0.0

        # Tukey HSD（两两比较，Bonferroni 校正 alpha=0.05/3）
        alpha_bonf = 0.05 / 3
        tukey_pairs: dict[str, float] = {}
        for (la, lb) in [("en", "zh"), ("en", "jp"), ("zh", "jp")]:
            _, p_pair = scipy_stats.ttest_ind(
                groups[la], groups[lb], equal_var=False
            )
            tukey_pairs[f"tukey_{la}_{lb}_p"] = round(float(p_pair), 6)

        row = {
            "axis":        key,
            "axis_label":  AXIS_LABEL_EN[key],
            "axis_desc":   AXIS_LABEL_DESC[key],
            "F_stat":      round(float(f_stat), 4),
            "p_value":     round(float(p_val), 6),
            "eta_squared": round(float(eta_sq), 4),
            "significant": "***" if p_val < 0.001 else
                           "**"  if p_val < 0.01  else
                           "*"   if p_val < 0.05  else "ns",
        }
        # 各语言均值 / 标准差 / 样本量
        for lang in ["en", "zh", "jp"]:
            row[f"mean_{lang}"] = round(float(groups[lang].mean()), 5)
            row[f"std_{lang}"]  = round(float(groups[lang].std()),  5)
            row[f"n_{lang}"]    = int(len(groups[lang]))
        row.update(tukey_pairs)
        results.append(row)

        log.info(
            f"[STEP3]   [{key}] F={f_stat:.2f}, p={p_val:.4e}, "
            f"η²={eta_sq:.4f}, sig={row['significant']} | "
            f"mean: en={row['mean_en']:.4f}, zh={row['mean_zh']:.4f}, jp={row['mean_jp']:.4f}"
        )

    anova_df = pd.DataFrame(results)
    anova_df.to_csv(ANOVA_CSV, index=False, encoding="utf-8-sig")
    log.info(f"[STEP3] ✅ ANOVA 结果已保存: {ANOVA_CSV}")

    # 终端摘要
    sig_axes = anova_df[anova_df["significant"] != "ns"]["axis"].tolist()
    log.info(f"[STEP3] 显著差异轴 (p<0.05): {sig_axes}")

    return anova_df


# ═══════════════════════════════════════════════════════════════════════════════
# Step 4: 聚合可视化
# ═══════════════════════════════════════════════════════════════════════════════

def _safe_zscore(series: pd.Series) -> pd.Series:
    """按组 z-score，避免标准差为 0 时产生 NaN。"""
    std = float(series.std(ddof=0))
    if std < 1e-12 or np.isnan(std):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.mean()) / std


def _prepare_topic_language_profiles(bias_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    生成两层数据：
      1. 文档级 bias_df（附带各语言内部 z-score）
      2. (lang, topic) 聚合后的 topic-language profile

    这里显式把“跨语言绝对均值”与“语言内部相对偏移”分开，
    避免直接拿原始余弦值比较时，把 embedding 语言基线误读成社会学差异。
    """
    df = bias_df.copy()
    df["lang"] = df["lang"].apply(normalize_lang).replace({"ja": "jp"})

    bias_cols = [f"bias_{k}" for k in AXIS_KEYS]
    z_cols = []
    for key in AXIS_KEYS:
        col = f"bias_{key}"
        z_col = f"z_{key}"
        df[z_col] = df.groupby("lang")[col].transform(_safe_zscore)
        z_cols.append(z_col)

    agg_spec = {"n_docs": ("topic", "size")}
    for key in AXIS_KEYS:
        agg_spec[f"bias_{key}"] = (f"bias_{key}", "mean")
        agg_spec[f"z_{key}"] = (f"z_{key}", "mean")

    topic_lang_df = (
        df[df["topic"] >= 0]
        .groupby(["lang", "topic"], as_index=False)
        .agg(**agg_spec)
    )
    topic_lang_df["topic_label"] = topic_lang_df["topic"].map(TOPIC_LABELS).fillna(
        topic_lang_df["topic"].map(lambda t: f"Topic {t}")
    )
    topic_lang_df["topic_short"] = topic_lang_df["topic_label"].map(
        lambda s: s if len(s) <= 28 else s[:28] + "…"
    )
    topic_lang_df["row_label"] = topic_lang_df.apply(
        lambda r: f"{r['lang'].upper()} · T{int(r['topic'])} · {r['topic_short']}",
        axis=1,
    )

    z_matrix_cols = [f"z_{k}" for k in AXIS_KEYS]
    topic_lang_df["dominant_negative_axis"] = (
        topic_lang_df[z_matrix_cols].idxmin(axis=1).str.replace("z_", "", regex=False)
    )
    topic_lang_df["dominant_negative_z"] = topic_lang_df[z_matrix_cols].min(axis=1)
    topic_lang_df["dominant_negative_axis_label"] = topic_lang_df["dominant_negative_axis"].map(
        AXIS_LABEL_EN
    )
    topic_lang_df["dominant_negative_raw_bias"] = topic_lang_df.apply(
        lambda r: r[f"bias_{r['dominant_negative_axis']}"], axis=1
    )
    return df, topic_lang_df


def _compute_pca_projection(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """用 SVD 计算二维 PCA 投影，不额外依赖 sklearn。"""
    x = np.asarray(matrix, dtype=float)
    x = x - x.mean(axis=0, keepdims=True)
    if x.ndim != 2 or x.shape[0] == 0:
        return np.zeros((0, 2)), np.zeros(2), np.zeros((x.shape[1] if x.ndim == 2 else 0, 2))
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    comps = min(2, vt.shape[0])
    scores = u[:, :comps] * s[:comps]
    if comps < 2:
        scores = np.pad(scores, ((0, 0), (0, 2 - comps)))
    denom = max(x.shape[0] - 1, 1)
    eigvals = (s ** 2) / denom
    total = eigvals.sum() if eigvals.size else 1.0
    explained = np.zeros(2)
    explained[: min(2, eigvals.size)] = eigvals[:2] / total
    loadings = vt[:2].T if vt.shape[0] >= 2 else np.pad(vt[:1].T, ((0, 0), (0, 1)))
    return scores[:, :2], explained, loadings


def _build_topic_exemplar_chart(topic_lang_df: pd.DataFrame):
    """
    图E: 每种语言中，哪些 topic 最明显地被某个“负向动机轴”主导。
    用语言内部 z-score 的最小值做排序，更适合论文叙事。
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    plot_df = topic_lang_df[topic_lang_df["n_docs"] >= 5].copy()
    if plot_df.empty:
        log.warning("[STEP4] 图E 跳过：topic-language 聚合结果为空")
        return

    fig_e = make_subplots(
        rows=1, cols=3,
        subplot_titles=[LANG_LABEL["en"], LANG_LABEL["zh"], LANG_LABEL["jp"]],
        horizontal_spacing=0.06,
    )

    for col_idx, lang in enumerate(["en", "zh", "jp"], start=1):
        sub = (
            plot_df[plot_df["lang"] == lang]
            .sort_values(["dominant_negative_z", "n_docs"], ascending=[True, False])
            .head(10)
            .sort_values("dominant_negative_z", ascending=False)
        )
        if sub.empty:
            continue

        colors = [AXIS_COLOR.get(a, "#999999") for a in sub["dominant_negative_axis"]]
        hover = []
        for _, row in sub.iterrows():
            axis_bits = "<br>".join(
                f"{AXIS_LABEL_EN[key]}: raw={row[f'bias_{key}']:+.3f}, z={row[f'z_{key}']:+.2f}"
                for key in AXIS_KEYS
            )
            hover.append(
                f"<b>{row['row_label']}</b><br>"
                f"文档数: {int(row['n_docs'])}<br>"
                f"主导负向动机: {row['dominant_negative_axis_label']}<br>"
                f"相对强度 z: {row['dominant_negative_z']:+.2f}<br><br>"
                f"{axis_bits}"
            )

        fig_e.add_trace(
            go.Bar(
                x=(-sub["dominant_negative_z"]).values,
                y=sub["row_label"],
                orientation="h",
                marker_color=colors,
                customdata=np.array(hover, dtype=object),
                hovertemplate="%{customdata}<extra></extra>",
                showlegend=False,
            ),
            row=1, col=col_idx,
        )
        fig_e.update_xaxes(title="Relative Attack Strength (-z)", row=1, col=col_idx)
        fig_e.update_yaxes(automargin=True, row=1, col=col_idx)

    fig_e.update_layout(
        **_LAYOUT_BASE,
        title=dict(
            text=(
                "Fig E: Topic Exemplars by Dominant Negative Motive<br>"
                "<sup>各语言中最典型的 topic-language 单元；长度越长表示其主导负向动机在本语言内部越突出</sup>"
            ),
            font=dict(size=13),
        ),
        height=620,
    )
    p = DATA_DIR / "rq3_E_topic_network.html"
    fig_e.write_html(str(p))
    inject_font(p)
    log.info(f"[STEP4] ✅ 图E: {p.name}")

    # 图E2 保留：Topic 的语言构成占比
    topic_lang = (
        topic_lang_df.groupby(["topic", "lang"], as_index=False)["n_docs"].sum()
    )
    topic_lang["pct"] = (
        topic_lang["n_docs"] / topic_lang.groupby("topic")["n_docs"].transform("sum") * 100
    ).round(1)
    all_topics_sorted = sorted(topic_lang["topic"].unique())
    lang_pct_pivot = (
        topic_lang.pivot(index="topic", columns="lang", values="pct")
        .fillna(0)
        .reindex(all_topics_sorted)
    )

    fig_e2 = go.Figure()
    for lang in ["en", "zh", "jp"]:
        if lang not in lang_pct_pivot.columns:
            continue
        fig_e2.add_trace(go.Bar(
            name=LANG_LABEL.get(lang, lang),
            x=[f"T{t}: {TOPIC_LABELS.get(t, '')[:18]}" for t in lang_pct_pivot.index],
            y=lang_pct_pivot[lang].values,
            marker_color=LANG_COLOR[lang],
            hovertemplate=f"{LANG_LABEL.get(lang, lang)}<br>%{{x}}<br>占比: %{{y:.1f}}%<extra></extra>",
        ))

    fig_e2.update_layout(
        **_LAYOUT_BASE,
        barmode="stack",
        title=dict(
            text=(
                "Fig E2: Language Composition per Topic (stacked %)<br>"
                "<sup>哪些话题是跨语言共享的，哪些更接近单语言主导</sup>"
            ),
            font=dict(size=13),
        ),
        xaxis=dict(title="Topic", tickangle=-45, tickfont=dict(size=9)),
        yaxis=dict(title="语言占比 (%)"),
        legend=dict(title="Language"),
        height=500,
    )
    p2 = DATA_DIR / "rq3_E2_topic_lang_composition.html"
    fig_e2.write_html(str(p2))
    inject_font(p2)
    log.info(f"[STEP4] ✅ 图E2: {p2.name}")


def _build_violin_plot(bias_df: pd.DataFrame, anova_df: pd.DataFrame | None):
    """
    图F: 语言 × 轴 Bias 分布小提琴图（By-语言分析）

    每个子图对应一个道德轴，展示三语言在该轴 Bias 的完整分布形态。
    相比均值条形图，小提琴图能看出分布的偏态、双峰、离群点。
    显著性标注来自 ANOVA 结果（同 图B）。
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    bias_df = bias_df.copy()
    bias_df["lang"] = bias_df["lang"].apply(normalize_lang).replace({"ja": "jp"})

    n_axes = len(AXIS_KEYS)
    n_cols = min(3, n_axes)
    n_rows = (n_axes + n_cols - 1) // n_cols

    # 构建子图标题
    subplot_titles = []
    for key in AXIS_KEYS:
        sig = ""
        if anova_df is not None and not anova_df.empty:
            row = anova_df[anova_df["axis"] == key]
            if not row.empty:
                sig = f" {row.iloc[0]['significant']}"
        subplot_titles.append(
            f"{AXIS_LABEL_EN.get(key, key)}<br>"
            f"<sup>({AXIS_LABEL_ZH.get(key, '')}){sig}</sup>"
        )

    fig_f = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=subplot_titles,
        vertical_spacing=0.10,
        horizontal_spacing=0.06,
    )

    for idx, key in enumerate(AXIS_KEYS):
        row_idx = idx // n_cols + 1
        col_idx = idx % n_cols + 1
        col = f"bias_{key}"

        for lang in ["en", "zh", "jp"]:
            sub = bias_df[bias_df["lang"] == lang]
            if sub.empty or col not in sub.columns:
                continue
            vals = sub[col].dropna().values
            fig_f.add_trace(
                go.Violin(
                    y=vals,
                    name=LANG_LABEL.get(lang, lang),
                    box_visible=True,
                    meanline_visible=True,
                    fillcolor=LANG_COLOR[lang],
                    opacity=0.65,
                    line_color=LANG_COLOR[lang],
                    showlegend=(idx == 0),   # 仅第一个子图显示图例
                    legendgroup=lang,
                    hovertemplate=(
                        f"{LANG_LABEL.get(lang, lang)}<br>"
                        f"{AXIS_LABEL_EN.get(key, key)}<br>"
                        "Bias: %{y:.4f}<extra></extra>"
                    ),
                ),
                row=row_idx, col=col_idx,
            )

        # 添加 y=0 参考线
        fig_f.add_hline(
            y=0, line_dash="dash", line_color="gray", opacity=0.4,
            row=row_idx, col=col_idx,
        )

    fig_f.update_layout(
        **_LAYOUT_BASE,
        title=dict(
            text=(
                "Fig F: Moral Axis Bias Distribution by Language (Violin)<br>"
                "<sup>三语言在各道德轴上的 Bias 分布（小提琴图；* p<0.05, ** p<0.01, *** p<0.001）</sup>"
            ),
            font=dict(size=13),
        ),
        violinmode="group",
        height=300 * n_rows + 100,
        legend=dict(title="Language", orientation="h", y=-0.05),
    )

    p = DATA_DIR / "rq3_F_lang_bias_violin.html"
    fig_f.write_html(str(p))
    inject_font(p)
    log.info(f"[STEP4] ✅ 图F: {p.name}")


def aggregate_and_visualize(bias_df: pd.DataFrame, anova_df: pd.DataFrame):
    """
    新版图组把“绝对均值差异”和“语言内部 topic 结构”拆开表达：
      图A: topic-language × axis 热力图（语言内部 z-score，主图）
      图B: 语言层概览（上=raw mean±95%CI，下=语言内部相对显著性）
      图C: 轴关系热图（overall + 分语言）
      图D: topic-language 动机空间图（PCA biplot）
      图E: 每种语言最典型的负向动机 topic
      图E2: topic 语言构成
      图F: raw bias 分布小提琴图

    同时输出两个 CSV：
      - rq3_summary.csv       语言×轴汇总（增强版）
      - rq3_topic_summary.csv topic-language 级汇总（可直接写结果）
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    log.info(f"[STEP4] 开始生成可视化，共 {len(bias_df)} 条记录")

    bias_df, topic_lang_df = _prepare_topic_language_profiles(bias_df)
    log.info(f"[STEP4] 语言分布: {bias_df['lang'].value_counts().to_dict()}")
    log.info(f"[STEP4] Topic 数量: {bias_df['topic'].nunique()}")

    bias_cols = [f"bias_{k}" for k in AXIS_KEYS]
    z_cols = [f"z_{k}" for k in AXIS_KEYS]
    bilingual_x = [axis_bilingual_label(k) for k in AXIS_KEYS]

    # ── 图A: topic-language × axis 热力图（语言内部 z-score） ─────────────────
    log.info("[STEP4] 图A: topic-language × axis 热力图（z-score）")

    plot_topic_lang = topic_lang_df[topic_lang_df["n_docs"] >= 5].copy()
    plot_topic_lang = plot_topic_lang.sort_values(
        ["lang", "dominant_negative_axis", "dominant_negative_z", "n_docs"],
        ascending=[True, True, True, False],
    )

    heat_values = plot_topic_lang[z_cols].values
    heat_text = np.vectorize(lambda v: f"{v:+.2f}")(heat_values)
    customdata = np.dstack([
        np.tile(plot_topic_lang["row_label"].values.reshape(-1, 1), (1, len(AXIS_KEYS))),
        np.tile(plot_topic_lang["n_docs"].values.reshape(-1, 1), (1, len(AXIS_KEYS))),
        np.array([[plot_topic_lang.iloc[i][f"bias_{k}"] for k in AXIS_KEYS] for i in range(len(plot_topic_lang))]),
        np.array([[plot_topic_lang.iloc[i][f"z_{k}"] for k in AXIS_KEYS] for i in range(len(plot_topic_lang))]),
    ])

    fig_a = go.Figure(go.Heatmap(
        z=heat_values,
        x=bilingual_x,
        y=plot_topic_lang["row_label"].tolist(),
        colorscale="RdBu",
        zmid=0,
        zmin=-2.5,
        zmax=2.5,
        text=heat_text,
        texttemplate="%{text}",
        customdata=customdata,
        colorbar=dict(title="Within-language<br>z-score"),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "文档数: %{customdata[1]}<br>"
            "轴: %{x}<br>"
            "raw bias: %{customdata[2]:+.4f}<br>"
            "within-lang z: %{customdata[3]:+.2f}<extra></extra>"
        ),
    ))
    fig_a.update_layout(
        **_LAYOUT_BASE,
        title=dict(
            text=(
                "Fig A: Topic-Language Moral Profile Heatmap (within-language z-score)<br>"
                "<sup>主图：每一行是一个 topic-language 单元。红色表示该轴在该语言内部更偏负向、更能代表攻击性动机</sup>"
            ),
            font=dict(size=13),
        ),
        xaxis=dict(title="Moral Axis (道德维度)", tickangle=-15),
        yaxis=dict(title="Topic-Language Unit"),
        height=max(700, len(plot_topic_lang) * 18),
    )
    p = DATA_DIR / "rq3_A_topic_axis_heatmap.html"
    fig_a.write_html(str(p))
    inject_font(p)
    log.info(f"[STEP4] ✅ 图A: {p.name}")

    # ── 图B: 语言概览（上=raw mean±CI, 下=相对显著性） ───────────────────────
    log.info("[STEP4] 图B: 语言层概览")

    lang_stats_rows = []
    for lang in ["en", "zh", "jp"]:
        lang_sub = bias_df[bias_df["lang"] == lang]
        if lang_sub.empty:
            continue
        for key in AXIS_KEYS:
            vals = lang_sub[f"bias_{key}"].dropna().values
            n = len(vals)
            mean = float(vals.mean())
            se = float(vals.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
            ci = 1.96 * se
            lang_stats_rows.append({
                "lang": lang,
                "axis": key,
                "mean_bias": mean,
                "se_bias": se,
                "ci_low": mean - ci,
                "ci_high": mean + ci,
            })
    lang_stats = pd.DataFrame(lang_stats_rows)
    lang_stats["relative_salience"] = lang_stats.groupby("lang")["mean_bias"].transform(
        lambda s: s - s.mean()
    )
    lang_means = lang_stats.pivot(index="lang", columns="axis", values="mean_bias").reindex(
        index=["en", "zh", "jp"], columns=AXIS_KEYS
    )

    fig_b = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=[
            "Raw mean bias with 95% CI",
            "Relative salience within each language",
        ],
    )
    for lang in ["en", "zh", "jp"]:
        sub = lang_stats[lang_stats["lang"] == lang]
        if sub.empty:
            continue
        sub = sub.set_index("axis").reindex(AXIS_KEYS).reset_index()
        error_plus = sub["ci_high"] - sub["mean_bias"]
        error_minus = sub["mean_bias"] - sub["ci_low"]
        fig_b.add_trace(
            go.Scatter(
                x=bilingual_x,
                y=sub["mean_bias"],
                mode="lines+markers",
                name=LANG_LABEL.get(lang, lang),
                line=dict(color=LANG_COLOR[lang], width=2),
                marker=dict(size=8),
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=error_plus,
                    arrayminus=error_minus,
                    thickness=1.2,
                    width=3,
                ),
                hovertemplate=(
                    f"{LANG_LABEL.get(lang, lang)}<br>"
                    "轴: %{x}<br>"
                    "均值: %{y:+.4f}<extra></extra>"
                ),
                legendgroup=lang,
            ),
            row=1, col=1,
        )
        fig_b.add_trace(
            go.Scatter(
                x=bilingual_x,
                y=sub["relative_salience"],
                mode="lines+markers",
                name=LANG_LABEL.get(lang, lang),
                line=dict(color=LANG_COLOR[lang], width=2, dash="dot"),
                marker=dict(size=8),
                hovertemplate=(
                    f"{LANG_LABEL.get(lang, lang)}<br>"
                    "轴: %{x}<br>"
                    "相对显著性: %{y:+.4f}<extra></extra>"
                ),
                legendgroup=lang,
                showlegend=False,
            ),
            row=2, col=1,
        )

    fig_b.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=1)
    fig_b.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=1)
    if anova_df is not None and not anova_df.empty:
        for i, key in enumerate(AXIS_KEYS):
            row = anova_df[anova_df["axis"] == key]
            if not row.empty and row.iloc[0]["significant"] != "ns":
                fig_b.add_annotation(
                    x=bilingual_x[i],
                    y=float(lang_stats[lang_stats["axis"] == key]["ci_high"].max()) + 0.01,
                    text=row.iloc[0]["significant"],
                    showarrow=False,
                    font=dict(size=12, color="red"),
                    row=1, col=1,
                )

    fig_b.update_layout(
        **_LAYOUT_BASE,
        title=dict(
            text=(
                "Fig B: Language-Level Overview of Moral Motivation<br>"
                "<sup>上图保留 raw bias；下图去掉语言整体基线后，展示每种语言内部哪些轴更突出</sup>"
            ),
            font=dict(size=13),
        ),
        xaxis2=dict(title="Moral Axis (道德维度)", tickangle=-15),
        yaxis=dict(title="Raw mean bias"),
        yaxis2=dict(title="Relative salience"),
        legend=dict(title="Language", orientation="h", y=1.08),
        height=760,
    )
    p = DATA_DIR / "rq3_B_lang_axis_bar.html"
    fig_b.write_html(str(p))
    inject_font(p)
    log.info(f"[STEP4] ✅ 图B: {p.name}")

    # ── 图C: 轴之间的联系（相关矩阵） ────────────────────────────────────────
    log.info("[STEP4] 图C: 轴关系热图")

    corr_targets = [("all", "All topic-language units"), ("en", LANG_LABEL["en"]), ("zh", LANG_LABEL["zh"]), ("jp", LANG_LABEL["jp"])]
    fig_c = make_subplots(
        rows=2, cols=2,
        subplot_titles=[label for _, label in corr_targets],
        horizontal_spacing=0.08,
        vertical_spacing=0.14,
    )
    axis_short = [AXIS_LABEL_EN[k] for k in AXIS_KEYS]
    for idx, (lang_key, label) in enumerate(corr_targets):
        row_idx = idx // 2 + 1
        col_idx = idx % 2 + 1
        sub = topic_lang_df if lang_key == "all" else topic_lang_df[topic_lang_df["lang"] == lang_key]
        corr = sub[z_cols].corr().fillna(0)
        heatmap_kwargs = dict(
            z=corr.values,
            x=axis_short,
            y=axis_short,
            zmin=-1,
            zmax=1,
            zmid=0,
            colorscale="RdBu",
            text=np.vectorize(lambda v: f"{v:+.2f}")(corr.values),
            texttemplate="%{text}",
            hovertemplate="X: %{x}<br>Y: %{y}<br>r=%{z:+.2f}<extra></extra>",
            showscale=(idx == 0),
        )
        if idx == 0:
            heatmap_kwargs["colorbar"] = dict(title="corr")
        fig_c.add_trace(
            go.Heatmap(**heatmap_kwargs),
            row=row_idx, col=col_idx,
        )
    fig_c.update_layout(
        **_LAYOUT_BASE,
        title=dict(
            text=(
                "Fig C: Relations Among Moral Axes<br>"
                "<sup>基于 topic-language 的语言内标准化 profile；能直接看出哪些轴在各语言里总是一起出现</sup>"
            ),
            font=dict(size=13),
        ),
        height=820,
    )
    p = DATA_DIR / "rq3_C_lang_radar.html"
    fig_c.write_html(str(p))
    inject_font(p)
    log.info(f"[STEP4] ✅ 图C: {p.name}")

    # ── 图D: topic-language 动机空间（PCA biplot） ───────────────────────────
    log.info("[STEP4] 图D: topic-language 动机空间图")

    scatter_df = topic_lang_df[topic_lang_df["n_docs"] >= 5].copy()
    scores, explained, loadings = _compute_pca_projection(scatter_df[z_cols].values)
    scatter_df["pc1"] = scores[:, 0]
    scatter_df["pc2"] = scores[:, 1]
    scatter_df["label_text"] = ""
    label_candidates = scatter_df.assign(
        extremeness=scatter_df["pc1"].abs() + scatter_df["pc2"].abs()
    ).sort_values(["extremeness", "n_docs"], ascending=False).head(18).index
    scatter_df.loc[label_candidates, "label_text"] = scatter_df.loc[label_candidates].apply(
        lambda r: f"{r['lang'].upper()}-T{int(r['topic'])}", axis=1
    )

    fig_d = go.Figure()
    for lang in ["en", "zh", "jp"]:
        sub = scatter_df[scatter_df["lang"] == lang]
        if sub.empty:
            continue
        hover = []
        for _, row in sub.iterrows():
            axis_bits = "<br>".join(
                f"{AXIS_LABEL_EN[key]}: z={row[f'z_{key}']:+.2f}, raw={row[f'bias_{key}']:+.3f}"
                for key in AXIS_KEYS
            )
            hover.append(
                f"<b>{row['row_label']}</b><br>"
                f"文档数: {int(row['n_docs'])}<br>"
                f"主导负向动机: {row['dominant_negative_axis_label']}<br><br>"
                f"{axis_bits}"
            )
        fig_d.add_trace(go.Scatter(
            x=sub["pc1"],
            y=sub["pc2"],
            mode="markers+text",
            text=sub["label_text"],
            textposition="top center",
            textfont=dict(size=9),
            name=LANG_LABEL.get(lang, lang),
            marker=dict(
                size=8 + (sub["n_docs"] / sub["n_docs"].max() * 16),
                color=LANG_COLOR[lang],
                opacity=0.75,
                line=dict(width=0.8, color="white"),
            ),
            customdata=np.array(hover, dtype=object),
            hovertemplate="%{customdata}<extra></extra>",
        ))

    arrow_scale = max(scatter_df["pc1"].abs().max(), scatter_df["pc2"].abs().max(), 1.0) * 0.55
    for idx, key in enumerate(AXIS_KEYS):
        end_x = loadings[idx, 0] * arrow_scale
        end_y = loadings[idx, 1] * arrow_scale
        fig_d.add_trace(go.Scatter(
            x=[0, end_x],
            y=[0, end_y],
            mode="lines+text",
            text=["", AXIS_LABEL_EN[key]],
            textposition="top center",
            line=dict(color=AXIS_COLOR.get(key, "#666666"), width=2),
            hovertemplate=f"{AXIS_LABEL_EN[key]} loading<extra></extra>",
            showlegend=False,
        ))

    fig_d.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.4)
    fig_d.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.4)
    fig_d.update_layout(
        **_LAYOUT_BASE,
        title=dict(
            text=(
                "Fig D: Topic-Language Moral Profile Map (PCA biplot)<br>"
                f"<sup>点=topic-language 单元；箭头=7个动机轴。PC1 解释 {explained[0]*100:.1f}% 方差，PC2 解释 {explained[1]*100:.1f}%</sup>"
            ),
            font=dict(size=13),
        ),
        xaxis=dict(title="PC1"),
        yaxis=dict(title="PC2"),
        legend=dict(title="Language"),
        height=720,
    )

    p = DATA_DIR / "rq3_D_topic_lang_scatter.html"
    fig_d.write_html(str(p))
    inject_font(p)
    log.info(f"[STEP4] ✅ 图D: {p.name}")

    # ── 图E / E2: 最典型 topic + topic 语言构成 ──────────────────────────────
    log.info("[STEP4] 图E: topic 典型单元")
    _build_topic_exemplar_chart(topic_lang_df)

    # ── 图F: 语言 × 轴 Bias 分布小提琴图 ─────────────────────────────────
    log.info("[STEP4] 图F: 语言×轴 Bias 分布小提琴图")
    _build_violin_plot(bias_df, anova_df)

    # ── CSV 汇总（论文附录） ───────────────────────────────────────────────
    summary_rows = []
    for lang in ["en", "zh", "jp"]:
        lang_sub = bias_df[bias_df["lang"] == lang]
        if lang_sub.empty:
            continue
        lang_axis_means = {
            key: float(lang_sub[f"bias_{key}"].mean()) for key in AXIS_KEYS
        }
        lang_grand_mean = float(np.mean(list(lang_axis_means.values())))
        rank_map = {
            axis: rank + 1
            for rank, axis in enumerate(
                sorted(AXIS_KEYS, key=lambda ax: lang_axis_means[ax] - lang_grand_mean)
            )
        }
        for key in AXIS_KEYS:
            col = f"bias_{key}"
            if col not in lang_sub.columns:
                continue
            vals = lang_sub[col].dropna().values
            n = len(vals)
            se = float(vals.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
            ci = 1.96 * se
            # 找出每个 topic 内 Bias 最负的（攻击动机最强的）
            topic_means = lang_sub.groupby("topic")[col].mean()
            most_negative_topic = topic_means.idxmin() if not topic_means.empty else -1
            summary_rows.append({
                "lang":         lang,
                "lang_label":   LANG_LABEL.get(lang, lang),
                "axis":         key,
                "axis_label":   AXIS_LABEL_EN[key],
                "axis_desc":    AXIS_LABEL_DESC[key],
                "n_docs":       int(len(vals)),
                "mean_bias":    round(float(vals.mean()), 5),
                "std_bias":     round(float(vals.std()),  5),
                "se_bias":      round(se, 5),
                "ci_low":       round(float(vals.mean() - ci), 5),
                "ci_high":      round(float(vals.mean() + ci), 5),
                "median_bias":  round(float(np.median(vals)), 5),
                "q25_bias":     round(float(np.percentile(vals, 25)), 5),
                "q75_bias":     round(float(np.percentile(vals, 75)), 5),
                "relative_salience": round(lang_axis_means[key] - lang_grand_mean, 5),
                "axis_rank_within_lang": rank_map[key],
                "most_negative_topic": most_negative_topic,
            })
    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(["lang", "axis_rank_within_lang", "mean_bias"])
    summary_df.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")
    log.info(f"[STEP4] ✅ 汇总CSV: {SUMMARY_CSV}")

    topic_summary_df = topic_lang_df.merge(
        scatter_df[["lang", "topic", "pc1", "pc2"]],
        on=["lang", "topic"],
        how="left",
    )
    topic_summary_df = topic_summary_df.sort_values(
        ["lang", "dominant_negative_z", "n_docs"], ascending=[True, True, False]
    )
    topic_summary_df.to_csv(TOPIC_SUMMARY_CSV, index=False, encoding="utf-8-sig")
    log.info(f"[STEP4] ✅ Topic汇总CSV: {TOPIC_SUMMARY_CSV}")

    # ── 终端摘要 ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("📊 RQ3 核心发现（FrameAxis 道德动机投影）")
    print("=" * 70)
    print("\n▶ 各语言在各道德轴上的平均 Bias（越负=该轴攻击动机越强）：")
    for lang in ["en", "zh", "jp"]:
        if lang not in lang_means.index:
            continue
        label = LANG_LABEL.get(lang, lang)
        print(f"\n  {label}:")
        for key in AXIS_KEYS:
            mean_val = lang_means.loc[lang, key]
            bar = "▓" * int(abs(mean_val) * 500)
            direction = "←攻击" if mean_val < 0 else "→正向"
            print(f"    {AXIS_LABEL_EN[key]:20s}: {mean_val:+.4f} {direction} {bar}")
        lang_topics = topic_lang_df[topic_lang_df["lang"] == lang].sort_values(
            "dominant_negative_z"
        ).head(3)
        print("    典型 topic:")
        for _, row in lang_topics.iterrows():
            print(
                f"      T{int(row['topic']):<2d} "
                f"{row['dominant_negative_axis_label']:<18s} "
                f"z={row['dominant_negative_z']:+.2f} | {row['topic_short']}"
            )
    if anova_df is not None and not anova_df.empty:
        print("\n▶ ANOVA 显著性检验：")
        for _, row in anova_df.iterrows():
            print(
                f"  [{row['axis']:20s}] F={row['F_stat']:.2f}, "
                f"p={row['p_value']:.4e}, η²={row['eta_squared']:.4f} {row['significant']}"
            )
    print(f"\n▶ 输出文件目录: {DATA_DIR}")
    print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="RQ3 分析管线 — 词典轴道德动机投影 (FrameAxis)",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
运行模式:
  完整运行           python main.py
  跳过轴构建         python main.py --from-bias           # 已有 rq3_bias_matrix.csv
  只重跑可视化       python main.py --viz-only            # 已有 rq3_bias_matrix.csv
  调试（小数据）     python main.py --max-docs 200
""",
    )
    parser.add_argument(
        "--viz-only", action="store_true",
        help="跳过 Step1+2+3，直接从 rq3_bias_matrix.csv 重跑可视化",
    )
    parser.add_argument(
        "--from-bias", action="store_true",
        help="跳过 Step1（轴构建）和 Step2（文档编码），直接从 bias_matrix.csv 开始",
    )
    parser.add_argument(
        "--max-docs", type=int, default=None,
        help="调试：只处理前 N 条文档（仅影响 Step2）",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="E5 编码批大小（默认 64，显存不足时调小）",
    )
    args = parser.parse_args()

    # ── 初始化日志（使用项目统一的 set_logger） ──────────────────────────────
    global log
    log, log_path = setup_logging("rq3_frameaxis")
    log.info(f"日志文件: {log_path}")

    # ── 可视化重跑模式 ───────────────────────────────────────────────────────
    if args.viz_only:
        if not CKPT_BIAS.exists():
            log.error(f"[VIZ-ONLY] 找不到 {CKPT_BIAS}，请先完整运行管线")
            sys.exit(1)
        bias_df = load_bias_matrix()
        anova_df = pd.read_csv(ANOVA_CSV) if ANOVA_CSV.exists() else None
        if anova_df is None:
            log.info("[VIZ-ONLY] 未找到 ANOVA 结果，跳过显著性标注")
        aggregate_and_visualize(bias_df, anova_df)
        log.info(f"✅ 可视化重跑完成。日志: {log_path}")
        return

    log.info("=" * 60)
    log.info("RQ3 管线启动：FrameAxis 道德动机投影")
    log.info("=" * 60)

    # ── 加载原始文档数据 ─────────────────────────────────────────────────────
    if not CKPT_BIAS.exists() or not args.from_bias:
        if not DOC_PATH.exists():
            log.error(f"[DATA] 找不到文档数据: {DOC_PATH}")
            sys.exit(1)
        doc_df = pd.read_csv(DOC_PATH)
        doc_df = doc_df[doc_df["topic"] != -1].reset_index(drop=True)
        log.info(
            f"[DATA] 加载文档 {len(doc_df)} 条 | "
            f"语言分布: {doc_df['lang'].value_counts().to_dict()} | "
            f"Topic 数量: {doc_df['topic'].nunique()}"
        )
        if args.max_docs:
            doc_df = doc_df.head(args.max_docs)
            log.info(f"[DEBUG] 限制前 {args.max_docs} 条")

    # ── Step 1: 构建道德轴向量 ───────────────────────────────────────────────
    if args.from_bias:
        # 跳过 Step1 + Step2，直接加载 bias matrix
        if not CKPT_BIAS.exists():
            log.error(
                f"[STEP1] --from-bias 要求跳过编码阶段，"
                f"但找不到 {CKPT_BIAS}，请先完整运行一次"
            )
            sys.exit(1)
        log.info("[STEP1/2] ⏭ --from-bias: 跳过轴构建与文档编码")
        bias_df = load_bias_matrix()
    else:
        # 轴向量：若 checkpoint 已存在可直接加载
        if CKPT_AXIS.exists():
            log.info("[STEP1] ✅ 发现已有轴向量 checkpoint，直接加载（如需重建请删除 rq3_axis_vectors.npz）")
            axis_vectors = load_axis_vectors()
            model = None
        else:
            model = load_encoder()
            axis_vectors = build_axis_vectors(model)

        # ── Step 2: 文档投影 ─────────────────────────────────────────────────
        log.info("[STEP2] 文档向量投影开始...")
        if model is None:
            model = load_encoder()
        bias_df = compute_bias_matrix(
            doc_df, axis_vectors, model, batch_size=args.batch_size
        )

    # ── Step 3: ANOVA 统计检验 ───────────────────────────────────────────────
    log.info("[STEP3] 统计检验...")
    anova_df = run_anova(bias_df)

    # ── Step 4: 聚合可视化 ───────────────────────────────────────────────────
    log.info("[STEP4] 聚合分析与可视化...")
    aggregate_and_visualize(bias_df, anova_df)

    log.info("=" * 60)
    log.info(f"✅ RQ3 全部完成。日志: {log_path}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
