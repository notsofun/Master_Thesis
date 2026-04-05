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
  完整运行           python main.py
  跳过轴构建         python main.py --from-bias             # 已有 bias_matrix.csv
  只重跑可视化       python main.py --viz-only              # 已有 bias_matrix.csv
  调试（小数据）     python main.py --max-docs 200

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
from scipy import stats as scipy_stats

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

def _build_topic_network(bias_df: pd.DataFrame, anova_df: pd.DataFrame | None):
    """
    图E: Topic 共现网络（交互式 Plotly）

    节点设计：
      - 每个 Topic 是一个节点
      - 节点大小 = 该 Topic 文档总数
      - 节点颜色 = 该 Topic 在"最显著轴"（ANOVA η² 最大轴）上的平均 Bias
        （红色=攻击动机强，蓝色=偏正向）
      - 节点内显示 Topic 英文短标签
      - Hover 显示：Topic 标签、语言占比（en/zh/jp %）、各轴 Bias 均值

    边设计：
      - 两个 Topic 的 Bias 向量余弦相似度 > 阈值 时连边
      - 边粗细 = 相似度大小
      - 布局用 Fruchterman-Reingold (networkx spring layout)

    另单独输出 rq3_E2_topic_lang_pie.html：
      每个 Topic 的语言占比条形图（stacked bar，横轴=Topic，颜色=语言）
    """
    import plotly.graph_objects as go
    try:
        import networkx as nx
    except ImportError:
        log.warning("[STEP4] networkx 未安装，跳过图E。pip install networkx")
        return

    bias_df = bias_df.copy()
    bias_df["lang"] = bias_df["lang"].apply(normalize_lang).replace({"ja": "jp"})
    bias_cols = [f"bias_{k}" for k in AXIS_KEYS]

    # ── 计算每个 topic 的节点属性 ────────────────────────────────────────────
    topic_ids = sorted([t for t in bias_df["topic"].unique() if t >= 0])

    # 各 topic 平均 Bias 向量（用于连边相似度 + 颜色）
    topic_bias_mean = (
        bias_df[bias_df["topic"] >= 0]
        .groupby("topic")[bias_cols].mean()
    )

    # 选取"主轴"：ANOVA η² 最大的轴（若无 ANOVA 结果则用 sanctity）
    main_axis = "sanctity"
    if anova_df is not None and not anova_df.empty and "eta_squared" in anova_df.columns:
        valid = anova_df.dropna(subset=["eta_squared"])
        if not valid.empty:
            main_axis = valid.loc[valid["eta_squared"].idxmax(), "axis"]
    main_col = f"bias_{main_axis}"
    log.info(f"[E] 网络图颜色轴（η²最大）: {main_axis}")

    # 各 topic 文档总数
    topic_counts = bias_df[bias_df["topic"] >= 0]["topic"].value_counts()

    # 各 topic 语言占比
    topic_lang = (
        bias_df[bias_df["topic"] >= 0]
        .groupby(["topic", "lang"]).size()
        .reset_index(name="n")
    )
    topic_total = topic_lang.groupby("topic")["n"].transform("sum")
    topic_lang["pct"] = (topic_lang["n"] / topic_total * 100).round(1)

    # ── 构建网络图 ────────────────────────────────────────────────────────────
    # 仅使用在 topic_bias_mean 中存在的 topic（有足够文档的话题）
    valid_topics = [t for t in topic_ids if t in topic_bias_mean.index]

    G = nx.Graph()
    for t in valid_topics:
        G.add_node(t)

    # 计算两两 topic Bias 向量余弦相似度，超过阈值则连边
    SIM_THRESHOLD = 0.85   # 相似度阈值（0.85 表示道德动机高度相似）
    bias_matrix = topic_bias_mean.loc[valid_topics, bias_cols].values
    # L2 归一化
    norms = np.linalg.norm(bias_matrix, axis=1, keepdims=True)
    bias_matrix_norm = bias_matrix / (norms + 1e-12)
    sim_mat = bias_matrix_norm @ bias_matrix_norm.T   # (n_topics × n_topics)

    edge_weights = []
    for i, ti in enumerate(valid_topics):
        for j, tj in enumerate(valid_topics):
            if j <= i:
                continue
            sim = float(sim_mat[i, j])
            if sim >= SIM_THRESHOLD:
                G.add_edge(ti, tj, weight=sim)
                edge_weights.append(sim)

    log.info(f"[E] 网络: {G.number_of_nodes()} 节点, {G.number_of_edges()} 条边 (阈值={SIM_THRESHOLD})")

    # 布局（spring layout）
    pos = nx.spring_layout(G, seed=42, k=2.5 / max(len(valid_topics) ** 0.5, 1))

    # ── 构建 Plotly 图形 ──────────────────────────────────────────────────────
    # 边迹
    edge_x, edge_y, edge_hover = [], [], []
    for (u, v, d) in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_hover.append(f"T{u}↔T{v}: sim={d['weight']:.3f}")

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=0.8, color="#BBBBBB"),
        hoverinfo="none",
        showlegend=False,
    )

    # 节点迹
    node_x, node_y, node_sizes, node_colors = [], [], [], []
    node_texts, node_hovers = [], []

    for t in valid_topics:
        if t not in pos:
            continue
        x, y = pos[t]
        node_x.append(x)
        node_y.append(y)

        n_docs = int(topic_counts.get(t, 10))
        # 节点大小：文档数映射到 [10, 45]
        size = 10 + min(35, n_docs / max(topic_counts.max(), 1) * 35)
        node_sizes.append(size)

        # 节点颜色：主轴 Bias（负=红，正=蓝）
        bias_val = float(topic_bias_mean.loc[t, main_col]) if main_col in topic_bias_mean.columns else 0.0
        node_colors.append(bias_val)

        # 节点文本（短标签）
        label = TOPIC_LABELS.get(t, f"T{t}")
        short = label[:22] + "…" if len(label) > 22 else label
        node_texts.append(f"T{t}")

        # Hover 内容
        lang_info = topic_lang[topic_lang["topic"] == t]
        lang_str = " | ".join(
            f"{LANG_LABEL.get(r['lang'], r['lang'])}: {r['pct']:.0f}%"
            for _, r in lang_info.iterrows()
        )
        bias_str = " | ".join(
            f"{AXIS_LABEL_EN.get(k, k)}: {topic_bias_mean.loc[t, f'bias_{k}']:+.3f}"
            for k in AXIS_KEYS if f"bias_{k}" in topic_bias_mean.columns
        )
        node_hovers.append(
            f"<b>T{t}: {label}</b><br>"
            f"文档数: {n_docs}<br>"
            f"语言占比: {lang_str}<br>"
            f"道德偏移: {bias_str}"
        )

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=node_texts,
        textposition="top center",
        textfont=dict(size=8),
        hovertext=node_hovers,
        hoverinfo="text",
        marker=dict(
            size=node_sizes,
            color=node_colors,
            colorscale="RdBu",
            cmin=-0.04,
            cmax=0.04,
            colorbar=dict(
                title=f"{AXIS_LABEL_EN.get(main_axis, main_axis)} Bias<br>(主轴颜色)",
                thickness=12,
                len=0.6,
            ),
            line=dict(width=1.2, color="white"),
            showscale=True,
        ),
        showlegend=False,
    )

    fig_e = go.Figure(data=[edge_trace, node_trace])
    fig_e.update_layout(
        **_LAYOUT_BASE,
        title=dict(
            text=(
                "Fig E: Topic Moral Motivation Network "
                f"(cosine sim ≥ {SIM_THRESHOLD}, color={AXIS_LABEL_EN.get(main_axis,'?')} Bias)<br>"
                "<sup>话题道德动机共现网络（节点大小=文档数，颜色=主轴偏移，连边=动机相似）</sup>"
            ),
            font=dict(size=13),
        ),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=780,
        hovermode="closest",
    )
    p = DATA_DIR / "rq3_E_topic_network.html"
    fig_e.write_html(str(p))
    inject_font(p)
    log.info(f"[STEP4] ✅ 图E: {p.name}")

    # ── 图E2: Topic 语言占比堆叠条形图 ──────────────────────────────────────
    all_topics_sorted = sorted(valid_topics)
    lang_pct_pivot = (
        topic_lang[topic_lang["topic"].isin(all_topics_sorted)]
        .pivot(index="topic", columns="lang", values="pct")
        .fillna(0)
        .reindex(all_topics_sorted)
    )

    fig_e2 = go.Figure()
    for lang in ["en", "zh", "jp"]:
        if lang not in lang_pct_pivot.columns:
            continue
        x_labels = [
            f"T{t}: {TOPIC_LABELS.get(t, '')[:18]}"
            for t in lang_pct_pivot.index
        ]
        fig_e2.add_trace(go.Bar(
            name=LANG_LABEL.get(lang, lang),
            x=x_labels,
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
                "<sup>各话题语言构成占比（按语言堆叠）</sup>"
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
    生成四张 Plotly 可视化图表：
      图A: 47 Topic × 7 Axis Bias 热力图（各 Topic 的道德倾向全貌）
      图B: 三语言 × 7 Axis 均值条形图（语言间道德动机对比，核心图）
      图C: 语言雷达图（每种语言在7个轴上的 Bias 均值，直观展示动机剖面）
      图D: 重要 Topic 的道德动机散点图（两轴散点，颜色=语言，展示语言聚集性）

    同时生成 rq3_summary.csv（论文附录）。
    """
    import plotly.graph_objects as go
    import plotly.express as px

    log.info(f"[STEP4] 开始生成可视化，共 {len(bias_df)} 条记录")

    # 规范化语言列
    bias_df = bias_df.copy()
    bias_df["lang"] = bias_df["lang"].apply(normalize_lang).replace({"ja": "jp"})
    log.info(f"[STEP4] 语言分布: {bias_df['lang'].value_counts().to_dict()}")
    log.info(f"[STEP4] Topic 数量: {bias_df['topic'].nunique()}")

    bias_cols = [f"bias_{k}" for k in AXIS_KEYS]
    bilingual_x = [axis_bilingual_label(k) for k in AXIS_KEYS]

    # ── 图A: Topic × Axis Bias 热力图 ────────────────────────────────────────
    log.info("[STEP4] 图A: Topic × Axis 热力图")

    # 取出现次数最多的 30 个 Topic（避免图太拥挤）
    top_topics = bias_df["topic"].value_counts().head(30).index
    topic_df = bias_df[bias_df["topic"].isin(top_topics)]

    # 计算每个 Topic 在每个轴上的平均 Bias
    topic_bias = (
        topic_df.groupby("topic")[bias_cols].mean().round(4)
    )
    topic_bias.index = [f"T{t}" for t in topic_bias.index]

    fig_a = go.Figure(go.Heatmap(
        z=topic_bias.values,
        x=bilingual_x,
        y=topic_bias.index.tolist(),
        colorscale="RdBu",      # 红=负(攻击动机强), 蓝=正(正向语义)
        zmid=0,
        text=topic_bias.values.round(3),
        texttemplate="%{text}",
        colorbar=dict(title="Bias<br>(cos sim)"),
        hovertemplate="Topic %{y}<br>轴: %{x}<br>Bias: %{z:.4f}<extra></extra>",
    ))
    fig_a.update_layout(
        **_LAYOUT_BASE,
        title=dict(
            text=(
                "Fig A: Moral Axis Bias per Topic (avg cosine similarity)<br>"
                "<sup>各话题在七个道德轴上的平均语义偏移（红色=攻击性动机强）</sup>"
            ),
            font=dict(size=13),
        ),
        xaxis=dict(title="Moral Axis (道德维度)", tickangle=-15),
        yaxis=dict(title="Topic"),
        height=max(500, len(topic_bias) * 22),
    )
    p = DATA_DIR / "rq3_A_topic_axis_heatmap.html"
    fig_a.write_html(str(p))
    inject_font(p)
    log.info(f"[STEP4] ✅ 图A: {p.name}")

    # ── 图B: 三语言 × 7 Axis 均值条形图（核心对比图） ─────────────────────
    log.info("[STEP4] 图B: 语言 × 轴均值条形图")

    lang_means = (
        bias_df.groupby("lang")[bias_cols].mean().round(4)
    )

    fig_b = go.Figure()
    for lang in ["en", "zh", "jp"]:
        if lang not in lang_means.index:
            log.warning(f"[STEP4] 语言 {lang} 无数据，跳过图B中该语言")
            continue
        means = lang_means.loc[lang].values
        fig_b.add_trace(go.Bar(
            name=LANG_LABEL.get(lang, lang),
            x=bilingual_x,
            y=means,
            marker_color=LANG_COLOR[lang],
            hovertemplate=(
                f"{LANG_LABEL.get(lang, lang)}<br>"
                "轴: %{x}<br>Bias均值: %{y:.4f}<extra></extra>"
            ),
        ))

    # 添加 y=0 参考线
    fig_b.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.6)

    # 在图上标注 ANOVA 显著性标记
    if anova_df is not None and not anova_df.empty:
        for i, key in enumerate(AXIS_KEYS):
            row = anova_df[anova_df["axis"] == key]
            if not row.empty:
                sig = row.iloc[0]["significant"]
                if sig != "ns":
                    fig_b.add_annotation(
                        x=bilingual_x[i],
                        y=lang_means[f"bias_{key}"].max() + 0.005,
                        text=sig,
                        showarrow=False,
                        font=dict(size=12, color="red"),
                    )

    fig_b.update_layout(
        **_LAYOUT_BASE,
        barmode="group",
        title=dict(
            text=(
                "Fig B: Moral Motivation Bias by Language "
                "(avg cosine similarity to axis)<br>"
                "<sup>三语言在七个道德动机轴上的平均偏移对比（* p<0.05, ** p<0.01, *** p<0.001）</sup>"
            ),
            font=dict(size=13),
        ),
        xaxis=dict(title="Moral Axis (道德维度)", tickangle=-15),
        yaxis=dict(title="Avg Bias (平均余弦相似度)"),
        legend=dict(title="Language"),
        height=560,
    )
    p = DATA_DIR / "rq3_B_lang_axis_bar.html"
    fig_b.write_html(str(p))
    inject_font(p)
    log.info(f"[STEP4] ✅ 图B: {p.name}")

    # ── 图C: 语言雷达图（道德动机剖面） ──────────────────────────────────
    log.info("[STEP4] 图C: 语言雷达图")

    # 雷达图需要封闭（首尾相同）
    theta_labels = [AXIS_LABEL_EN.get(k, k) for k in AXIS_KEYS]
    theta_labels_closed = theta_labels + [theta_labels[0]]

    fig_c = go.Figure()
    for lang in ["en", "zh", "jp"]:
        if lang not in lang_means.index:
            continue
        r_vals = lang_means.loc[lang].values.tolist()
        r_vals_closed = r_vals + [r_vals[0]]
        fig_c.add_trace(go.Scatterpolar(
            r=r_vals_closed,
            theta=theta_labels_closed,
            fill="toself",
            name=LANG_LABEL.get(lang, lang),
            line=dict(color=LANG_COLOR[lang], width=2),
            opacity=0.7,
            hovertemplate="%{theta}: %{r:.4f}<extra></extra>",
        ))

    fig_c.update_layout(
        **_LAYOUT_BASE,
        polar=dict(
            radialaxis=dict(visible=True, range=[-0.05, 0.05]),
        ),
        title=dict(
            text=(
                "Fig C: Moral Motivation Profile by Language (Radar)<br>"
                "<sup>三语言道德动机剖面雷达图（中心=0, 外侧=正极, 内侧=攻击动机强）</sup>"
            ),
            font=dict(size=13),
        ),
        legend=dict(title="Language"),
        height=600,
    )
    p = DATA_DIR / "rq3_C_lang_radar.html"
    fig_c.write_html(str(p))
    inject_font(p)
    log.info(f"[STEP4] ✅ 图C: {p.name}")

    # ── 图D: Sanctity vs. Loyalty 散点图（展示两个最具区分性的轴） ─────────
    log.info("[STEP4] 图D: 关键轴散点图")

    # 取两个最有区分性的轴：sanctity（圣洁/堕落）和 loyalty（忠诚/背叛）
    # 这两轴在 Notion 设计方案中被特别强调
    x_axis_key = "sanctity"
    y_axis_key = "loyalty"

    # 每个 (topic, lang) 的平均 Bias
    scatter_df = (
        bias_df.groupby(["topic", "lang"])[
            [f"bias_{x_axis_key}", f"bias_{y_axis_key}"]
        ].mean().reset_index().round(4)
    )
    scatter_df["lang_label"] = scatter_df["lang"].map(LANG_LABEL)
    scatter_df["topic_label"] = scatter_df["topic"].apply(lambda t: f"T{t}")

    fig_d = px.scatter(
        scatter_df,
        x=f"bias_{x_axis_key}",
        y=f"bias_{y_axis_key}",
        color="lang",
        color_discrete_map=LANG_COLOR,
        hover_data=["topic_label", "lang_label"],
        text="topic_label",
        labels={
            f"bias_{x_axis_key}": f"{AXIS_LABEL_EN[x_axis_key]} Bias ({AXIS_LABEL_DESC[x_axis_key]})",
            f"bias_{y_axis_key}": f"{AXIS_LABEL_EN[y_axis_key]} Bias ({AXIS_LABEL_DESC[y_axis_key]})",
        },
        title=(
            f"Fig D: Topic Clusters on Sanctity vs. Loyalty Axes<br>"
            f"<sup>各话题在「{AXIS_LABEL_ZH[x_axis_key]}」与「{AXIS_LABEL_ZH[y_axis_key]}」上的分布（颜色=语言）</sup>"
        ),
    )
    fig_d.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig_d.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig_d.update_traces(textposition="top center", textfont_size=8)
    fig_d.update_layout(**_LAYOUT_BASE, height=650, legend=dict(title="Language"))

    p = DATA_DIR / "rq3_D_topic_lang_scatter.html"
    fig_d.write_html(str(p))
    inject_font(p)
    log.info(f"[STEP4] ✅ 图D: {p.name}")

    # ── 图E: Topic 共现网络（道德动机 × 语言占比节点图） ─────────────────────
    log.info("[STEP4] 图E: Topic 共现网络")
    _build_topic_network(bias_df, anova_df)

    # ── 图F: 语言 × 轴 Bias 分布小提琴图 ─────────────────────────────────
    log.info("[STEP4] 图F: 语言×轴 Bias 分布小提琴图")
    _build_violin_plot(bias_df, anova_df)

    # ── CSV 汇总（论文附录） ───────────────────────────────────────────────
    summary_rows = []
    for lang in ["en", "zh", "jp"]:
        lang_sub = bias_df[bias_df["lang"] == lang]
        if lang_sub.empty:
            continue
        for key in AXIS_KEYS:
            col = f"bias_{key}"
            if col not in lang_sub.columns:
                continue
            vals = lang_sub[col].dropna().values
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
                "median_bias":  round(float(np.median(vals)), 5),
                "q25_bias":     round(float(np.percentile(vals, 25)), 5),
                "q75_bias":     round(float(np.percentile(vals, 75)), 5),
                "most_negative_topic": most_negative_topic,
            })
    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(["lang", "mean_bias"])
    summary_df.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")
    log.info(f"[STEP4] ✅ 汇总CSV: {SUMMARY_CSV}")

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
            col = f"bias_{key}"
            mean_val = lang_means.loc[lang, col]
            bar = "▓" * int(abs(mean_val) * 500)
            direction = "←攻击" if mean_val < 0 else "→正向"
            print(f"    {AXIS_LABEL_EN[key]:20s}: {mean_val:+.4f} {direction} {bar}")
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
