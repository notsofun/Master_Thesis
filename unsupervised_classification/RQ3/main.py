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

理论轴（Moral Foundations Dictionary 2.0 + Intergroup Threat Lexicon）：
  1. Harm      ← Care (关爱) vs. Harm (伤害/虐待)
  2. Fairness  ← Fairness (公平) vs. Cheating (欺骗/剥削)
  3. Loyalty   ← Loyalty (忠诚/爱国) vs. Betrayal (背叛/渗透)
  4. Authority ← Authority (权威/法律) vs. Subversion (颠覆/叛乱)
  5. Sanctity  ← Sanctity (神圣/纯洁) vs. Degradation (堕落/病理化)
  6. RealThreat ← Safety (安全/稳定) vs. RealThreat (资源抢夺/政治控制)
  7. SymThreat  ← Cohesion (文化凝聚) vs. SymThreat (文化入侵/信仰崩塌)

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
  Step 4: 聚合可视化（47 Topic × 7 Axis 热力图、语言雷达图等）
          output: rq3_A_topic_axis_heatmap.html
                  rq3_B_lang_radar.html
                  rq3_C_lang_axis_bar.html
                  rq3_D_topic_lang_scatter.html
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
log, _ = setup_logging(name="RQ3_pipeline")


# ═══════════════════════════════════════════════════════════════════════════════
# 一、理论轴词典（MFD 2.0 + 群际威胁词典，多语言对齐）
# ═══════════════════════════════════════════════════════════════════════════════

# 格式: {axis_key: {"pos": [...], "neg": [...], "label_en": "...", "label_zh": "..."}}
# 正极 = 道德正向 / 安全侧；负极 = 道德负向 / 攻击侧
# 词汇覆盖中英日，保证跨语言 E5 质心对齐的合法性

MORAL_AXES: dict[str, dict] = {

    # ── Harm/Care axis（伤害/关爱） ──────────────────────────────────────────
    "harm": {
        "label_en": "Harm",
        "label_zh": "伤害轴",
        "label_desc": "Care(+) vs. Harm(−)",
        "pos": [
            # EN – care pole
            "care", "caring", "protect", "protection", "nurture", "nurturing",
            "safe", "safety", "compassion", "empathy", "kindness", "gentle",
            "heal", "healing", "support", "comfort", "welfare", "wellbeing",
            "guardian", "guardian", "mercy", "loving", "tender",
            # ZH – 关爱极
            "关爱", "保护", "呵护", "慈悲", "善待", "仁慈", "温柔",
            "救助", "关怀", "爱护", "养育", "同情", "怜悯", "安慰",
            # JA – 保護・ケア極
            "保護", "ケア", "優しい", "思いやり", "慈悲", "慈しみ",
            "守る", "助ける", "救う", "安全", "安心", "癒し",
        ],
        "neg": [
            # EN – harm pole
            "harm", "hurt", "abuse", "violence", "cruel", "cruelty",
            "torture", "damage", "injure", "injury", "bully", "oppress",
            "assault", "attack", "molest", "exploit", "wound", "beat",
            "terrorize", "traumatize", "brutal", "savage", "vicious",
            # ZH – 伤害极
            "伤害", "虐待", "暴力", "残忍", "折磨", "欺凌", "压迫",
            "攻击", "骚扰", "猥亵", "剥削", "创伤", "毒害", "毁灭",
            "摧残", "凌辱", "蹂躏",
            # JA – 危害極
            "危害", "暴力", "虐待", "残酷", "拷問", "いじめ", "傷つける",
            "攻撃", "搾取", "ハラスメント", "性的虐待",
        ],
    },

    # ── Fairness axis（公平/欺骗） ───────────────────────────────────────────
    "fairness": {
        "label_en": "Fairness",
        "label_zh": "公平轴",
        "label_desc": "Fairness(+) vs. Cheating(−)",
        "pos": [
            # EN
            "fair", "fairness", "justice", "equal", "equality", "rights",
            "honest", "honesty", "impartial", "unbiased", "equitable",
            "transparent", "accountability", "integrity", "legitimate",
            "proportional", "deserve", "merit",
            # ZH
            "公平", "正义", "平等", "公正", "权利", "诚实",
            "廉洁", "守信", "透明", "负责", "合法", "合理",
            # JA
            "公平", "正義", "平等", "誠実", "公正", "権利",
            "正直", "透明性", "責任", "合法", "公明正大",
        ],
        "neg": [
            # EN
            "cheat", "cheating", "fraud", "deception", "deceive", "lie",
            "scam", "exploit", "corrupt", "corruption", "bribe", "bribery",
            "manipulate", "manipulation", "swindle", "unfair", "biased",
            "rigged", "dishonest", "illegitimate", "hypocrisy",
            # ZH
            "欺骗", "欺诈", "诈骗", "腐败", "舞弊", "贿赂",
            "操控", "虚伪", "伪善", "不公", "偏袒", "歪曲",
            "蒙蔽", "骗取", "勾结", "黑幕",
            # JA
            "詐欺", "偽善", "汚職", "不正", "腐敗", "騙す",
            "操作", "不公平", "贈収賄", "欺く", "ごまかし",
        ],
    },

    # ── Loyalty axis（忠诚/背叛） ────────────────────────────────────────────
    "loyalty": {
        "label_en": "Loyalty",
        "label_zh": "忠诚轴",
        "label_desc": "Loyalty(+) vs. Betrayal(−)",
        "pos": [
            # EN
            "loyal", "loyalty", "patriot", "patriotism", "devoted", "devotion",
            "solidarity", "unity", "faithful", "allegiance", "cohesion",
            "teamwork", "commitment", "trustworthy", "dependable",
            # ZH
            "忠诚", "爱国", "团结", "凝聚", "奉献", "忠心",
            "义气", "同心", "一致", "守护", "坚守",
            # JA
            "忠誠", "愛国", "団結", "連帯", "献身", "信義",
            "一致", "守る", "絆", "仲間", "協力",
        ],
        "neg": [
            # EN
            "betray", "betrayal", "traitor", "treason", "infiltrate", "infiltration",
            "subvert", "sabotage", "defect", "defection", "backstab",
            "collude", "collusion", "turncoat", "disloyal", "treacherous",
            "sedition", "conspire", "conspiracy",
            # ZH
            "背叛", "叛徒", "叛国", "渗透", "颠覆", "勾结",
            "叛变", "出卖", "通敌", "内奸", "卧底", "阴谋",
            "破坏", "腐蚀", "侵蚀",
            # JA
            "裏切り", "反逆", "売国", "浸透", "転覆", "共謀",
            "内通", "スパイ", "陰謀", "破壊", "裏切る",
        ],
    },

    # ── Authority axis（权威/颠覆） ──────────────────────────────────────────
    "authority": {
        "label_en": "Authority",
        "label_zh": "权威轴",
        "label_desc": "Authority(+) vs. Subversion(−)",
        "pos": [
            # EN
            "authority", "order", "law", "discipline", "respect", "tradition",
            "hierarchy", "institution", "rule", "obedience", "structure",
            "stability", "convention", "orthodox", "legitimate authority",
            # ZH
            "权威", "秩序", "法律", "纪律", "尊重", "传统",
            "制度", "规则", "服从", "规范", "稳定", "正统",
            # JA
            "権威", "秩序", "法律", "規律", "尊重", "伝統",
            "制度", "ルール", "服従", "安定", "正統",
        ],
        "neg": [
            # EN
            "subvert", "subversion", "rebel", "rebellion", "defy", "defiance",
            "overthrow", "undermine", "anarchy", "chaos", "disorder",
            "radical", "extremist", "insurgent", "revolt", "sedition",
            "destabilize", "corrupt institution", "rogue",
            # ZH
            "颠覆", "叛乱", "反抗", "破坏秩序", "混乱", "无法无天",
            "激进", "极端", "推翻", "叛逆", "违法", "腐蚀制度",
            # JA
            "転覆", "反乱", "反抗", "無政府", "混乱", "過激",
            "急進", "制度破壊", "反体制", "反逆",
        ],
    },

    # ── Sanctity axis（圣洁/堕落） ───────────────────────────────────────────
    "sanctity": {
        "label_en": "Sanctity",
        "label_zh": "圣洁轴",
        "label_desc": "Sanctity(+) vs. Degradation(−)",
        "pos": [
            # EN
            "sacred", "pure", "holy", "sanctity", "divine", "righteous",
            "wholesome", "virtuous", "blessed", "reverence", "dignity",
            "spiritual", "noble", "honorable", "clean", "innocent",
            # ZH
            "神圣", "纯洁", "圣洁", "正直", "高尚", "虔诚",
            "尊严", "清白", "美德", "敬畏", "荣耀", "崇高",
            # JA
            "神聖", "清純", "清浄", "高潔", "崇高", "尊厳",
            "神聖な", "神々しい", "畏敬", "清い",
        ],
        "neg": [
            # EN
            "degrade", "degradation", "filth", "filthy", "corrupt", "pollution",
            "parasite", "pest", "vermin", "cancer", "disease", "virus",
            "disgusting", "perverse", "perverted", "degenerate", "immoral",
            "contaminate", "defile", "blaspheme", "heretic", "obscene",
            "grotesque", "toxic", "pathological",
            # ZH
            "堕落", "腐化", "肮脏", "寄生虫", "毒瘤", "害虫",
            "病毒", "败类", "恶魔", "渣滓", "邪恶", "变态",
            "龌龊", "下流", "异端", "亵渎",
            # JA
            "堕落", "汚染", "寄生虫", "害悪", "ゴキブリ", "ウイルス",
            "邪悪", "汚い", "卑劣", "腐敗", "不浄", "汚らわしい",
        ],
    },

    # ── Realistic Threat axis（现实威胁） ─────────────────────────────────────
    "realistic_threat": {
        "label_en": "Realistic Threat",
        "label_zh": "现实威胁轴",
        "label_desc": "Safety(+) vs. Realistic Threat(−)",
        "pos": [
            # EN – safety / stability pole
            "safe", "safety", "secure", "security", "stable", "stability",
            "sovereign", "sovereignty", "independent", "independence",
            "protected", "autonomy", "self-determination", "peaceful",
            # ZH
            "安全", "稳定", "主权", "独立", "自主", "和平",
            "保障", "领土", "秩序", "安宁",
            # JA
            "安全", "安定", "主権", "独立", "自主", "平和",
            "保障", "領土", "秩序",
        ],
        "neg": [
            # EN – realistic threat pole
            "invasion", "takeover", "control", "dominate", "dominance",
            "steal", "resource", "exploit resource", "political control",
            "infiltrate government", "lobby", "lobbying", "foreign influence",
            "colonize", "colonization", "occupy", "economic threat",
            "competition", "displacement", "replace", "replacement",
            # ZH
            "入侵", "控制", "夺权", "资源掠夺", "政治渗透",
            "游说", "外国势力", "殖民", "占领", "经济威胁",
            "势力扩张", "抢占", "侵占", "干政",
            # JA
            "侵略", "支配", "乗っ取り", "資源収奪", "政治工作",
            "ロビー活動", "外国勢力", "植民地化", "占領",
            "経済的脅威", "勢力拡大",
        ],
    },

    # ── Symbolic Threat axis（象征威胁） ──────────────────────────────────────
    "symbolic_threat": {
        "label_en": "Symbolic Threat",
        "label_zh": "象征威胁轴",
        "label_desc": "Cultural Cohesion(+) vs. Symbolic Threat(−)",
        "pos": [
            # EN – cultural cohesion pole
            "culture", "heritage", "tradition", "identity", "community",
            "shared value", "common ground", "unity", "cohesion", "civilization",
            "national identity", "cultural pride", "indigenous", "native",
            # ZH
            "文化", "传统", "认同", "共同体", "文明",
            "民族认同", "文化自信", "本土", "根源", "凝聚力",
            # JA
            "文化", "伝統", "アイデンティティ", "共同体", "文明",
            "民族", "文化的誇り", "固有", "国民性",
        ],
        "neg": [
            # EN – symbolic threat pole
            "brainwash", "indoctrinate", "cult", "sect", "heresy",
            "false belief", "superstition", "irrationality", "backward",
            "anti-intellectual", "pseudoscience", "delusion", "fanaticism",
            "cultural invasion", "westernize", "erode culture",
            "destroy tradition", "foreign religion", "alien belief",
            # ZH
            "洗脑", "邪教", "迷信", "愚昧", "落后", "反智",
            "文化入侵", "侵蚀", "破坏传统", "外来宗教",
            "异端", "蛊惑", "盲目崇拜",
            # JA
            "洗脳", "カルト", "迷信", "非合理", "反知性",
            "文化侵略", "伝統破壊", "外来宗教", "狂信",
            "マインドコントロール", "妄信",
        ],
    },
}

# 轴名列表（保持顺序，与可视化顺序一致）
AXIS_KEYS = list(MORAL_AXES.keys())

# 双语标签映射（用于图表轴标签）
AXIS_LABEL_EN = {k: v["label_en"] for k, v in MORAL_AXES.items()}
AXIS_LABEL_ZH = {k: v["label_zh"] for k, v in MORAL_AXES.items()}
AXIS_LABEL_DESC = {k: v["label_desc"] for k, v in MORAL_AXES.items()}

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

    axis_vectors: dict[str, np.ndarray] = {}

    for key in tqdm(AXIS_KEYS, desc="构建轴向量", unit="轴"):
        axis_def = MORAL_AXES[key]
        pos_words = axis_def["pos"]
        neg_words = axis_def["neg"]

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
