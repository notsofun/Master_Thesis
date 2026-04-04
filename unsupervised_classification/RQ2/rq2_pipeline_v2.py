"""
python unsupervised_classification/RQ2/rq2_pipeline_v2.py
python unsupervised_classification/RQ2/rq2_pipeline_v2.py --from-raw
RQ2 分析管线 v2.2 — 分语言、分话题的仇恨表达方式分析
====================================================

核心设计哲学（基于老师意见 + 样本观察）
─────────────────────────────────────────
老师指出"要注意多样化的句式，有的句子没有主语没有宾语"。

样本观察（真实语料特点）：
  中文: 大量零主语句、话题链、感叹句 ("真是丧尽天良的毒瘤！")
  日文: 省略主语极为普遍；谓语动词在句尾，依存分析误差大
  英文: 命令句(no subject)、切割句、推特体碎片较多

因此，本管线放弃纯依存SVO作为唯一手段，改用三层互补策略：
  Layer 1: 依存SVO（有主宾的完整句）
  Layer 2: 谓词窗口法（残缺句：target附近±N词内的行为词）
  Layer 3: LLM归因（对Layer1+2结果统一做框架分类）

"分语言+分话题" 维度在数据聚合阶段实现，不在提取阶段区分。

Gemini调用设计：
  - 使用项目统一的 google.genai SDK（同 data_augmentation/LLM/google_api.py）
  - APIRequester：令牌桶限流 + 动态并发 + 指数退避重试（完整并发控制）
  - 通过 asyncio.run() 驱动异步批量请求
  - 断点续传缓存（每批后立即写盘）

日志：
  - 使用项目统一的 scripts/set_logger.py → setup_logging()
  - 自动在脚本同级 logs/ 目录创建日志文件

可视化：
  - 中日文字体：Plotly HTML 内嵌 Google Fonts (Noto Sans SC / Noto Sans JP)
  - 每个图表的 x/y 轴标签同时显示中文+英文翻译
  - 输出为自包含 HTML，在浏览器直接查看无需额外依赖

输出策略（数据与可视化分离）：
  rq2_raw_extractions.csv    ← Step1 完成后即保存（原始三元组）
  rq2_framing_labeled.csv    ← Step2 完成后即保存（含frame_type）
  rq2_framing_cache.json     ← Gemini缓存（断点续传）
  rq2_aggregated_summary.csv ← Step3 汇总表（论文附录）
  rq2_A_topic_frame_heatmap.html
  rq2_B_lang_frame_bar.html
  rq2_C_target_frame_matrix.html
  rq2_D_sunburst_lang_topic_frame.html
  rq2_E_top_verbs_[en/zh/jp].csv

可视化重跑（无需重新调API）：
  python rq2_pipeline_v2.py --viz-only

可视化重跑（无需重新调API）：
  python rq2_pipeline_v2.py --reclassify-other  ← 只对 other 重分类，不重跑 Step1

框架类型（10类，v2.3新增 social_exclusion + rhetorical_attack）：
  cognitive_manipulation  认知操纵/洗脑
  moral_accusation        道德指控/伪善
  dehumanization          非人化/病理化
  sexual_abuse_frame      性侵虐待指控
  economic_exploitation   经济剥削
  political_interference  政治干预
  institutional_rot       制度性腐败
  social_exclusion        社会排斥/逐出  (v2.3新增)
  rhetorical_attack       修辞攻击/讽刺  (v2.3新增)
  other                   其他
  [noise]                 真噪声（过滤，不进图表）
"""

import os
import sys
import json
import re
import asyncio
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm
from dotenv import load_dotenv

# ── 路径配置 ──────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent.parent
DOC_PATH = ROOT / "unsupervised_classification/topic_modeling_results/sixth/data/document_topic_mapping.csv"
RQ1_PATH = ROOT / "unsupervised_classification/RQ1/data/rq1_topic_targets_v3.csv"
OUT_DIR  = ROOT / "unsupervised_classification/RQ2/data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 日志：使用项目统一的 set_logger ──────────────────────────────────────────
sys.path.insert(0, str(ROOT))
from scripts.set_logger import setup_logging
log, LOG_PATH = setup_logging("rq2_pipeline")
log.info(f"日志文件: {LOG_PATH}")

# ── Gemini 并发管理：复用项目统一的 APIRequester ─────────────────────────────
sys.path.insert(0, str(ROOT))
from data_augmentation.LLM.google_api import APIRequester

load_dotenv()
# ── 框架定义（中英双语标签） ──────────────────────────────────────────────────
FRAME_TYPES = {
    "cognitive_manipulation": "认知操纵/洗脑\nCognitive Manipulation",
    "moral_accusation":       "道德指控/伪善\nMoral Accusation",
    "dehumanization":         "非人化/病理化\nDehumanization",
    "sexual_abuse_frame":     "性侵虐待指控\nSexual Abuse Frame",
    "economic_exploitation":  "经济剥削\nEconomic Exploitation",
    "political_interference": "政治干预\nPolitical Interference",
    "institutional_rot":      "制度性腐败\nInstitutional Rot",
    "other":                  "其他\nOther",
}

# 纯 key → 短标签（用于图表轴标签，两行：中文 + EN）
FRAME_LABEL_BILINGUAL = {k: v for k, v in FRAME_TYPES.items()}

# key → 英文全称（图表主标签）
FRAME_EN = {
    "cognitive_manipulation": "Cognitive Manipulation",
    "moral_accusation":       "Moral Accusation",
    "dehumanization":         "Dehumanization",
    "sexual_abuse_frame":     "Sexual Abuse / Cover-up",
    "economic_exploitation":  "Economic Exploitation",
    "political_interference": "Political Interference",
    "institutional_rot":      "Institutional Rot",
    # ── v2.3 新增两类，覆盖原 other 中的漏网案例 ──────────────────
    "social_exclusion":       "Social Exclusion",      # 拒绝圣礼/开除/排斥
    "rhetorical_attack":      "Rhetorical Attack",     # 讽刺/批判/质疑（不含明确暴力）
    # ── 真噪声标签（不出现在图表中，仅用于过滤） ─────────────────
    "noise":                  "[noise]",
    "other":                  "Other / Unclassified",
}

# key → 中日文对照（CSV 人读 + 图表副标签保留原文）
FRAME_ZH = {
    "cognitive_manipulation": "认知操纵/洗脑",
    "moral_accusation":       "道德指控/伪善",
    "dehumanization":         "非人化/病理化",
    "sexual_abuse_frame":     "性侵虐待/掩盖",
    "economic_exploitation":  "经济剥削",
    "political_interference": "政治干预",
    "institutional_rot":      "制度性腐败",
    "social_exclusion":       "社会排斥/逐出",
    "rhetorical_attack":      "修辞攻击/讽刺批判",
    "noise":                  "[噪声-不展示]",
    "other":                  "其他",
}

# 仅用于图表展示的有效框架（排除 noise）
FRAME_KEYS = [k for k in FRAME_EN if k != "noise"]

# ── 停用动词 ──────────────────────────────────────────────────────────────────
STOP_VERBS = {
    "be","is","are","was","were","been","being","have","has","had",
    "do","does","did","say","said","know","think","get","go","make",
    "see","want","need","come","look","feel","seem","become","keep",
    "give","take","use","find","tell","ask","show","leave","let",
    "是","有","说","做","去","想","看","让","把","被","了","在",
    "可以","应该","就是","觉得","知道","认为","表示","进行",
    "する","いる","ある","なる","いう","れる","られる","です","だ",
    "思う","言う","見る","来る","行く","知る","できる",
}

# ── 规则框架分类器（无API时fallback） ────────────────────────────────────────
FRAME_RULES = {
    "cognitive_manipulation": [
        "brainwash","manipulate","deceive","indoctrinat","cult","洗脑","欺骗","操控",
        "骗","蒙蔽","蛊惑","愚弄","騙す","洗脳","マインドコントロール","狂信",
        "mislead","delude","程序","蒙骗","诱导","信仰足","心理控制",
    ],
    "moral_accusation": [
        "hypocrit","corrupt","liar","fraud","cheat","immoral","sinful","伪善","虚伪",
        "腐败","堕落","无耻","丧尽天良","偽善","腐敗","汚職","不正",
        "讽刺","阴阳怪气","道貌岸然","邪恶","肮脏","罪孽","shame","disgrace",
        "condemn","condemning","骂","批","抨击","指责",
    ],
    "dehumanization": [
        "parasite","pest","virus","vermin","cancer","monster","predator","toxic",
        "毒瘤","寄生虫","害虫","病毒","恶魔","ゴキブリ","害悪","癌",
        "garbage","trash","filth","邪教","カルト","cult","异端","heretic",
    ],
    "sexual_abuse_frame": [
        "abuse","molest","rape","pedophil","assault","child","victim","scandal",
        "猥亵","性侵","儿童","受害者","丑闻","强奸","恋童","痴漢","性的虐待","児童",
        "封口","掩盖","隐瞒","揉み消す","cover up","grooming","harass",
        "修女","大腿","性","搭上",  # 覆盖中文语料里的隐晦描述
    ],
    "economic_exploitation": [
        "extort","scam","donation","献金","敛财","骗钱","封口费",
        "搜刮","诈骗","お布施","搾取","詐欺","霊感商法","money","财","钱",
        "collect","fund","collect money","金もらう","もらう","金",
    ],
    "political_interference": [
        "election","vote","politic","lobby","自民党","选举","干政","渗透",
        "政党","国会","議員","選挙","政権","government","parliament",
        "政治家","安倍","バックアップ","选票","操纵","政权","影响力",
    ],
    "institutional_rot": [
        "institution","systemic","cover","protect","silence","impunity",
        "体制","官官相护","系统性","掩盖","包庇","教廷","バチカン",
        "rotten","rot","组织腐败","罗马","hierarchy","diocese","bishop",
        "任命","调到","management","ran","run the","protect abuser",
        "烂到根","官官","制度","组织",
    ],
    # ── v2.3 新增 ─────────────────────────────────────────────────────────
    "social_exclusion": [
        "excommunicat","ban","forbid","expel","reject","deny","refuse",
        "开除","逐出","排斥","禁止","不准","拒绝","离开","驱逐",
        "禁じ","排除","拒否","除名","追放",
        "kick out","bar from","denied sacrament","圣餐","洗礼","天堂",
    ],
    "rhetorical_attack": [
        "criticiz","mock","ridicul","sarcas","condemn","challenge","attack",
        "讽刺","嘲","批评","质疑","反驳","驳斥","揭露","揭批","戳穿",
        "嘲笑","批判","arguement","argue","debunk","absurd","ridiculous",
        "皮肉","やり玉","批判","糾弾","非難","おかしい","おかし",
    ],
}

# 真实噪声：语义上完全中性的动词，与仇恨表达无关
_NOISE_VERBS = {
    # EN
    "believe","agree","hope","hear","care","wonder","speak","read",
    "pray","thank","write","mean","remember","understand","consider","suggest",
    "wait","love","trust","marry","value","think","know","want","feel",
    "seem","become","set","arrive","attend","bless","bow","choose","check",
    "build","bring","belong","bear","appear","appreciate","assure","answer",
    "abide","accompany","acknowledge","admire","address","allow","back",
    "call","can","follow","forgive","grow","help","include","join","keep",
    "leave","listen","live","meet","miss","note","observe","participate",
    "protect","provide","pursue","raise","reach","receive","remain","respect",
    "return","share","stand","stay","stop","study","support","teach","tell",
    "try","turn","visit","walk","watch","welcome","wish","work","write",
    # ZH 中性动词
    "变成","转转","一样","待人接物","确认","禁止","念叨","得到","需要",
    "问问","找来","代行","开放","要求","想起","告诉","允许","完成",
    "离开","跑光","支持","核心","对应","正确","叫爽","合着","献给",
    "牺牲","离婚","设立","哭泣","独身制","命令","融合","报告",
    "分裂","引导","承受","正视","忠于","管理","替代","健健康康",
    # JP 中性动词
    "知ら","なっ","やっ","作っ","あっ","なり","あり","活動","学べ",
    "譲歩","呼ば","信仰","もらう","認め","バックアップ",
}

def rule_classify(text: str) -> str:
    """规则分类器：先检测噪声，再匹配框架关键词。"""
    t = text.lower().strip()
    # 纯噪声：短于3字符，或命中中性动词表
    if len(t) <= 2 or t in _NOISE_VERBS:
        return "noise"
    for frame, kws in FRAME_RULES.items():
        if any(kw in t for kw in kws):
            return frame
    return "other"


# ═══════════════════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════════════════

def load_target_vocab(rq1_csv: Path) -> dict[int, list[str]]:
    """从 RQ1 结果加载每个 topic 的 top-5 靶子词（小写）。"""
    df = pd.read_csv(rq1_csv)
    vocab = {}
    for _, row in df.iterrows():
        tid = int(row["Topic_ID"])
        raw = re.findall(r"'([^']+)\(\d+\)'", str(row["Top_Targets"]))
        vocab[tid] = [t.strip().lower() for t in raw[:5] if len(t.strip()) >= 2]
    log.info(f"[VOCAB] 加载 {len(vocab)} 个 topic 的 target 词表")
    return vocab


CACHE_PATH = OUT_DIR / "rq2_framing_cache.json"

def load_cache() -> dict:
    if CACHE_PATH.exists():
        with open(CACHE_PATH, encoding="utf-8") as f:
            data = json.load(f)
        log.info(f"[CACHE] 加载已有缓存 {len(data)} 条")
        return data
    return {}

def save_cache(cache: dict):
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    log.debug(f"[CACHE] 缓存已写盘 {len(cache)} 条")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1：三层互补提取
# ═══════════════════════════════════════════════════════════════════════════════

def try_load_spacy() -> dict | None:
    try:
        import spacy
        models = {
            "en": spacy.load("en_core_web_sm"),
            "zh": spacy.load("zh_core_web_sm"),
            "jp": spacy.load("ja_core_news_sm"),
        }
        log.info("[SPACY] 多语言模型加载成功")
        return models
    except Exception as e:
        log.warning(f"[SPACY] 加载失败（{e}），将使用 Layer2 窗口法")
        return None


def layer1_svo(text: str, lang: str, target_vocab: list[str], nlp_map: dict) -> list[dict]:
    """
    Layer 1: 依存SVO精准提取（适用于有完整主谓宾的句子）。
    对残缺句（无nsubj/obj）会自然返回空列表，由Layer2补充。
    """
    nlp = nlp_map.get(lang, nlp_map.get("en"))
    if nlp is None:
        return []
    if lang == "jp" and len(text.encode("utf-8")) > 48000:
        text = text.encode("utf-8")[:48000].decode("utf-8", errors="ignore")
    try:
        doc = nlp(text)
    except Exception as e:
        log.debug(f"[L1] spaCy解析异常: {e}")
        return []

    text_lower = text.lower()
    present_targets = [t for t in target_vocab if t in text_lower]
    if not present_targets:
        return []

    results = []
    for token in doc:
        if token.pos_ not in ("VERB", "AUX"):
            continue
        verb = (token.lemma_ if lang == "en" else token.text).lower().strip()
        if verb in STOP_VERBS or len(verb) <= 1:
            continue

        for child in token.children:
            child_lower = child.text.lower()
            is_subj = child.dep_ in ("nsubj", "nsubjpass", "csubj", "top")
            is_obj  = child.dep_ in ("obj", "dobj", "iobj", "pobj", "obl", "nmod")
            if not (is_subj or is_obj):
                continue
            matched = next((t for t in present_targets
                            if t in child_lower or child_lower in t), None)
            if not matched:
                continue
            extras = [c.text.lower() for c in token.children
                      if c.dep_ in ("obj","dobj","attr","acomp","xcomp")
                      and c.text != child.text][:1]
            predicate = f"{verb} {extras[0]}" if extras else verb
            results.append({
                "layer": "svo", "target": matched, "verb": verb,
                "predicate": predicate,
                "role": "agent" if is_subj else "patient",
                "context": text[:150],
            })
    return results


# 各语言行为词提取正则（Layer2 窗口法专用）
_ZH_ACT = re.compile(
    r'(?:洗脑|欺骗|骗取|猥亵|性侵|敛财|掩盖|包庇|操控|虐待|剥削|腐败|堕落|丑闻|'
    r'诈骗|官官相护|伪善|毒瘤|揭露|献金|封口|渗透|强奸|扭曲|蒙蔽|蛊惑|凌辱|压迫|剥夺)'
)
_JP_ACT = re.compile(
    r'(?:洗脳|騙す|詐欺|性的虐待|児童|献金|隠蔽|揉み消す|腐敗|支配|搾取|狂信|'
    r'マインドコントロール|ゴキブリ|害悪|汚職|不正|裏切|虐待|欺く|強要|恐喝)'
)
_EN_ACT = re.compile(
    r'\b(?:abuse|molest|exploit|manipulat|deceiv|brainwash|corrupt|extort|'
    r'coverup|cover.up|rape|assault|indoctrinat|pedophil|defraud|scam|'
    r'oppress|silence|victimiz|harass|coerce|traffick|groom)\w*\b',
    re.IGNORECASE
)

def extract_action_words(tokens: list[str], lang: str) -> list[str]:
    text = " ".join(tokens)
    if lang == "zh":
        return _ZH_ACT.findall(text)
    elif lang == "jp":
        return _JP_ACT.findall(text)
    else:
        return _EN_ACT.findall(text)


def layer2_window(text: str, lang: str, target_vocab: list[str]) -> list[dict]:
    """
    Layer 2: 目标词附近 ±8 token 窗口法。
    处理省略主语/宾语的残缺句式——不依赖句法树，直接在
    target 词周围寻找语义行为词（来自各语言专属词典）。
    role 标记为 'unknown'，因为无法从残缺句判断施/受动关系。
    """
    WINDOW = 8
    if lang == "zh":
        tokens = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+', text)
    elif lang == "jp":
        tokens = re.findall(r'[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]+|[a-zA-Z]+', text)
    else:
        tokens = re.findall(r"[a-zA-Z']+", text.lower())

    text_lower = text.lower()
    present_targets = [t for t in target_vocab if t in text_lower]
    if not present_targets:
        return []

    results = []
    tokens_lower = [t.lower() for t in tokens]

    for target in present_targets:
        positions = [i for i, tok in enumerate(tokens_lower)
                     if target in tok or tok in target]
        for pos in positions:
            window_tokens = tokens_lower[max(0, pos-WINDOW): pos+WINDOW+1]
            action_words  = extract_action_words(window_tokens, lang)
            ctx_start = max(0, text.lower().find(target) - 30)
            for word in action_words:
                results.append({
                    "layer": "window", "target": target, "verb": word,
                    "predicate": word, "role": "unknown",
                    "context": text[ctx_start: ctx_start+150],
                })
    return results


def extract_expressions(text: str, lang: str, target_vocab: list[str],
                         nlp_map: dict | None) -> list[dict]:
    """三层互补提取入口，(target, verb) 去重，Layer1 优先。"""
    results = []
    if nlp_map:
        results.extend(layer1_svo(text, lang, target_vocab, nlp_map))
    results.extend(layer2_window(text, lang, target_vocab))

    seen: dict = {}
    deduped: list = []
    for r in results:
        key = (r["target"], r["verb"])
        if key not in seen:
            seen[key] = r
            deduped.append(r)
        elif r["layer"] == "svo" and seen[key]["layer"] != "svo":
            seen[key] = r
            deduped = [r if (x["target"], x["verb"]) == key else x for x in deduped]
    return deduped


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2：Gemini 框架归因（异步并发 + APIRequester + 缓存）
# ═══════════════════════════════════════════════════════════════════════════════

def _build_classify_prompt(batch: list[str]) -> str:
    frame_list = "\n".join(f"  {k}: {FRAME_EN[k]} / {FRAME_ZH[k]}" for k in FRAME_KEYS)
    numbered   = "\n".join(f"{i+1}. {p}" for i, p in enumerate(batch))
    return f"""You are classifying hate speech rhetoric patterns in a multilingual study
(Chinese / Japanese / English) about religious hate speech.

For each predicate/verb phrase below, choose the SINGLE best framing type from the list:
{frame_list}

Classification hints:
- cognitive_manipulation → cult brainwashing, deceiving/indoctrinating believers (洗脑/欺骗/蛊惑)
- moral_accusation       → hypocrisy, corruption, moral condemnation (伪善/腐败/指责)
- dehumanization         → calling groups parasites, cancer, pests (毒瘤/寄生虫/ゴキブリ)
- sexual_abuse_frame     → child abuse, molestation, cover-up of clergy abuse (性侵/猥亵/儿童/揉み消す)
- economic_exploitation  → coerced donations, financial fraud against followers (献金/敛财/霊感商法)
- political_interference → lobbying, election manipulation, political infiltration (选举/干政/渗透)
- institutional_rot      → systemic cover-up, protecting perpetrators, hierarchy complicity (包庇/组织腐败)
- social_exclusion       → excommunication, banning, expelling, denying sacraments (开除/逐出/排除)
- rhetorical_attack      → mocking, sarcasm, criticism, debunking (讽刺/批判/嘲笑/糾弾)
- other                  → ONLY use when the predicate clearly fits none of the above

IMPORTANT: Prefer a specific frame over "other". Short verbs in CJK languages often carry
strong frame signals — check the context carefully before choosing "other".

Predicates to classify:
{numbered}

Reply ONLY with compact JSON mapping index to frame key.
Example: {{"1":"moral_accusation","2":"dehumanization","3":"rhetorical_attack"}}
Valid keys: {FRAME_KEYS}"""


async def classify_frames_async(predicates: list[str], cache: dict,
                                 api_key: str, raw_df: pd.DataFrame) -> dict[str, str]:
    """
    使用项目统一的 APIRequester（令牌桶限流 + 动态并发 + 指数退避）。

    改进（v2.3）：
    ① 逐批串行处理（保证进度条准确 + 时间估算）
    ② 每批完成后立即将本批行 append 写入 rq2_framing_labeled.csv
       （中途崩溃重启后 labeled CSV 已有真实结果，不只是 key）
    ③ tqdm 进度条显示：已完成批次、剩余、预计完成时间
    """
    import time
    from google import genai as google_genai

    client    = google_genai.Client(api_key=api_key)
    requester = APIRequester(client, model="gemini-2.5-flash", max_retries=5)

    to_do = [p for p in predicates if p not in cache]
    log.info(f"[GEMINI] 需分类 {len(to_do)} 条，已缓存 {len(predicates)-len(to_do)} 条")

    if not to_do:
        return {p: cache.get(p, "other") for p in predicates}

    BATCH    = 40
    batches  = [to_do[i:i+BATCH] for i in range(0, len(to_do), BATCH)]
    n_total  = len(to_do)
    n_done   = 0
    t_start  = time.time()

    LABELED_PATH = OUT_DIR / "rq2_framing_labeled.csv"
    # 如果是续跑，labeled CSV 可能已有部分行；先检查写入模式
    labeled_header_written = LABELED_PATH.exists()

    log.info(f"[GEMINI] 共 {len(batches)} 个批次，逐批串行（进度可见）...")

    # tqdm 进度条：单位是"条谓语"，方便估算剩余时间
    pbar = tqdm(total=n_total, desc="Gemini分类", unit="pred",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

    for batch_idx, batch in enumerate(batches):
        prompt    = _build_classify_prompt(batch)
        task_name = f"RQ2-Batch-{batch_idx+1}/{len(batches)}"
        batch_result: dict[str, str] = {}

        try:
            raw = await requester.request_async(prompt, temperature=0.1,
                                                task_name=task_name)
            m = re.search(r'\{[^{}]+\}', raw, re.DOTALL)
            if m:
                parsed = json.loads(m.group())
                for i, pred in enumerate(batch):
                    label = parsed.get(str(i+1), "other")
                    label = label if label in FRAME_ZH else "other"
                    cache[pred]        = label
                    batch_result[pred] = label
                log.info(f"[GEMINI] {task_name}: {len(batch)} 条完成")
            else:
                log.warning(f"[GEMINI] {task_name} JSON解析失败: {raw[:80]}")
                for pred in batch:
                    label              = rule_classify(pred)
                    cache[pred]        = label
                    batch_result[pred] = label
        except Exception as e:
            log.error(f"[GEMINI] {task_name} 最终失败({e})，回退规则分类")
            for pred in batch:
                label              = rule_classify(pred)
                cache[pred]        = label
                batch_result[pred] = label

        # ── 断点续传①：更新 key-label 缓存 ──────────────────────────────
        save_cache(cache)

        # ── 断点续传②：把本批对应的原始行（含frame_type）append写入CSV ──
        # 找出 raw_df 中 predicate 属于本批的所有行
        batch_mask = raw_df["predicate"].isin(batch_result.keys())
        batch_rows = raw_df[batch_mask].copy()
        batch_rows["frame_type"] = batch_rows["predicate"].map(batch_result)
        batch_rows.to_csv(
            LABELED_PATH,
            mode="a",                              # append
            header=not labeled_header_written,     # 第一批写 header
            index=False,
            encoding="utf-8-sig",
        )
        labeled_header_written = True              # 之后的批次不再写 header

        # ── 进度更新 ─────────────────────────────────────────────────────
        n_done += len(batch)
        pbar.update(len(batch))

        elapsed   = time.time() - t_start
        rate      = n_done / elapsed if elapsed > 0 else 1
        remaining = (n_total - n_done) / rate if rate > 0 else 0
        log.info(
            f"[GEMINI] 进度 {n_done}/{n_total} "
            f"({n_done/n_total*100:.1f}%) | "
            f"已用 {elapsed/60:.1f}min | "
            f"预计剩余 {remaining/60:.1f}min"
        )

    pbar.close()
    save_cache(cache)
    stats = requester.concurrency_manager.get_stats()
    log.info(f"[GEMINI] 完成统计: {stats}")
    return {p: cache.get(p, "other") for p in predicates}


def classify_frames_gemini(predicates: list[str], cache: dict,
                            api_key: str, raw_df: pd.DataFrame) -> dict[str, str]:
    """同步入口，内部用 asyncio.run() 驱动异步逐批请求。"""
    return asyncio.run(classify_frames_async(predicates, cache, api_key, raw_df))


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3：聚合分析 + 可视化（支持中日文字体 + 双语标签）
# ═══════════════════════════════════════════════════════════════════════════════

# 中日文字体注入片段（嵌入每个 HTML 的 <head> 中）
_FONT_CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;700&family=Noto+Sans+JP:wght@400;700&display=swap');
  body, .plotly-graph-div, .gtitle, .xtitle, .ytitle, text {
    font-family: 'Noto Sans SC', 'Noto Sans JP', 'Hiragino Sans', 'Microsoft YaHei',
                 'Meiryo', Arial, sans-serif !important;
  }
</style>
"""

def inject_font(html_path: Path):
    """向已生成的 HTML 文件注入中日文字体 CSS。"""
    content = html_path.read_text(encoding="utf-8")
    if "Noto Sans SC" not in content:
        content = content.replace("<head>", f"<head>{_FONT_CSS}", 1)
        html_path.write_text(content, encoding="utf-8")
        log.debug(f"[FONT] 字体注入完成: {html_path.name}")


def frame_bilingual_label(key: str) -> str:
    """
    英文为主、中文为副的双语轴标签。
    格式：English Label\n(中文)
    论文图表以英文为主，保留中文供对照。
    """
    en = FRAME_EN.get(key, key)
    zh = FRAME_ZH.get(key, "")
    if zh and zh != en:
        return f"{en}<br><sup>({zh})</sup>"
    return en


LANG_LABEL = {"en": "English 🇬🇧", "zh": "中文 🇨🇳", "jp": "日本語 🇯🇵"}
LANG_COLOR = {"en": "#4C78A8", "zh": "#F58518", "jp": "#54A24B"}

_LAYOUT_BASE = dict(
    font=dict(
        family="'Noto Sans SC','Noto Sans JP','Hiragino Sans','Microsoft YaHei',Arial,sans-serif",
        size=12,
    ),
    paper_bgcolor="white",
    plot_bgcolor="white",
)


def aggregate_and_visualize(labeled_df: pd.DataFrame):
    """
    ── 如何看这五张图 ──────────────────────────────────────────────────
    图A  Topic × Frame 热力图（百分比）
         每行=一个Topic，深色格=该Topic主导攻击框架。
         → 直接读：Topic 0 最深的是哪列 = 该话题最常用的攻击手法。

    图B  三语言框架分布条形图（各语言内百分比）
         同一框架类型的三根柱子横向对比三种语言。
         → 哪种语言的哪根柱最高 = 该语言偏好的攻击框架。
         预期：英文→institutional_rot；中文→moral_accusation；日文→economic_exploitation

    图C  Target × Frame 矩阵（原始频次）
         行=被攻击target，列=框架，颜色越深=受此框架攻击越多。
         → 配合 RQ1 结论直接读："神父"一行最红的格 = 神父遭受的主要攻击手法。

    图D  三层旭日图（交互式）lang → topic → frame
         点击语言区块→钻入各topic→看框架占比。
         → 用于演示/探索，适合跟老师会议时展示。

    图E  分语言 Top 动词 CSV（每框架最高频原始词）
         → 验证框架归因是否准确；可挑具体例词放论文正文。
    ────────────────────────────────────────────────────────────────────
    """
    import plotly.graph_objects as go
    import plotly.express as px

    log.info(f"[VIZ] 开始生成可视化，共 {len(labeled_df)} 条记录")
    log.info(f"[VIZ] 语言分布: {labeled_df['lang'].value_counts().to_dict()}")
    log.info(f"[VIZ] 框架分布（含noise）: {labeled_df['frame_type'].value_counts().to_dict()}")
    log.info(f"[VIZ] 提取层级: {labeled_df['layer'].value_counts().to_dict()}")

    # ── 过滤噪声（noise 不参与可视化，other 保留但排在末尾） ─────────────
    noise_count = (labeled_df["frame_type"] == "noise").sum()
    if noise_count:
        log.info(f"[VIZ] 过滤 {noise_count} 条 noise 标签（占{noise_count/len(labeled_df)*100:.1f}%），不进入图表")
    labeled_df = labeled_df[labeled_df["frame_type"] != "noise"].copy()

    labeled_df["frame_en"]   = labeled_df["frame_type"].map(FRAME_EN)
    labeled_df["frame_zh"]   = labeled_df["frame_type"].map(FRAME_ZH)
    labeled_df["lang_label"] = labeled_df["lang"].map(LANG_LABEL)

    # 双语轴标签列表（顺序与 FRAME_KEYS 一致）
    bilingual_x = [frame_bilingual_label(k) for k in FRAME_KEYS]

    # ── 图A: Topic × Frame 热力图 ──────────────────────────────────────
    top20_topics = labeled_df["topic"].value_counts().head(20).index
    tf = (labeled_df[labeled_df["topic"].isin(top20_topics)]
          .groupby(["topic", "frame_type"]).size()
          .reset_index(name="count"))
    tf["pct"] = (tf["count"] / tf.groupby("topic")["count"].transform("sum") * 100).round(1)
    tf_pivot = (tf.pivot(index="topic", columns="frame_type", values="pct")
                  .reindex(columns=FRAME_KEYS, fill_value=0).fillna(0))

    fig_a = go.Figure(go.Heatmap(
        z=tf_pivot.values,
        x=bilingual_x,
        y=[f"Topic {t}" for t in tf_pivot.index],
        colorscale="Blues",
        text=tf_pivot.values.round(1),
        texttemplate="%{text}%",
        colorbar=dict(title="占比 %"),
        hovertemplate="Topic %{y}<br>框架: %{x}<br>占比: %{z:.1f}%<extra></extra>",
    ))
    fig_a.update_layout(
        **_LAYOUT_BASE,
        title=dict(text="Fig A: Dominant Attack Framing per Topic (% within topic)<br><sup>各话题主导攻击框架热力图</sup>",
                   font=dict(size=14)),
        xaxis=dict(title="Framing Type (攻击框架类型)", tickangle=-20),
        yaxis=dict(title="Topic"),
        height=700,
    )
    p = OUT_DIR / "rq2_A_topic_frame_heatmap.html"
    fig_a.write_html(str(p))
    inject_font(p)
    log.info(f"[VIZ] ✅ 图A: {p.name}")

    # ── 图B: 三语框架分布条形图（% within lang） ───────────────────────
    lf = (labeled_df.groupby(["lang", "frame_type"]).size()
          .reset_index(name="count"))
    lf["pct"] = (lf["count"] / lf.groupby("lang")["count"].transform("sum") * 100).round(1)
    lf["frame_zh"]   = lf["frame_type"].map(FRAME_ZH)
    lf["lang_label"] = lf["lang"].map(LANG_LABEL)

    # 保证 x 轴顺序
    lf["frame_order"] = lf["frame_type"].map({k: i for i, k in enumerate(FRAME_KEYS)})
    lf = lf.sort_values("frame_order")

    fig_b = go.Figure()
    for lang in ["en", "zh", "jp"]:
        sub = lf[lf["lang"] == lang]
        fig_b.add_trace(go.Bar(
            name=LANG_LABEL.get(lang, lang),
            x=[frame_bilingual_label(k) for k in sub["frame_type"]],
            y=sub["pct"],
            marker_color=LANG_COLOR[lang],
            hovertemplate=f"{LANG_LABEL.get(lang,lang)}<br>框架: %{{x}}<br>占比: %{{y:.1f}}%<extra></extra>",
        ))
    fig_b.update_layout(
        **_LAYOUT_BASE,
        barmode="group",
        title=dict(text="Fig B: Attack Framing Distribution by Language (% within language)<br><sup>三语言攻击框架分布对比</sup>",
                   font=dict(size=14)),
        xaxis=dict(title="Framing Type (攻击框架类型)", tickangle=-20),
        yaxis=dict(title="% within Language (各语言内占比)"),
        legend=dict(title="Language"),
        height=520,
    )
    p = OUT_DIR / "rq2_B_lang_frame_bar.html"
    fig_b.write_html(str(p))
    inject_font(p)
    log.info(f"[VIZ] ✅ 图B: {p.name}")

    # ── 图C: Target × Frame 矩阵（原始频次） ───────────────────────────
    top15 = labeled_df["target"].value_counts().head(15).index
    tgf = (labeled_df[labeled_df["target"].isin(top15)]
           .groupby(["target", "frame_type"]).size()
           .reset_index(name="count"))
    tgf_pivot = (tgf.pivot(index="target", columns="frame_type", values="count")
                    .reindex(columns=FRAME_KEYS, fill_value=0).fillna(0))

    fig_c = go.Figure(go.Heatmap(
        z=tgf_pivot.values,
        x=bilingual_x,
        y=tgf_pivot.index.tolist(),
        colorscale="Reds",
        text=tgf_pivot.values.astype(int),
        texttemplate="%{text}",
        colorbar=dict(title="频次 / Count"),
        hovertemplate="Target: %{y}<br>框架: %{x}<br>频次: %{z}<extra></extra>",
    ))
    fig_c.update_layout(
        **_LAYOUT_BASE,
        title=dict(text="Fig C: Attack Frame Matrix per Target (raw count)<br><sup>RQ1目标群体 × 攻击框架矩阵</sup>",
                   font=dict(size=14)),
        xaxis=dict(title="Framing Type (攻击框架类型)", tickangle=-20),
        yaxis=dict(title="Target Group (被攻击目标)"),
        height=620,
    )
    p = OUT_DIR / "rq2_C_target_frame_matrix.html"
    fig_c.write_html(str(p))
    inject_font(p)
    log.info(f"[VIZ] ✅ 图C: {p.name}")

    # ── 图D: 三层旭日图（交互式） ──────────────────────────────────────
    sun = (labeled_df.groupby(["lang", "topic", "frame_type"]).size()
           .reset_index(name="count"))
    sun["lang_label"]  = sun["lang"].map(LANG_LABEL)
    sun["topic_label"] = sun["topic"].apply(lambda x: f"T{x}")
    sun["frame_zh"]    = sun["frame_type"].map(FRAME_ZH)

    sun["frame_label"] = sun["frame_type"].map(FRAME_EN)
    fig_d = px.sunburst(
        sun, path=["lang_label", "topic_label", "frame_label"], values="count",
        title="Fig D: Language → Topic → Framing Type (interactive sunburst)<br><sup>语言 → 话题 → 框架 三层旭日图（点击钻取）</sup>",
        color="count", color_continuous_scale="RdYlBu_r",
    )
    fig_d.update_layout(**_LAYOUT_BASE, height=720)
    p = OUT_DIR / "rq2_D_sunburst_lang_topic_frame.html"
    fig_d.write_html(str(p))
    inject_font(p)
    log.info(f"[VIZ] ✅ 图D: {p.name}")

    # ── 图E: 分语言 Top 动词 CSV ────────────────────────────────────────
    for lang in ["en", "zh", "jp"]:
        sub = labeled_df[labeled_df["lang"] == lang]
        if sub.empty:
            continue
        vc = (sub.groupby(["frame_type", "verb"]).size()
                 .reset_index(name="count")
                 .sort_values(["frame_type","count"], ascending=[True,False]))
        vc["frame_zh"] = vc["frame_type"].map(FRAME_ZH)
        vc.to_csv(OUT_DIR / f"rq2_E_top_verbs_{lang}.csv", index=False, encoding="utf-8-sig")
    log.info("[VIZ] ✅ 图E (词频CSV) saved")

    # ── CSV 汇总（论文附录） ────────────────────────────────────────────
    summary = (labeled_df.groupby(["lang", "topic", "target", "frame_type"])
               .agg(count=("verb","count"),
                    top_verb=("verb", lambda x: x.value_counts().index[0]))
               .reset_index())
    summary["frame_zh"]    = summary["frame_type"].map(FRAME_ZH)
    summary["lang_label"]  = summary["lang"].map(LANG_LABEL)
    lang_total = summary.groupby("lang")["count"].transform("sum")
    summary["pct_in_lang"] = (summary["count"] / lang_total * 100).round(2)
    summary = summary.sort_values(["lang","count"], ascending=[True,False])
    summary.to_csv(OUT_DIR / "rq2_aggregated_summary.csv", index=False, encoding="utf-8-sig")
    log.info("[VIZ] ✅ 汇总CSV saved: rq2_aggregated_summary.csv")

    # ── 终端摘要 ────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("📊 RQ2 核心发现（直接读取）")
    print("="*65)
    print("\n▶ 各语言主导攻击框架 Top 3：")
    for lang in ["en", "zh", "jp"]:
        sub = labeled_df[labeled_df["lang"] == lang]["frame_type"].value_counts()
        top3 = " | ".join(f"{FRAME_ZH.get(k,k)}({v})" for k,v in sub.head(3).items())
        print(f"  {LANG_LABEL[lang]}: {top3}")
    print("\n▶ 攻击最密集的 Target Top 5：")
    for tgt, cnt in labeled_df["target"].value_counts().head(5).items():
        print(f"  {tgt}: {cnt}")
    print("\n▶ Layer 覆盖（SVO精准 vs 窗口残缺句）：")
    lc = labeled_df["layer"].value_counts()
    print(f"  SVO: {lc.get('svo',0)}  |  Window: {lc.get('window',0)}")
    print("\n▶ 输出文件目录：", OUT_DIR)
    print("="*65)


def run_viz_only():
    p = OUT_DIR / "rq2_framing_labeled.csv"
    if not p.exists():
        log.error(f"[VIZ-ONLY] 找不到 {p}，请先完整运行管线")
        sys.exit(1)
    df = pd.read_csv(p)
    log.info(f"[VIZ-ONLY] 加载 {len(df)} 条，重跑可视化")
    aggregate_and_visualize(df)


def run_reclassify_other(no_gemini: bool = False):
    """
    --reclassify-other 模式：只对缓存中标为 'other' 的条目重新分类。

    流程：
    1. 读取 rq2_framing_cache.json，找出所有 label=='other' 的谓语
    2. 先用更新后的 rule_classify() 过一遍（捕获新增关键词能命中的）
    3. 剩下仍为 other 的，重新送 Gemini（用含新框架类型的 prompt）
    4. 更新 cache + 重写 rq2_framing_labeled.csv（从 raw CSV + 新 cache 重建）
    5. 重跑可视化
    """
    LABEL_CSV = OUT_DIR / "rq2_framing_labeled.csv"
    RAW_CSV   = OUT_DIR / "rq2_raw_extractions.csv"

    log.info("=" * 60)
    log.info("[RECLASSIFY] --reclassify-other 模式启动")
    log.info("=" * 60)

    # ── 1. 加载现有 cache，找出 other ────────────────────────────────────
    cache = load_cache()
    if not cache:
        log.error("[RECLASSIFY] 缓存为空，请先完整运行管线")
        sys.exit(1)

    other_preds = [p for p, label in cache.items() if label == "other"]
    log.info(f"[RECLASSIFY] 缓存中共 {len(cache)} 条，other: {len(other_preds)} 条")

    # ── 2. 规则分类器先过一遍（用更新后的 FRAME_RULES + _NOISE_VERBS） ──
    rule_fixed = 0
    noise_fixed = 0
    for pred in other_preds:
        new_label = rule_classify(pred)
        if new_label != "other":
            cache[pred] = new_label
            if new_label == "noise":
                noise_fixed += 1
            else:
                rule_fixed += 1

    log.info(f"[RECLASSIFY] 规则分类器修复: {rule_fixed} 条改为具体框架, {noise_fixed} 条标为 noise")

    # ── 3. 对剩余 other 用 Gemini 重分类 ─────────────────────────────────
    still_other = [p for p in other_preds if cache.get(p) == "other"]
    log.info(f"[RECLASSIFY] 规则后仍为 other: {len(still_other)} 条，送 Gemini 重分类")

    if still_other and not no_gemini:
        api_key = get_api_key()
        if api_key:
            # 构建一个最小化的 raw_df（仅含 still_other 行，用于 per-batch append）
            if RAW_CSV.exists():
                raw_df_full = pd.read_csv(RAW_CSV)
                raw_df_sub  = raw_df_full[raw_df_full["predicate"].isin(still_other)].copy()
            else:
                # raw CSV 不存在时，构造一个虚拟 DataFrame（仅 predicate 列）
                raw_df_sub = pd.DataFrame({"predicate": still_other,
                                           "topic": -1, "lang": "unk", "layer": "unk",
                                           "target": "", "verb": "", "role": "", "context": ""})

            # 用 Gemini 重分类（重用现有逐批机制，但 labeled_path 先不覆盖旧文件）
            TEMP_LABEL = OUT_DIR / "rq2_framing_labeled_reclassify_temp.csv"
            if TEMP_LABEL.exists():
                TEMP_LABEL.unlink()

            # 临时替换 OUT_DIR/labeled 路径，在 classify_frames_async 内部会写 LABELED_PATH
            # 但 classify_frames_async 硬编码了 LABELED_PATH=OUT_DIR/"rq2_framing_labeled.csv"
            # 为避免破坏正在进行中的文件，这里直接调用异步核心但不写中间 labeled
            import asyncio, time
            from google import genai as google_genai
            from data_augmentation.LLM.google_api import APIRequester

            async def _reclassify_batch():
                client    = google_genai.Client(api_key=api_key)
                requester = APIRequester(client, model="gemini-2.5-flash", max_retries=5)
                BATCH  = 40
                batches = [still_other[i:i+BATCH] for i in range(0, len(still_other), BATCH)]
                n_total = len(still_other)
                n_done  = 0
                t_start = time.time()

                pbar = tqdm(total=n_total, desc="Gemini重分类", unit="pred",
                            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

                for batch_idx, batch in enumerate(batches):
                    prompt    = _build_classify_prompt(batch)
                    task_name = f"Reclassify-{batch_idx+1}/{len(batches)}"
                    try:
                        raw = await requester.request_async(prompt, temperature=0.1,
                                                            task_name=task_name)
                        m = re.search(r'\{[^{}]+\}', raw, re.DOTALL)
                        if m:
                            parsed = json.loads(m.group())
                            for i, pred in enumerate(batch):
                                label = parsed.get(str(i+1), "other")
                                label = label if label in FRAME_ZH else "other"
                                cache[pred] = label
                            log.info(f"[RECLASSIFY] {task_name}: {len(batch)} 条完成")
                        else:
                            log.warning(f"[RECLASSIFY] {task_name} JSON解析失败，回退规则")
                            for pred in batch:
                                cache[pred] = rule_classify(pred)
                    except Exception as e:
                        log.error(f"[RECLASSIFY] {task_name} 失败({e})，回退规则")
                        for pred in batch:
                            cache[pred] = rule_classify(pred)

                    save_cache(cache)
                    n_done += len(batch)
                    pbar.update(len(batch))
                    elapsed   = time.time() - t_start
                    rate      = n_done / elapsed if elapsed > 0 else 1
                    remaining = (n_total - n_done) / rate if rate > 0 else 0
                    log.info(
                        f"[RECLASSIFY] 进度 {n_done}/{n_total} "
                        f"({n_done/n_total*100:.1f}%) | "
                        f"已用 {elapsed/60:.1f}min | 预计剩余 {remaining/60:.1f}min"
                    )
                pbar.close()
                stats = requester.concurrency_manager.get_stats()
                log.info(f"[RECLASSIFY] Gemini统计: {stats}")

            asyncio.run(_reclassify_batch())
        else:
            log.warning("[RECLASSIFY] 无 API key，仅用规则分类器处理剩余 other")
            for pred in still_other:
                cache[pred] = rule_classify(pred)
            save_cache(cache)
    elif still_other and no_gemini:
        log.info("[RECLASSIFY] --no-gemini 模式，仅用规则分类器处理剩余 other")
        for pred in still_other:
            cache[pred] = rule_classify(pred)
        save_cache(cache)

    save_cache(cache)

    # ── 4. 从 raw CSV + 更新后的 cache 重建完整 labeled CSV ──────────────
    if not RAW_CSV.exists():
        log.error(f"[RECLASSIFY] 找不到 {RAW_CSV}，无法重建 labeled CSV")
        sys.exit(1)

    raw_df = pd.read_csv(RAW_CSV)
    raw_df["frame_type"] = raw_df["predicate"].map(cache).fillna("other")
    raw_df.to_csv(LABEL_CSV, index=False, encoding="utf-8-sig")

    # 统计效果
    new_other_count = (raw_df["frame_type"] == "other").sum()
    new_other_pct   = new_other_count / len(raw_df) * 100 if len(raw_df) > 0 else 0
    log.info(f"[RECLASSIFY] ✅ labeled CSV 已重建: {len(raw_df)} 行")
    log.info(f"[RECLASSIFY] 重分类后 other 占比: {new_other_pct:.1f}% (原来约42.5%)")
    log.info(f"[RECLASSIFY] 框架分布: {raw_df['frame_type'].value_counts().to_dict()}")

    # ── 5. 重跑可视化 ─────────────────────────────────────────────────────
    log.info("[RECLASSIFY] 自动重跑可视化...")
    aggregate_and_visualize(raw_df)
    log.info("[RECLASSIFY] ✅ 全部完成")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def get_api_key() -> str:
    if k := os.environ.get("GEMINI_API_KEY", ""):
        return k
    for p in [Path.home() / ".gemini_api_key", ROOT / ".env",
              ROOT / "scripts/api_key.txt"]:
        if p.exists():
            txt = p.read_text(encoding="utf-8").strip()
            for line in txt.splitlines():
                if "GEMINI_API_KEY" in line:
                    return line.split("=", 1)[-1].strip().strip("\"'")
            if txt and "=" not in txt:
                return txt
    return ""


def main():
    parser = argparse.ArgumentParser(
        description="RQ2 分析管线 v2.3",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
运行模式:
  完整运行         python rq2_pipeline_v2.py
  跳过Step1        python rq2_pipeline_v2.py --from-raw              # 已有raw_extractions.csv
  只重跑可视化     python rq2_pipeline_v2.py --viz-only               # 已有framing_labeled.csv
  修复other过多    python rq2_pipeline_v2.py --reclassify-other       # 只对缓存中other条目重分类
  规则分类(无API)  python rq2_pipeline_v2.py --no-gemini
  调试模式         python rq2_pipeline_v2.py --max-rows 100 --no-gemini
""",
    )
    parser.add_argument("--viz-only",  action="store_true",
                        help="跳过Step1+2，直接从 rq2_framing_labeled.csv 重跑可视化")
    parser.add_argument("--from-raw",  action="store_true",
                        help="跳过Step1，从已有 rq2_raw_extractions.csv 直接开始LLM分类")
    parser.add_argument("--no-gemini", action="store_true",
                        help="跳过Gemini，全用规则分类器（调试/离线用）")
    parser.add_argument("--max-rows",  type=int, default=None,
                        help="调试：只处理前N行文档（仅影响Step1）")
    parser.add_argument("--reclassify-other", action="store_true",
                        help=(
                            "只对缓存中标为 'other' 的条目重新分类（用更新后的规则+Gemini）。\n"
                            "无需从头重跑，直接修复 other 占比过高的问题。\n"
                            "完成后自动重建 labeled CSV 并重跑可视化。"
                        ))
    args = parser.parse_args()

    if args.viz_only:
        run_viz_only()
        return

    if args.reclassify_other:
        run_reclassify_other(no_gemini=args.no_gemini)
        return

    log.info("=" * 60)
    log.info("RQ2 管线 v2.3 启动")
    log.info("=" * 60)

    # ── Step 1: 提取（或从已有 CSV 加载） ───────────────────────────────
    RAW_CSV    = OUT_DIR / "rq2_raw_extractions.csv"
    LABEL_CSV  = OUT_DIR / "rq2_framing_labeled.csv"

    if args.from_raw:
        if not RAW_CSV.exists():
            log.error(f"[STEP1] --from-raw 指定跳过提取，但找不到 {RAW_CSV}")
            sys.exit(1)
        raw_df = pd.read_csv(RAW_CSV)
        log.info(f"[STEP1] ⏭ 跳过提取，加载已有 {RAW_CSV.name}: {len(raw_df)} 条")
    else:
        doc_df = pd.read_csv(DOC_PATH)
        doc_df = doc_df[doc_df["topic"] != -1].reset_index(drop=True)
        if args.max_rows:
            doc_df = doc_df.head(args.max_rows)
            log.info(f"[DEBUG] 限制前 {args.max_rows} 行")
        log.info(f"[DATA] {len(doc_df)} 条 | {doc_df['lang'].value_counts().to_dict()}")

        target_map = load_target_vocab(RQ1_PATH)
        nlp_map    = try_load_spacy()

        log.info("[STEP1] 三层互补提取开始...")
        raw_records = []
        for _, row in tqdm(doc_df.iterrows(), total=len(doc_df), desc="提取"):
            tid   = int(row["topic"])
            lang  = str(row.get("lang", "en")).strip()
            text  = str(row["text"])
            vocab = target_map.get(tid, [])
            if not vocab:
                continue
            for e in extract_expressions(text, lang, vocab, nlp_map):
                raw_records.append({"topic": tid, "lang": lang, **e})

        raw_df = pd.DataFrame(raw_records)
        raw_df.to_csv(RAW_CSV, index=False, encoding="utf-8-sig")
        log.info(f"[STEP1] ✅ {len(raw_df)} 条 → {RAW_CSV.name}")

    if raw_df.empty:
        log.error("[STEP1] 无提取结果，退出")
        return

    # ── Step 2: 框架归因 ────────────────────────────────────────────────
    log.info("[STEP2] 框架归因开始...")
    cache = load_cache()

    # 如果 labeled CSV 已有部分中间结果（上次断点续传留下的），
    # 从中找出已完成的 predicate，跳过它们（效果等同于 cache，但保证行级结果存在）
    already_labeled_preds: set[str] = set()
    if LABEL_CSV.exists():
        try:
            done_df = pd.read_csv(LABEL_CSV, usecols=["predicate"])
            already_labeled_preds = set(done_df["predicate"].dropna().unique())
            log.info(f"[STEP2] 检测到已有 labeled CSV，已完成 {len(already_labeled_preds)} 个唯一谓语，跳过")
        except Exception as e:
            log.warning(f"[STEP2] 读取旧 labeled CSV 失败({e})，将整体重跑")
            already_labeled_preds = set()
        # 清理 labeled CSV 中已有记录，避免重复 append（把旧文件备份，然后重写去重版）
        # 只在续跑场景下做此操作
        if already_labeled_preds:
            old_labeled = pd.read_csv(LABEL_CSV)
            old_labeled.drop_duplicates(subset=["predicate","target","topic","lang"], keep="last")\
                       .to_csv(LABEL_CSV, index=False, encoding="utf-8-sig")
            log.info(f"[STEP2] 已清理 labeled CSV 重复行")

    # 把已完成的 pred 也更新进 cache（防止 cache 未命中导致重跑）
    if already_labeled_preds and LABEL_CSV.exists():
        done_full = pd.read_csv(LABEL_CSV, usecols=["predicate","frame_type"]).dropna()
        for _, r in done_full.iterrows():
            if r["predicate"] not in cache:
                cache[r["predicate"]] = r["frame_type"]

    # 需要分类的谓语 = 全集 - 已在 cache 中的（cache 已含 already_labeled_preds）
    all_preds = raw_df["predicate"].unique().tolist()
    log.info(f"[STEP2] 唯一谓语总计 {len(all_preds)} 条")

    api_key = "" if args.no_gemini else get_api_key()

    if api_key:
        log.info("[STEP2] 使用 Gemini 异步逐批分类（APIRequester + 进度条）")
        # 如果是全新运行，先删除旧的 labeled CSV 避免 append 到残留数据上
        if not already_labeled_preds and LABEL_CSV.exists():
            LABEL_CSV.unlink()
            log.info("[STEP2] 清除旧 labeled CSV，重新开始")
        frame_map = classify_frames_gemini(all_preds, cache, api_key, raw_df)
    else:
        log.warning("[STEP2] 无 API key，使用规则分类器（一次性完成，无进度条）")
        frame_map = {p: rule_classify(p) for p in all_preds}
        raw_df["frame_type"] = raw_df["predicate"].map(frame_map).fillna("other")
        raw_df.to_csv(LABEL_CSV, index=False, encoding="utf-8-sig")
        log.info(f"[STEP2] ✅ → {LABEL_CSV.name}")

    # ── 合并最终 labeled 结果（Gemini路径下 labeled CSV 已被逐批写入） ──
    if api_key:
        # 用 cache 补全所有行（理论上 labeled CSV 已完整，此处双保险）
        raw_df["frame_type"] = raw_df["predicate"].map(cache).fillna("other")
        # 重写一次完整去重版本作为最终文件
        raw_df.to_csv(LABEL_CSV, index=False, encoding="utf-8-sig")
        log.info(f"[STEP2] ✅ 最终 labeled CSV 已整合 → {LABEL_CSV.name}")

    # ── Step 3: 聚合 + 可视化 ───────────────────────────────────────────
    log.info("[STEP3] 聚合分析与可视化...")
    labeled_df = pd.read_csv(LABEL_CSV)
    aggregate_and_visualize(labeled_df)

    log.info("=" * 60)
    log.info(f"✅ 全部完成。日志: {LOG_PATH}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
