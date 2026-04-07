"""
RQ1 Target Extraction v3 — 三阶段隔离管线
==========================================
三个阶段完全解耦，各自输出 checkpoint，任意阶段可独立重跑：

  Stage 1+2 (layer12): spaCy NER + 领域词典  →  checkpoint_layer12.csv
  Stage 3   (llm):     并发 Gemini 兜底       →  checkpoint_llm.csv
  Stage viz (viz):     从任意 checkpoint 生图  →  *.png

运行方式：

  # 全流程一次跑完
  python target_extraction_v3.py --stage full

  # 轻量 spaCy（显存不足时）
  python target_extraction_v3.py --stage layer12 --spacy-fallback

推荐工作流
# Step 1: 跑 Layer1+2（安全，无 API 消耗）
python unsupervised_classification/RQ1/target_extraction_v3.py --stage layer12

# Step 2: 先看看结果对不对
python unsupervised_classification/RQ1/target_extraction_v3.py --stage viz

# Step 3: 觉得可以了，再花 API 配额跑 LLM 兜底
python unsupervised_classification/RQ1/target_extraction_v3.py --stage llm

# Step 4: 更新图表
python unsupervised_classification/RQ1/target_extraction_v3.py --stage viz

作者: Zhidian  |  日期: 2026-04
"""

# ── 标准库 ────────────────────────────────────────────────
import argparse
import ast
import asyncio
import json
import logging as _logging
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# ── 第三方库 ──────────────────────────────────────────────
import pandas as pd
from tqdm import tqdm
from tqdm.asyncio import tqdm

# ── 项目内工具 ────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.set_logger import setup_logging

# 模块级 fallback logger（被 __main__ 覆盖）
logger: _logging.Logger = _logging.getLogger(__name__)

# ── 路径配置 ──────────────────────────────────────────────
THIS_DIR   = Path(__file__).resolve().parent
DATA_DIR   = THIS_DIR / "data"
VIS_DIR = THIS_DIR / "visualizations"
CKPT_L12   = DATA_DIR / "checkpoint_layer12.csv"
CKPT_LLM   = DATA_DIR / "checkpoint_llm.csv"
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "unsupervised_classification"
    / "topic_modeling_results"
    / "sixth" / "data" / "document_topic_mapping.csv"
)


# ============================================================
# 零、环境 & spaCy 配置
# ============================================================

def load_env():
    env_path = PROJECT_ROOT / ".env"
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=env_path, override=False)
    except ImportError:
        pass


def get_api_key(api_type: str = "gemini") -> str | None:
    key_map = {"gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY"}
    key = os.environ.get(key_map.get(api_type, ""), "").strip()
    if not key:
        logger.warning(
            f"未找到 {key_map[api_type]}，"
            f"请在 {PROJECT_ROOT / '.env'} 中添加：\n"
            f"  {key_map[api_type]}=your_key_here"
        )
        return None
    return key


def normalize_lang(lang: str) -> str:
    """规范化语言代码：jp → ja，其他保持不变"""
    return "ja" if lang.lower() in ("ja", "jp") else lang.lower()


SPACY_MODELS = {
    "en": "en_core_web_trf",
    "zh": "zh_core_web_trf",
    "ja": "ja_core_news_trf",
}
SPACY_MODELS_FALLBACK = {
    "en": "en_core_web_sm",
    "zh": "zh_core_web_sm",
    "ja": "ja_core_news_sm",
}
SPACY_KEEP_LABELS = {
    "en": {"PERSON", "ORG", "NORP", "GPE", "FAC"},
    "zh": {"PERSON", "ORG", "GPE", "NORP", "LOC"},
    "ja": {
        "Person", "Organization", "Country", "City", "Province",
        "Location_Other", "Political_Party",
        "PERSON", "ORG", "GPE", "NORP", "LOC",
        "人名", "組織名", "地名", "法人名", "政治的組織名",
    },
}


# ============================================================
# 一、黑名单
# ============================================================

PRONOUN_BLACKLIST = {
    "i","me","my","mine","myself","you","your","yours","yourself",
    "he","him","his","himself","she","her","hers","herself",
    "it","its","itself","we","us","our","ours","ourselves",
    "they","them","their","theirs","themselves",
    "one","who","whom","that","this","these","those",
    "someone","anyone","everyone","nobody","anybody","everybody",
    "people","person","man","woman","men","women","child","children",
    # 中文
    "我","你","他","她","它","我们","你们","他们","她们",
    "自己","别人","大家","有人","某人","人们","人家",
    "什么","谁","哪里","这里","那里",
    # 日文
    "私","僕","俺","あなた","君","彼","彼女","彼ら",
    "我々","自分","皆","誰","誰か","みんな","人々",
    "お前","あいつ","そいつ","こいつ",
}

NOISE_BLACKLIST = {
    "god","religion","faith","bible","prayer","church",
    "christianity","belief","worship","theology",
    "宗教","信仰","圣经","宗教信仰","上帝","神",
    "宗教","信仰","聖書","神",
}


# ============================================================
# 二、领域词典（基于预分析 top-terms + GLiNER 历史结果）
# ============================================================

DOMAIN_GAZETTEER: dict[str, list[str]] = {

    "Religious Group": [
        # EN
        "catholics","catholic","roman catholics",
        "protestants","protestant",
        "evangelicals","evangelical",
        "mormons","mormon","latter-day saints","lds",
        "baptists","baptist","southern baptists",
        "methodists","methodist",
        "adventists","adventist","seventh-day adventists",
        "pentecostals","pentecostal",
        "orthodox","eastern orthodox","greek orthodox",
        "anglicans","anglican","episcopalians","episcopalian",
        "lutherans","lutheran",
        "presbyterians","presbyterian",
        "calvinists","calvinist",
        "jehovah's witnesses","jehovah witnesses",
        "quakers","amish","mennonites","mennonite",
        "unitarians","unitarian",
        "fundamentalists","fundamentalist",
        "charismatics","charismatic",
        "traditionalists","traditionalist","trads",
        "sedevacantists",
        "sunni","shia","shi'a","shiite",
        "muslims","muslim","islamists","islamist",
        "jews","jewish","orthodox jews","ultra-orthodox",
        "hindus","hindu","buddhists","buddhist",
        "sikhs","sikh","atheists","atheist",
        "agnostics","agnostic","pagans","pagan","cultists",
        # ZH
        "天主教徒","罗马天主教徒",
        "新教徒","基督徒","基督教徒",
        "东正教徒","东方正教徒",
        "穆斯林","伊斯兰教徒",
        "犹太人","犹太教徒",
        "佛教徒","道教徒",
        "信徒","教徒","信众","信教者",
        "福音派","浸信会教徒","卫理公会教徒","长老会教徒",
        "加尔文主义者",
        "二世信徒","二世",
        "邪教徒","邪教信徒",
        "保守派基督徒","保守派天主教徒","自由派基督徒",
        # JA
        "カトリック","カトリック教徒",
        "プロテスタント","プロテスタント教徒",
        "福音派","バプテスト","メソジスト",
        "末日聖徒","モルモン",
        "セブンスデー・アドベンチスト",
        "エホバの証人",
        "ムスリム","イスラム教徒",
        "ユダヤ人","ユダヤ教徒",
        "仏教徒","ヒンドゥー教徒",
        "信者","教徒","信徒",
        "カルト信者","カルト",
        "二世信者","宗教二世",
        "キリスト教徒","正教会信徒",
        "保守派クリスチャン",
    ],

    "Organization": [
        # EN
        "catholic church","roman catholic church","vatican","holy see","curia",
        "church of england","anglican church",
        "southern baptist convention",
        "united church","united church of christ",
        "adventist church","seventh-day adventist church",
        "church of jesus christ","lds church",
        "jesuit","jesuits","opus dei",
        "knights of columbus",
        "national council of churches","world council of churches",
        "planned parenthood","supreme court",
        "diocese","archdiocese",
        # ZH
        "天主教会","罗马天主教会","教廷","梵蒂冈",
        "统一教会","世界和平统一家庭联合会",
        "全能神教会","东方闪电",
        "法轮功","法轮大法",
        "创价学会",
        "耶和华见证人","新天地",
        "摩门教","耶稣基督后期圣徒教会",
        "天地门教",
        "自民党","中国共产党","共产党","国民党",
        "教会","教堂","修道院",
        "主教团","主教会议",
        # JA
        "統一教会","世界平和統一家庭連合","旧統一教会",
        "創価学会","創価",
        "日本会議",
        "幸福の科学",
        "オウム真理教","アレフ",
        "全能神教会",
        "エホバの証人",
        "キリスト教福音宣教会","JMS",
        "カトリック教会",
        "自民党","公明党","立憲民主党",
        "日本共産党","維新の会",
        "ユダヤ国際金融資本","ユダヤ共産主義",
    ],

    "Specific Person": [
        # EN
        "trump","donald trump",
        "biden","joe biden",
        "obama","barack obama",
        "pope francis","francis",
        "pope benedict","benedict xvi",
        "pope john paul","john paul ii",
        "the pope","pope",
        "martin luther","luther",
        "john calvin","calvin",
        "billy graham","graham",
        "joel osteen","osteen",
        "rick warren","jerry falwell","pat robertson",
        "paul","saint paul","apostle paul",
        "peter","saint peter","moses","abraham","noah",
        "trudeau","justin trudeau","harper",
        "burke","cardinal burke","baum",
        "jesus","jesus christ",
        # ZH
        "安倍晋三","安倍",
        "习近平",
        "教宗","教皇","方济各","本笃十六世",
        "孔庆东","汪海林",
        "张角","洪秀全",
        "马丁路德","约翰加尔文",
        "保罗","彼得","摩西","亚伯拉罕",
        "耶稣","耶稣基督",
        # JA
        "安倍","安倍晋三","安倍元首相",
        "岸田","岸田文雄",
        "麻生","麻生太郎",
        "山上","山上徹也",
        "トランプ","バイデン",
        "フランシスコ教皇",
        "イエス","イエス・キリスト",
        "パウロ","ペテロ",
        "豊臣秀吉","織田信長",
    ],

    "Social Identity": [
        # EN
        "liberals","liberal","conservatives","conservative",
        "progressives","progressive",
        "trads","traditionalists",
        "fundamentalists",
        "white christians","white american christians",
        "christian nationalists","christian nationalism",
        "gay people","gays","lgbtq","lgbt",
        "clergy","priests","priest",
        "pastors","pastor","bishops","bishop",
        "nuns","nun","monks","monk",
        "laity","laypeople","layman",
        "televangelists","televangelist",
        "cult leaders","cult leader",
        "heretics","heretic","apostates","apostate",
        "nonbelievers","unbelievers","infidels",
        # ZH
        "圣母婊","神棍","骗子神父",
        "基督狗","公知",
        "女权","女权主义者",
        "左派","右派","保守派","自由派",
        "无神论者","迷信者","传教士",
        "神父","牧师","修女","修士",
        "主教","神职人员","异端","叛教者","邪教",
        # JA
        "カルト","スパイ","工作員",
        "反日","売国奴","売国",
        "保守派","リベラル","左翼","右翼",
        "保守派クリスチャン","保守派天主教徒",
        "悪魔","サタン","異端","背教者",
        "神父","牧師","修道士","修道女",
        "司教","主教","連中",
        "ユダ","偽クリスチャン",
    ],

    "Political Group": [
        # EN
        "republicans","republican","gop",
        "democrats","democrat","democratic party","maga",
        "conservatives","progressives",
        "white nationalists","white nationalist","nativists",
        # ZH
        "共产党","中国共产党","国民党",
        "左派","左翼","右派","右翼",
        "保守派","自由派",
        # JA
        "自民党","公明党","立憲民主党",
        "共産党","日本共産党","維新の会","保守党",
        "韓国","中国","ロシア",
        "ユダヤ","日本人",
    ],
}

# 按长度降序排列：最长优先匹配
_GAZ_INDEX: list[tuple[str, str]] = sorted(
    ((t.lower(), c) for c, ts in DOMAIN_GAZETTEER.items() for t in ts),
    key=lambda x: len(x[0]),
    reverse=True,
)


# ============================================================
# 三、别名归一化
# ============================================================

ALIAS_MAP: dict[str, str] = {
    "roman catholic church": "Catholic Church",
    "catholic church": "Catholic Church",
    "the church": "Church",
    "holy see": "Vatican",
    "cc": "Catholic Church",
    "pope francis": "Pope Francis",
    "the pope": "Pope",
    "pope": "Pope",
    "donald trump": "Trump",
    "joe biden": "Biden",
    "jesus christ": "Jesus",
    "jesus": "Jesus",
    "christ": "Jesus",
    "jehovah's witnesses": "Jehovah's Witnesses",
    "latter-day saints": "LDS Church",
    "lds church": "LDS Church",
    # ZH
    "罗马天主教会": "天主教会",
    "教廷": "梵蒂冈",
    "全能神教会": "全能神教会",
    "东方闪电": "全能神教会",
    "世界和平统一家庭联合会": "统一教会",
    "法轮大法": "法轮功",
    "耶稣基督": "耶稣",
    "安倍": "安倍晋三",
    "教宗": "教皇",
    "方济各": "教皇方济各",
    # JA
    "世界平和統一家庭連合": "統一教会",
    "旧統一教会": "統一教会",
    "創価": "創価学会",
    "安倍晋三": "安倍晋三",
    "安倍元首相": "安倍晋三",
    "イエス": "イエス・キリスト",
    "キリスト": "イエス・キリスト",
    "フランシスコ教皇": "フランシスコ教皇",
}


# ============================================================
# Layer 1：spaCy NER
# ============================================================

def load_spacy_models(use_fallback: bool = False) -> dict:
    import spacy
    models = {}
    model_map = SPACY_MODELS_FALLBACK if use_fallback else SPACY_MODELS
    for lang, name in model_map.items():
        try:
            models[lang] = spacy.load(name)
            logger.info(f"[spaCy] ✓ {name}")
        except OSError:
            fb = SPACY_MODELS_FALLBACK[lang]
            try:
                models[lang] = spacy.load(fb)
                logger.warning(f"[spaCy] {name} 不可用，已回退至 {fb}")
            except OSError:
                logger.warning(f"[spaCy] {lang} 无可用模型，跳过")
                models[lang] = None
    return models


def extract_spacy_entities(text: str, lang: str, nlp_models: dict) -> list[dict]:
    nlp = nlp_models.get(lang)
    if not nlp:
        return []
    try:
        doc = nlp(text[:1000])
    except Exception:
        return []
    keep = SPACY_KEEP_LABELS.get(lang, set())
    return [
        {"text": e.text.strip(), "label": e.label_, "source": "spacy"}
        for e in doc.ents
        if e.label_ in keep and 1 < len(e.text.strip()) <= 30
    ]


# ============================================================
# Layer 2：领域词典
# ============================================================

def _overlaps(s: int, e: int, spans: list) -> bool:
    return any(not (e <= es or s >= ee) for es, ee in spans)


def extract_gazetteer_entities(text: str, lang: str) -> list[dict]:
    tl = text.lower()
    spans, results = [], []
    for term, cat in _GAZ_INDEX:
        if lang == "en":
            for m in re.finditer(r'\b' + re.escape(term) + r'\b', tl):
                s, e = m.start(), m.end()
                if not _overlaps(s, e, spans):
                    spans.append((s, e))
                    results.append({"text": text[s:e].strip(), "label": cat, "source": "gazetteer"})
        else:
            start = 0
            while True:
                idx = tl.find(term, start)
                if idx == -1:
                    break
                s, e = idx, idx + len(term)
                if not _overlaps(s, e, spans):
                    spans.append((s, e))
                    results.append({"text": text[s:e].strip(), "label": cat, "source": "gazetteer"})
                start = idx + len(term)
    return results


# ============================================================
# 后处理
# ============================================================

_JA_VERB = re.compile(r'(して|した|している|しまくり|なって|グルに|になった|してる|やってる|ぐずぐず|ズブズブ)')
_JA_SENT = re.compile(r'.{5,}[がはをにでもとへよりから].{3,}[るたいすくけれてじゃ]$')
_ZH_VERB = re.compile(r'[不没被让把给就也都还只]?.{0,2}[住来去起下上过着了把被让]')


def clean_entity(text: str, lang: str) -> str | None:
    text = re.sub(r'^[\s\.,;:!?、。，；：！？「」『』（）()\[\]【】…]+', '', text)
    text = re.sub(r'[\s\.,;:!?、。，；：！？「」『』（）()\[\]【】…]+$', '', text).strip()
    if not text:
        return None
    max_len = 15 if lang in ("zh", "ja") else 40
    if not (1 < len(text) <= max_len):
        return None
    if text.lower() in PRONOUN_BLACKLIST:
        return None
    if text.lower() in NOISE_BLACKLIST:
        return None
    if lang == "ja" and (_JA_SENT.search(text) or _JA_VERB.search(text)):
        return None
    if lang == "zh":
        if "的" in text and len(text) > 8:
            return None
        if _ZH_VERB.search(text) and len(text) > 6:
            return None
    return text


def normalize(text: str) -> str:
    return ALIAS_MAP.get(text.lower(), text)


def dedup(entities: list[dict]) -> list[dict]:
    sorted_e = sorted(entities, key=lambda x: len(x["text"]), reverse=True)
    kept: list[dict] = []
    for e in sorted_e:
        tl = e["text"].lower()
        if not any(tl in k["text"].lower() or k["text"].lower() in tl for k in kept):
            kept.append(e)
    return kept


def _guess_lang(text: str) -> str:
    ja = len(re.findall(r'[\u3040-\u309F\u30A0-\u30FF]', text))
    zh = len(re.findall(r'[\u4E00-\u9FFF]', text))
    if ja > 3:
        return "jp"  # 保持与数据一致的 "jp" 代码
    if zh / (len(text) or 1) > 0.3:
        return "zh"
    return "en"


def _process_row_entities(raw: list[dict], lang: str) -> tuple[list[str], list[str]]:
    """清洗、归一化、去重 → 返回 (targets_list, detail_list)"""
    cleaned, seen = [], set()
    for ent in raw:
        ct = clean_entity(ent["text"], lang)
        if ct and ct.lower() not in seen:
            nt = normalize(ct)
            cleaned.append({**ent, "text": nt})
            seen.add(nt.lower())
    cleaned = dedup(cleaned)
    targets = [e["text"] for e in cleaned]
    details = [f'{e["text"]}|{e["label"]}|{e["source"]}' for e in cleaned]
    return targets, details


# ============================================================
# Stage 1+2：spaCy + 词典 → checkpoint_layer12.csv
# ============================================================

def run_layer12(
    input_csv: Path,
    output_csv: Path = CKPT_L12,
    use_spacy_fallback: bool = False,
) -> pd.DataFrame:

    logger.info("=" * 60)
    logger.info("Stage 1+2: spaCy NER + 领域词典")
    logger.info("=" * 60)

    df = pd.read_csv(input_csv)
    logger.info(f"  读入: {len(df)} 条  来自 {input_csv}")

    if "lang" not in df.columns:
        logger.warning("  无 lang 列，启用启发式推断...")
        df["lang"] = df["text"].apply(_guess_lang)
    else:
        # 规范化语言代码（jp → ja）
        df["lang"] = df["lang"].apply(normalize_lang)

    logger.info("  加载 spaCy 模型...")
    nlp_models = load_spacy_models(use_fallback=use_spacy_fallback)

    targets_col, details_col = [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="  Layer 1+2"):
        text = str(row.get("text", "")).strip()
        lang = normalize_lang(str(row.get("lang", "en")).lower())
        if lang not in ("en", "zh", "ja"):
            lang = "en"

        if not text:
            targets_col.append([])
            details_col.append([])
            continue

        raw = (extract_spacy_entities(text, lang, nlp_models) +
               extract_gazetteer_entities(text, lang))
        targets, details = _process_row_entities(raw, lang)
        targets_col.append(targets)
        details_col.append(details)

    df["hate_targets"]        = targets_col
    df["hate_targets_detail"] = details_col
    df["llm_enriched"]        = False   # LLM 标记，Stage 3 会更新

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    total = sum(len(t) for t in targets_col)
    by_src = Counter(
        d.split("|")[2] for ds in details_col for d in ds if "|" in d
    )
    logger.info(f"  ✅ Layer 1+2 完成: 总实体 {total}, 来源 {dict(by_src)}")
    logger.info(f"  Checkpoint 已保存: {output_csv}")

    return df


# ============================================================
# Stage 3：并发 LLM 兜底 → checkpoint_llm.csv
# ============================================================

# ── LLM Prompt ──────────────────────────────────────────────

def build_prompt(text: str, lang: str) -> str:
    ln = {"en": "English", "zh": "Chinese", "ja": "Japanese"}.get(lang, "multilingual")
    return f"""You are a precise NER system for hate speech research.
Extract ONLY named entities (people, groups, organizations) that are TARGETS in this {ln} text.

RULES:
1. Return proper nouns / established group names only (e.g. "安倍晋三", "統一教会", "Catholics")
2. NO full sentences, clauses, or verb phrases
3. NO pronouns (I/you/he/私/あなた/我/你 etc.)
4. Max: 15 CJK chars OR 40 Latin chars per entity
5. Skip generic: religion, god, faith, 宗教, 信仰, 神

Output ONLY JSON array:
[{{"text":"entity","category":"Religious Group|Organization|Specific Person|Social Identity|Political Group"}}]
Empty → []

Text: {text}
JSON:"""


# ── 并发核心（对标项目 google_api.py 风格）────────────────────
async def _llm_request_async(
    client,
    api_type: str,
    model_name: str,
    text: str,
    lang: str,
    sem: asyncio.Semaphore,
    executor: ThreadPoolExecutor,  # 传入自定义线程池
    max_retries: int = 3,
    idx: int = 0,
) -> list[dict]:
    """单条异步请求，带 Semaphore 限流 + 自定义线程池 + 指数退避重试"""
    # 简单的 Prompt 构造逻辑（确保你已定义此函数）
    prompt = build_prompt(text, lang) 
    # prompt = f"Extract entities from this {lang} text: {text}. Return JSON list."

    async with sem:
        for attempt in range(max_retries):
            try:
                loop = asyncio.get_event_loop()
                # 根据 API 类型调用不同方法
                if api_type == "gemini":
                    response = await loop.run_in_executor(
                        executor,
                        lambda: client.models.generate_content(
                            model=model_name, contents=prompt
                        )
                    )
                    response_text = response.text.strip()
                elif api_type == "openai":
                    response = await loop.run_in_executor(
                        executor,
                        lambda: client.chat.completions.create(
                            model=model_name,
                            messages=[{"role": "user", "content": prompt}]
                        )
                    )
                    response_text = response.choices[0].message.content.strip()
                else:
                    raise ValueError(f"Unsupported api_type: {api_type}")
                
                # 正则解析 JSON
                m = re.search(r'\[.*?\]', response_text, re.DOTALL)
                if not m:
                    return []
                
                raw = json.loads(m.group())
                return [
                    {"text": e["text"].strip(),
                     "label": e.get("category", "Unknown"),
                     "source": "llm"}
                    for e in raw if isinstance(e, dict) and "text" in e
                ]
            except Exception as exc:
                err = str(exc).lower()
                # 判断是否为限流错误
                is_rate = any(k in err for k in ["429", "quota", "rate limit", "resource_exhausted"])
                
                # 指数退避：限流时等待时间更长
                wait = (2 ** attempt) * (5.0 if is_rate else 1.0)
                
                if attempt < max_retries - 1:
                    logger.debug(f"  [LLM #{idx}] 第{attempt+1}次失败, {wait:.1f}s 后重试: {exc}")
                    await asyncio.sleep(wait)
                else:
                    logger.error(f"  [LLM #{idx}] 彻底失败: {exc}")
                    return []
    return []

async def _run_llm_concurrent(
    tasks: list[tuple[int, str, str]],
    api_key: str,
    api_type: str = "gemini",
    model_name: str = "gemini-2.5-flash",
    concurrency: int = 25,  # 调整为 50，适合 OpenAI 500 RPM 限速
) -> dict[int, list[dict]]:
    """真正的高并发入口"""
    if api_type == "gemini":
        from google import genai
        client = genai.Client(api_key=api_key)
    elif api_type == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
    else:
        raise ValueError(f"Unsupported api_type: {api_type}")
    
    # 1. 撑大底层线程池：确保线程数与并发数匹配
    executor = ThreadPoolExecutor(max_workers=concurrency)
    
    # 2. 信号量控制：控制同时进入 API 调用的协程数
    sem = asyncio.Semaphore(concurrency)

    # 3. 构造协程任务
    coros = [
        _llm_request_async(client, api_type, model_name, t[1], t[2], sem, executor, idx=t[0])
        for t in tasks
    ]

    results = {}
    try:
        # 4. 使用 tqdm.asyncio.gather 实时监控
        # 这是最稳妥的进度条实现方案
        responses = await tqdm.gather(
            *coros,
            desc=f"🚀 并发提取 ({concurrency} 并发)",
            unit="条",
            colour="green"
        )

        # 5. 组装结果
        for (idx, _, _), ents in zip(tasks, responses):
            results[idx] = ents
            
    finally:
        # 6. 务必关闭线程池
        executor.shutdown(wait=True)

    return results


def run_llm(
    input_csv: Path  = CKPT_L12,
    output_csv: Path = CKPT_LLM,
    api_type: str    = "gemini",
    model_name: str  = "gemini-2-flash",
    min_entities: int = 2,    # Layer1+2 实体数 < 此值才触发 LLM
    concurrency: int  = 50,
) -> pd.DataFrame:

    logger.info("=" * 60)
    logger.info("Stage 3: LLM 并发兜底")
    logger.info("=" * 60)

    if not input_csv.exists():
        raise FileNotFoundError(
            f"找不到 Layer1+2 checkpoint: {input_csv}\n"
            f"请先运行: python target_extraction_v3.py --stage layer12"
        )

    api_key = get_api_key(api_type)
    if not api_key:
        raise SystemExit(1)

    df = pd.read_csv(input_csv)
    logger.info(f"  读入 checkpoint: {len(df)} 条  来自 {input_csv}")

    # 解析已有实体（CSV 中存的是字符串化 list）
    def parse_list_col(val):
        if isinstance(val, list):
            return val
        try:
            return ast.literal_eval(str(val))
        except Exception:
            return []

    df["hate_targets"]        = df["hate_targets"].apply(parse_list_col)
    df["hate_targets_detail"] = df["hate_targets_detail"].apply(parse_list_col)

    # 找出需要 LLM 兜底的行
    needs_llm = [
        (i, str(df.iloc[i]["text"]), str(df.iloc[i].get("lang", "en")).lower())
        for i in range(len(df))
        if len(df.iloc[i]["hate_targets"]) < min_entities
    ]
    logger.info(f"  需 LLM 兜底: {len(needs_llm)} / {len(df)} 条  "
                f"(并发={concurrency}, 模型={model_name})")

    if not needs_llm:
        logger.info("  所有行实体已足够，跳过 LLM 层")
        df.to_csv(output_csv, index=False, encoding="utf-8-sig")
        return df

    # 并发执行
    llm_results = asyncio.run(
        _run_llm_concurrent(needs_llm, api_key, api_type, model_name, concurrency)
    )

    # 合并结果
    added_total = 0
    for df_idx, llm_ents in llm_results.items():
        lang = str(df.iloc[df_idx].get("lang", "en")).lower()
        existing_texts = {t.lower() for t in df.iloc[df_idx]["hate_targets"]}
        new_targets, new_details = [], []

        for ent in llm_ents:
            ct = clean_entity(ent["text"], lang)
            if ct and ct.lower() not in existing_texts:
                nt = normalize(ct)
                new_targets.append(nt)
                new_details.append(f'{nt}|{ent["label"]}|llm')
                existing_texts.add(nt.lower())

        if new_targets:
            df.at[df_idx, "hate_targets"]        = df.iloc[df_idx]["hate_targets"] + new_targets
            df.at[df_idx, "hate_targets_detail"]  = df.iloc[df_idx]["hate_targets_detail"] + new_details
            df.at[df_idx, "llm_enriched"]         = True
            added_total += len(new_targets)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    by_src = Counter(
        d.split("|")[2]
        for ds in df["hate_targets_detail"] for d in ds if "|" in d
    )
    logger.info(f"  ✅ LLM 层完成: 新增实体 {added_total}")
    logger.info(f"  全量来源分布: {dict(by_src)}")
    logger.info(f"  Checkpoint 已保存: {output_csv}")
    return df


# ============================================================
# Stage viz：可视化（从任意 checkpoint 读取）
# ============================================================

def _latest_checkpoint() -> Path:
    """优先用 LLM checkpoint，回退到 Layer12 checkpoint"""
    if CKPT_LLM.exists():
        return CKPT_LLM
    if CKPT_L12.exists():
        return CKPT_L12
    raise FileNotFoundError(
        f"没有找到 checkpoint 文件。\n"
        f"请先运行: python target_extraction_v3.py --stage layer12"
    )


def analyze_topic_targets(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """按 Topic 汇总，返回 summary DataFrame（不保存，由调用方保存）"""
    import ast
    def pl(v):
        if isinstance(v, list): return v
        try: return ast.literal_eval(str(v))
        except: return []

    df = df.copy()
    df["hate_targets"]        = df["hate_targets"].apply(pl)
    df["hate_targets_detail"] = df["hate_targets_detail"].apply(pl)

    rows = []
    for tid, grp in df[df["topic"] != -1].groupby("topic"):
        all_t  = [t for sub in grp["hate_targets"] for t in sub]
        all_d  = [d for sub in grp["hate_targets_detail"] for d in sub]
        tc     = Counter(all_t)
        lc: Counter = Counter()
        for d in all_d:
            pts = d.split("|")
            if len(pts) >= 2:
                lc[pts[1]] += 1
        rows.append({
            "Topic_ID":          tid,
            "Topic_Size":        len(grp),
            "Top_Targets":       [f"{t}({c})" for t, c in tc.most_common(top_n)],
            "Target_Categories": dict(lc.most_common(5)),
            "Unique_Targets":    len(tc),
            "LLM_Enriched_Rows": int(grp["llm_enriched"].sum()) if "llm_enriched" in grp else 0,
        })
    return pd.DataFrame(rows)


def run_viz(
    checkpoint_csv: Path | None = None,
    output_dir: Path = DATA_DIR,
    top_n: int = 10,
):
    logger.info("=" * 60)
    logger.info("Stage viz: 生成可视化图表")
    logger.info("=" * 60)

    ckpt = checkpoint_csv or _latest_checkpoint()
    logger.info(f"  使用 checkpoint: {ckpt}")

    df = pd.read_csv(ckpt)
    summary_df = analyze_topic_targets(df, top_n=top_n)

    # 保存 summary CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "rq1_topic_targets_v3.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    logger.info(f"  Topic 汇总: {summary_path}")

    _draw_all(summary_df, df, VIS_DIR)


def _draw_all(summary_df: pd.DataFrame, df_full: pd.DataFrame, output_dir: Path):
    # ── 导入 & 字体初始化 ──────────────────────────────────────
    from viz_utils import setup_matplotlib, bilingual_labels_batch, get_cjk_font_prop
    import matplotlib.pyplot as plt
    import numpy as np
    import ast

    setup_matplotlib()                        # CJK 字体 + 风格一键配置
    font_prop = get_cjk_font_prop()           # 用于 set_yticklabels 的 FontProperties

    # 预加载 Gemini API key（从环境变量，如果有的话）
    translate_api_key = os.environ.get("GEMINI_API_KEY", "") or None

    CATEGORY_COLORS = {
        "Religious Group": "#4E79A7",
        "Organization":    "#F28E2B",
        "Specific Person": "#E15759",
        "Social Identity": "#76B7B2",
        "Political Group": "#59A14F",
        "Unknown":         "#BAB0AC",
    }
    LANG_COLORS = {"en": "#4E79A7", "zh": "#E15759", "ja": "#F28E2B", "other": "#BAB0AC"}
    categories  = [c for c in CATEGORY_COLORS if c != "Unknown"]
    topic_ids   = summary_df["Topic_ID"].tolist()

    def pl(v):
        if isinstance(v, list): return v
        try: return ast.literal_eval(str(v))
        except: return []

    def _safe_tc(row) -> dict:
        tc = row["Target_Categories"]
        if isinstance(tc, dict): return tc
        try: return ast.literal_eval(str(tc))
        except: return {}

    # ── 语言映射（tid → 主语言）────────────────────────────────
    lang_map: dict = {}
    if "lang" in df_full.columns:
        for tid, grp in df_full[df_full["topic"] != -1].groupby("topic"):
            lang_map[tid] = grp["lang"].mode()[0] if len(grp) else "other"

    # ══════════════════════════════════════════════════════════
    # 图1：类别堆叠横条图（Topic 级，纯英文标签，不需翻译）
    # ══════════════════════════════════════════════════════════
    logger.info("  [VIZ] 图1: 类别堆叠横条图")
    cat_counts = {c: [] for c in categories}
    for _, row in summary_df.iterrows():
        tc = _safe_tc(row)
        for c in categories:
            cat_counts[c].append(tc.get(c, 0))

    y  = np.arange(len(topic_ids))
    fig, ax = plt.subplots(figsize=(12, max(6, len(topic_ids) * 0.38)))
    lefts = np.zeros(len(topic_ids))
    for c in categories:
        vals = np.array(cat_counts[c])
        ax.barh(y, vals, left=lefts, color=CATEGORY_COLORS[c], label=c, height=0.72)
        lefts += vals
    ax.set_yticks(y)
    ax.set_yticklabels([f"T{t}" for t in topic_ids], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Entity Count")
    ax.set_title("RQ1: Target Category Distribution by Topic", fontsize=13)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.8)
    plt.tight_layout()
    p1 = output_dir / "rq1_topic_category_stacked.png"
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"    ✓ {p1}")

    # ══════════════════════════════════════════════════════════
    # 图2：每 Topic Top-N 实体条形图矩阵（双语标签）
    # ══════════════════════════════════════════════════════════
    logger.info("  [VIZ] 图2: Top-N 实体条形图矩阵（双语标签）")

    # 预收集全部实体，批量翻译（一次 API 调用）
    all_entities_flat: list[str] = []
    topic_entries_cache: dict = {}
    show = topic_ids

    for tid in show:
        rows = summary_df[summary_df["Topic_ID"] == tid]
        if rows.empty: continue
        raw = pl(rows.iloc[0]["Top_Targets"])[:7]
        entries = []
        for item in raw:
            m = re.match(r'^(.*)\((\d+)\)$', str(item).strip())
            if m:
                entries.append((m.group(1).strip(), int(m.group(2))))
        topic_entries_cache[tid] = entries
        all_entities_flat.extend([e[0] for e in entries])

    # 一次性批量翻译
    from viz_utils import translate_entities
    unique_ents = list(dict.fromkeys(all_entities_flat))   # 去重保序
    trans_map   = translate_entities(unique_ents, api_key=translate_api_key, use_api=bool(translate_api_key))
    logger.info(f"    翻译完成: {len(trans_map)} 个实体")

    # 绘图
    if len(show) <= 12:
        ncols = 3
    elif len(show) <= 20:
        ncols = 4
    elif len(show) <= 35:
        ncols = 5
    else:
        ncols = 6
    nrows = (len(show) + ncols - 1) // ncols
    fig2, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.2, max(3.6, nrows * 2.6)), squeeze=False)
    flat = axes.flatten()

    from viz_utils import bilingual_label
    for i, tid in enumerate(show):
        ax = flat[i]
        entries = topic_entries_cache.get(tid, [])
        if not entries:
            ax.axis("off")
            ax.set_title(f"T{tid} (–)", fontsize=8)
            continue

        entities = [e[0] for e in entries]
        values   = [e[1] for e in entries]
        lang     = str(lang_map.get(tid, "other")).lower()
        color    = LANG_COLORS.get(lang, LANG_COLORS["other"])

        # 生成双语标签：「統一教会\n(Unification Church)」
        bilabels = [bilingual_label(e, trans_map.get(e)) for e in entities]

        bars = ax.barh(range(len(bilabels)), values, color=color, alpha=0.85)

        ax.set_yticks(range(len(bilabels)))
        # 关键：将 CJK FontProperties 应用到每个 tick label
        ax.set_yticklabels(bilabels, fontsize=7)
        if font_prop:
            for lbl in ax.get_yticklabels():
                lbl.set_fontproperties(font_prop)
                lbl.set_fontsize(7)

        ax.invert_yaxis()
        ax.set_title(
            f"T{tid} [{lang.upper()}]  n={summary_df[summary_df['Topic_ID']==tid].iloc[0]['Topic_Size']}",
            fontsize=8,
        )
        ax.tick_params(axis="x", labelsize=6)

        # 在条形右侧标注计数
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                    str(val), va="center", fontsize=5.5, color="#444")

    for j in range(len(show), len(flat)):
        flat[j].axis("off")

    from matplotlib.patches import Patch
    fig2.legend(
        handles=[Patch(color=v, label=k.upper()) for k, v in LANG_COLORS.items()],
        loc="lower center", ncol=4, fontsize=9, title="Language"
    )
    fig2.suptitle("RQ1: Top Targets per Topic  [原文 (English translation)]",
                  fontsize=12, y=1.01)
    plt.tight_layout(h_pad=3.0, w_pad=2.0)
    p2 = output_dir / "rq1_topic_top_targets_bilingual.png"
    fig2.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    logger.info(f"    ✓ {p2}")

    # ══════════════════════════════════════════════════════════
    # 图3：全局 Top-20 实体排行（双语，按语言着色）
    # ══════════════════════════════════════════════════════════
    logger.info("  [VIZ] 图3: 全局 Top-20 实体排行（双语）")

    # 按语言分别统计全局 Top-20
    lang_entity_counts: dict[str, Counter] = {"en": Counter(), "zh": Counter(), "ja": Counter()}
    if "lang" in df_full.columns and "hate_targets" in df_full.columns:
        for _, row in df_full[df_full["topic"] != -1].iterrows():
            lang = normalize_lang(str(row.get("lang", "other")).lower())
            if lang not in ("en", "zh", "ja"): continue
            for t in pl(row["hate_targets"]):
                lang_entity_counts[lang][t] += 1

    fig3, axes3 = plt.subplots(1, 3, figsize=(18, 7))
    for ax3, (lang, counter) in zip(axes3, lang_entity_counts.items()):
        top20 = counter.most_common(20)
        if not top20:
            ax3.axis("off"); continue
        ents  = [e[0] for e in top20]
        cnts  = [e[1] for e in top20]
        # 批量翻译（利用之前缓存的 trans_map，不足的部分追加翻译）
        extra_trans = translate_entities(
            [e for e in ents if e not in trans_map],
            api_key=translate_api_key, use_api=bool(translate_api_key)
        )
        trans_map.update(extra_trans)

        bilabels = [bilingual_label(e, trans_map.get(e)) for e in ents]
        color    = LANG_COLORS[lang]
        bars3    = ax3.barh(range(len(bilabels)), cnts, color=color, alpha=0.85)
        ax3.set_yticks(range(len(bilabels)))
        ax3.set_yticklabels(bilabels, fontsize=8)
        if font_prop:
            for lbl in ax3.get_yticklabels():
                lbl.set_fontproperties(font_prop)
                lbl.set_fontsize(8)
        ax3.invert_yaxis()
        ax3.set_title(f"{lang.upper()} — Top 20 Targets", fontsize=11, fontweight="bold")
        ax3.set_xlabel("Frequency")
        for bar, val in zip(bars3, cnts):
            ax3.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                     str(val), va="center", fontsize=6.5, color="#444")

    fig3.suptitle("RQ1: Global Top-20 Hate Targets by Language  [原文 (EN translation)]",
                  fontsize=13, y=1.02)
    plt.tight_layout()
    p3 = output_dir / "rq1_global_top20_bilingual.png"
    fig3.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close(fig3)
    logger.info(f"    ✓ {p3}")

    # ══════════════════════════════════════════════════════════
    # 图4：语言 × 类别热力图（纯英文，标签已是英文）
    # ══════════════════════════════════════════════════════════
    logger.info("  [VIZ] 图4: 语言×类别热力图")
    try:
        import seaborn as sns
        rows_lc = []
        if "hate_targets_detail" in df_full.columns:
            for _, row in df_full[df_full["topic"] != -1].iterrows():
                lang = normalize_lang(str(row.get("lang", "other")).lower())
                if lang not in ("en", "zh", "ja"): continue
                for d in pl(row["hate_targets_detail"]):
                    pts = str(d).split("|")
                    if len(pts) >= 2:
                        rows_lc.append({"lang": lang.upper(), "category": pts[1]})
        if rows_lc:
            lc_df   = pd.DataFrame(rows_lc)
            hm      = lc_df.groupby(["lang", "category"]).size().unstack(fill_value=0)
            hm_pct  = hm.div(hm.sum(axis=1), axis=0)
            fig4, ax4 = plt.subplots(figsize=(11, 3.8))
            sns.heatmap(hm_pct, annot=True, fmt=".1%",
                        cmap="YlGnBu", ax=ax4, linewidths=0.5,
                        annot_kws={"size": 10})
            ax4.set_title("RQ1: Target Category Distribution by Language (%)", fontsize=12)
            ax4.set_xlabel("")
            ax4.set_ylabel("Language")
            plt.tight_layout()
            p4 = output_dir / "rq1_lang_category_heatmap.png"
            fig4.savefig(p4, dpi=150, bbox_inches="tight")
            plt.close(fig4)
            logger.info(f"    ✓ {p4}")
        else:
            logger.warning("    热力图跳过（无数据）")
    except ImportError:
        logger.warning("    热力图跳过（seaborn 未安装）")

    # ══════════════════════════════════════════════════════════
    # 图5：LLM 增益对比（仅当跑过 LLM 层时）
    # ══════════════════════════════════════════════════════════
    if "llm_enriched" in df_full.columns and df_full["llm_enriched"].any():
        logger.info("  [VIZ] 图5: LLM 增益对比")
        ent_counts = df_full["hate_targets"].apply(lambda x: len(pl(x)) if not isinstance(x, list) else len(x))
        enr = df_full["llm_enriched"].astype(bool)
        fig5, ax5 = plt.subplots(figsize=(7, 4))
        ax5.hist(ent_counts[~enr], bins=range(0, 15), alpha=0.7, label="Layer 1+2 only", color="#4E79A7")
        ax5.hist(ent_counts[enr],  bins=range(0, 15), alpha=0.7, label="+ LLM",          color="#E15759")
        ax5.set_xlabel("Entities per Document")
        ax5.set_ylabel("Count")
        ax5.set_title("RQ1: Entity Count Distribution — Layer 1+2 vs LLM Enriched")
        ax5.legend()
        plt.tight_layout()
        p5 = output_dir / "rq1_llm_gain.png"
        fig5.savefig(p5, dpi=150, bbox_inches="tight")
        plt.close(fig5)
        logger.info(f"    ✓ {p5}")

    logger.info(f"  ✅ 所有图表已保存至 {output_dir}")


# ============================================================
# CLI 入口
# ============================================================

if __name__ == "__main__":
    logger, log_path = setup_logging(name="rq1_target_extraction")
    import sys as _sys
    _sys.modules[__name__].__dict__["logger"] = logger
    logger.info(f"日志: {log_path}")

    load_env()

    parser = argparse.ArgumentParser(
        description="RQ1 Target Extraction v3 — 三阶段隔离管线",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--stage", choices=["layer12", "llm", "viz", "full"],
        default="layer12",
        help=(
            "layer12 : 运行 spaCy + 词典，保存 checkpoint_layer12.csv\n"
            "llm     : 读取 layer12 checkpoint，并发 LLM 兜底，保存 checkpoint_llm.csv\n"
            "viz     : 从最新 checkpoint 生成所有图表\n"
            "full    : 依次运行 layer12 → llm → viz"
        ),
    )
    parser.add_argument(
        "--input", type=str, default=str(DEFAULT_INPUT),
        help="输入 CSV（含 text/lang/topic 列），仅 layer12/full 阶段使用",
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(DATA_DIR),
        help="输出目录（checkpoint 和图表均保存于此）",
    )
    parser.add_argument(
        "--spacy-fallback", action="store_true",
        help="使用轻量 spaCy 模型（显存不足时）",
    )
    parser.add_argument(
        "--llm-type", default="gemini", choices=["gemini", "openai"],
        help="LLM 类型（API Key 从 .env 读取）",
    )
    parser.add_argument(
        "--llm-model", default="gemini-2.5-flash-lite",
        help="LLM 模型名（默认 gemini-2.5-flash-lite）",
    )
    parser.add_argument(
        "--concurrency", type=int, default=50,
        help="LLM 并发数（默认 50，1000 RPM 配额下可调高至 80）",
    )
    parser.add_argument(
        "--min-entities", type=int, default=2,
        help="Layer1+2 实体数低于此值时才触发 LLM 兜底（默认 2）",
    )
    parser.add_argument(
        "--top-n", type=int, default=10,
        help="每 Topic 展示 Top-N 实体（默认 10）",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="viz 阶段：手动指定 checkpoint 文件路径",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    ckpt_l12 = out_dir / "checkpoint_layer12.csv"
    ckpt_llm = out_dir / "checkpoint_llm.csv"

    stage = args.stage

    if stage in ("layer12", "full"):
        run_layer12(
            input_csv=Path(args.input),
            output_csv=ckpt_l12,
            use_spacy_fallback=args.spacy_fallback,
        )

    if stage in ("llm", "full"):
        run_llm(
            input_csv=ckpt_l12,
            output_csv=ckpt_llm,
            api_type=args.llm_type,
            model_name=args.llm_model,
            min_entities=args.min_entities,
            concurrency=args.concurrency,
        )

    if stage in ("viz", "full"):
        ckpt_arg = Path(args.checkpoint) if args.checkpoint else None
        # full 模式下：若 LLM 跑过则用 LLM checkpoint，否则用 layer12
        if ckpt_arg is None and stage == "full":
            ckpt_arg = ckpt_llm if ckpt_llm.exists() else ckpt_l12
        run_viz(
            checkpoint_csv=ckpt_arg,
            output_dir=out_dir,
            top_n=args.top_n,
        )

    logger.info("全部完成。")
