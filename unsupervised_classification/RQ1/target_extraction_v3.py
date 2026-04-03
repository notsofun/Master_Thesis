"""
RQ1 Target Extraction v3 — 三层混合管线 (Hybrid Pipeline)
==========================================================
解决 GLiNER 对中日文实体边界识别差的问题。

架构设计：
  Layer 1: spaCy NER — 利用语言专属模型精准切分实体边界 (PER/ORG/GPE)
  Layer 2: 领域词典匹配 — 捕获 spaCy 无法识别的宗教/社会群体实体
  Layer 3: LLM 结构化提取 — 对 Layer 1+2 遗漏的文本，用 Gemini/GPT 做最后兜底

后处理：代词过滤 + 长度裁剪 + 别名归一化 + 去重

# 最简用法（不用 LLM，纯 spaCy + 词典）
python unsupervised_classification/RQ1/target_extraction_v3.py --no-llm

# 完整管线（含 Gemini 兜底）
python unsupervised_classification/RQ1/target_extraction_v3.py --llm-type gemini --llm-key YOUR_KEY

# 如果 GPU 显存不足，用轻量 spaCy 模型
python unsupervised_classification/RQ1/target_extraction_v3.py --spacy-fallback --no-llm

作者: Zhidian
日期: 2026-04
"""

import pandas as pd
import re
import os
import json
import time
from collections import Counter, defaultdict
from tqdm import tqdm

# ============================================================
# 第零层：配置
# ============================================================

# spaCy 模型（需先安装）：
#   python -m spacy download en_core_web_trf
#   python -m spacy download zh_core_web_trf
#   python -m spacy download ja_core_news_trf
# 如果 GPU 显存不足，可替换为 _sm 或 _md 版本，但准确率会下降

SPACY_MODELS = {
    "en": "en_core_web_trf",   # 英文 Transformer 模型
    "zh": "zh_core_web_trf",   # 中文 Transformer 模型
    "ja": "ja_core_news_trf",  # 日文 Transformer 模型
}

# 如果 trf 模型太重，退而求其次用轻量版
SPACY_MODELS_FALLBACK = {
    "en": "en_core_web_sm",
    "zh": "zh_core_web_sm",
    "ja": "ja_core_news_sm",
}

# spaCy NER 只取这些类型（过滤掉 DATE/CARDINAL 等噪声）
SPACY_KEEP_LABELS = {
    "en": {"PERSON", "ORG", "NORP", "GPE", "FAC"},  # NORP = 民族/宗教/政治团体
    "zh": {"PERSON", "ORG", "GPE", "NORP"},
    "ja": {"Person", "Organization", "Country", "City",
           "PERSON", "ORG", "GPE", "NORP",  # Stanza 格式兼容
           "人名", "組織名", "地名"},          # ja_core_news 格式
}

# 三语代词黑名单（GLiNER 的重灾区）
PRONOUN_BLACKLIST = {
    # 英文
    "i", "me", "my", "you", "your", "he", "him", "his", "she", "her",
    "we", "us", "our", "they", "them", "their", "it", "its",
    "one", "who", "that", "this", "these", "those", "someone", "anyone",
    "everyone", "nobody", "people", "person", "man", "woman", "men", "women",
    # 中文
    "我", "你", "他", "她", "它", "我们", "你们", "他们", "她们",
    "自己", "别人", "大家", "有人", "某人", "人们", "人家",
    # 日文
    "私", "僕", "俺", "あなた", "君", "彼", "彼女", "彼ら",
    "我々", "自分", "皆", "誰", "誰か", "みんな", "人々",
}

# 通用噪声词（出现在结果中但不是实体的词）
NOISE_BLACKLIST = {
    "god", "religion", "christianity", "faith", "bible", "prayer",
    "宗教", "信仰", "圣经", "宗教信仰",
    "宗教", "信仰", "聖書",
}

# ============================================================
# 第一层：领域词典 (Domain Gazetteer)
# ============================================================

# 这些是你研究中的核心靶子词典——请根据你的预分析结果持续补充
DOMAIN_GAZETTEER = {
    # --- 宗教群体 ---
    "Religious Group": [
        # EN
        "catholics", "catholic", "protestants", "protestant", "evangelicals",
        "evangelical", "mormons", "mormon", "baptists", "baptist",
        "methodists", "adventists", "adventist", "pentecostals",
        "orthodox", "anglicans", "lutherans", "presbyterians",
        "jehovah's witnesses", "quakers", "amish", "mennonites",
        "sunni", "shia", "muslims", "muslim", "jews", "jewish",
        "hindus", "buddhists", "sikhs", "atheists",
        # ZH
        "天主教徒", "新教徒", "基督徒", "基督教徒", "东正教徒",
        "穆斯林", "佛教徒", "犹太人", "信徒", "教徒",
        "福音派", "浸信会", "卫理公会", "长老会",
        "二世信徒", "邪教徒",
        # JA
        "カトリック", "プロテスタント", "福音派", "モルモン",
        "エホバの証人", "ムスリム", "ユダヤ人", "仏教徒",
        "信者", "教徒", "二世信者", "信徒",
    ],

    # --- 宗教组织/教会 ---
    "Organization": [
        # EN
        "catholic church", "roman catholic church", "vatican",
        "church of england", "anglican church", "southern baptist convention",
        "united church", "adventist church",
        "planned parenthood", "supreme court",
        # ZH
        "天主教会", "教会", "教廷", "梵蒂冈",
        "统一教会", "全能神教会", "法轮功", "创价学会",
        "自民党", "共产党",
        # JA
        "統一教会", "世界平和統一家庭連合", "創価学会", "日本会議",
        "自民党", "全能神教会", "幸福の科学",
        "オウム真理教", "エホバの証人",
        "キリスト教福音宣教会",
    ],

    # --- 特定人物 ---
    "Specific Person": [
        # EN
        "trump", "donald trump", "biden", "obama", "pope francis",
        "pope benedict", "pope", "francis", "luther", "martin luther",
        "paul", "peter", "moses",
        # ZH
        "特朗普", "拜登", "方济各", "教宗", "教皇",
        "孔庆东", "汪海林", "洪秀全", "耶稣", "保罗",
        "安倍", "习近平",
        # JA
        "安倍", "安倍晋三", "トランプ", "バイデン",
        "フランシスコ教皇", "イエス", "キリスト",
        "山上", "山上徹也", "岸田", "麻生",
        "豊臣秀吉", "織田信長",
    ],

    # --- 社会身份/蔑称 ---
    "Social Identity": [
        # EN
        "liberals", "conservatives", "progressives", "trads",
        "traditionalists", "fundamentalists", "white christians",
        "gay people", "gays", "lgbtq",
        # ZH
        "圣母婊", "神棍", "基督狗", "左派", "右派",
        "保守派", "自由派", "女权", "公知",
        # JA
        "カルト", "保守派", "リベラル", "左翼", "右翼",
        "スパイ", "工作員", "反日",
        "保守派クリスチャン", "保守派天主教徒",
    ],

    # --- 政治群体 ---
    "Political Group": [
        # EN
        "republicans", "democrats", "gop", "maga",
        # ZH
        "共产党", "国民党",
        # JA
        "自民党", "公明党", "共産党", "立憲民主党",
        "ユダヤ国際金融資本",
    ],
}

# 构建一个反向索引：entity_text -> category（用于快速查找）
def build_gazetteer_index(gazetteer):
    """构建词典的反向索引，按长度降序排列以支持最长匹配"""
    index = {}
    for category, entities in gazetteer.items():
        for ent in entities:
            index[ent.lower()] = category
    # 按长度降序排列，确保最长匹配优先
    sorted_entries = sorted(index.items(), key=lambda x: len(x[0]), reverse=True)
    return sorted_entries

GAZETTEER_INDEX = build_gazetteer_index(DOMAIN_GAZETTEER)


# ============================================================
# 第二层：别名归一化表 (Alias Normalization)
# ============================================================

# 将各种写法/简称/蔑称统一到规范形式
ALIAS_MAP = {
    # 英文别名
    "catholic church": "Catholic Church",
    "roman catholic church": "Catholic Church",
    "the church": "Church",
    "church": "Church",
    "cc": "Catholic Church",
    "pope francis": "Pope Francis",
    "the pope": "Pope Francis",
    "pope": "Pope",
    "donald trump": "Trump",
    "trump": "Trump",
    "jesus": "Jesus",
    "jesus christ": "Jesus",
    "christ": "Jesus",
    "god": "God",

    # 中文别名
    "天主教会": "天主教会",
    "天主教": "天主教会",
    "教会": "教会",
    "教廷": "梵蒂冈",
    "梵蒂冈": "梵蒂冈",
    "耶稣": "耶稣",
    "基督": "基督",
    "神父": "神父",
    "安倍": "安倍晋三",

    # 日文别名
    "統一教会": "統一教会",
    "世界平和統一家庭連合": "統一教会",
    "旧統一教会": "統一教会",
    "創価学会": "創価学会",
    "創価": "創価学会",
    "自民党": "自民党",
    "安倍": "安倍晋三",
    "安倍晋三": "安倍晋三",
    "キリスト": "キリスト",
    "イエス": "イエス・キリスト",
    "カトリック": "カトリック",
    "プロテスタント": "プロテスタント",
}


# ============================================================
# Layer 1: spaCy NER（精准边界）
# ============================================================

def load_spacy_models(use_fallback=False):
    """加载三语 spaCy 模型"""
    import spacy
    models = {}
    model_map = SPACY_MODELS_FALLBACK if use_fallback else SPACY_MODELS

    for lang, model_name in model_map.items():
        try:
            models[lang] = spacy.load(model_name)
            print(f"  ✓ 已加载 {model_name}")
        except OSError:
            print(f"  ✗ 未找到 {model_name}，尝试 fallback...")
            try:
                fb = SPACY_MODELS_FALLBACK[lang]
                models[lang] = spacy.load(fb)
                print(f"  ✓ 已加载 fallback: {fb}")
            except OSError:
                print(f"  ✗ {lang} 无可用模型，将跳过 spaCy NER")
                models[lang] = None
    return models


def extract_spacy_entities(text, lang, spacy_models):
    """用 spaCy 提取标准 NER 实体"""
    nlp = spacy_models.get(lang)
    if nlp is None:
        return []

    # 截断过长文本（spaCy trf 模型有 512 token 限制）
    max_chars = 1000 if "trf" in str(type(nlp)) else 5000
    text_truncated = text[:max_chars]

    try:
        doc = nlp(text_truncated)
    except Exception as e:
        return []

    keep_labels = SPACY_KEEP_LABELS.get(lang, set())
    entities = []

    for ent in doc.ents:
        if ent.label_ in keep_labels:
            clean_text = ent.text.strip()
            # 基本质量检查
            if len(clean_text) >= 2 and len(clean_text) <= 30:
                entities.append({
                    "text": clean_text,
                    "label": ent.label_,
                    "source": "spacy",
                })
    return entities


# ============================================================
# Layer 2: 词典匹配 (Gazetteer Matching)
# ============================================================

def extract_gazetteer_entities(text, lang):
    """用领域词典做最长匹配提取"""
    text_lower = text.lower()
    entities = []
    found_spans = []  # 记录已匹配的区间，避免重叠

    for term, category in GAZETTEER_INDEX:
        # 对英文做词边界匹配；对中日文做子串匹配
        if lang == "en":
            pattern = r'\b' + re.escape(term) + r'\b'
            matches = list(re.finditer(pattern, text_lower))
        else:
            # 中日文直接子串查找
            start = 0
            matches = []
            while True:
                idx = text_lower.find(term, start)
                if idx == -1:
                    break
                # 创建一个 match-like 对象
                matches.append(type('Match', (), {'start': lambda s=idx: s, 'end': lambda e=idx+len(term): e})())
                start = idx + len(term)

        for m in matches:
            span = (m.start(), m.end())
            # 检查是否与已有匹配重叠
            overlaps = any(
                not (span[1] <= existing[0] or span[0] >= existing[1])
                for existing in found_spans
            )
            if not overlaps:
                found_spans.append(span)
                # 取原文中的实际大小写形式
                original_text = text[span[0]:span[1]]
                entities.append({
                    "text": original_text.strip(),
                    "label": category,
                    "source": "gazetteer",
                })

    return entities


# ============================================================
# Layer 3: LLM 结构化提取（兜底）
# ============================================================

def build_llm_prompt(text, lang):
    """构建 LLM 提取 Prompt（多语言适配）"""

    lang_instruction = {
        "en": "English",
        "zh": "Chinese",
        "ja": "Japanese",
    }

    prompt = f"""You are an expert NER system for hate speech analysis.
Extract ONLY the specific named entities (people, groups, organizations) that are
TARGETS of discussion or hatred in the following {lang_instruction.get(lang, 'multilingual')} text.

STRICT RULES:
1. Extract ONLY proper nouns or established group names (e.g., "安倍", "統一教会", "Catholics")
2. NEVER extract full sentences or clauses
3. NEVER extract verbs, adjectives, or descriptive phrases
4. Each entity must be ≤ 5 words / 10 characters (CJK) / 30 characters (Latin)
5. Do NOT extract pronouns (I, you, he, 我, 你, 私, etc.)
6. Do NOT extract generic words like "religion", "god", "faith", "宗教", "信仰"

Output as JSON array of objects: [{{"text": "entity", "category": "one of: Religious Group / Organization / Specific Person / Social Identity / Political Group"}}]
If no valid entities found, return: []

Text: {text}

JSON:"""
    return prompt


def extract_llm_entities_batch(texts_with_langs, api_type="gemini", api_key=None):
    """
    批量调用 LLM 提取实体

    支持:
      - "gemini": Google Gemini API (你项目中已有)
      - "openai": OpenAI API
      - "local": 本地 Qwen2.5 / Llama (通过 transformers)

    注意：这一层是兜底层，只在 Layer1 + Layer2 提取数量不足时才调用
    """
    results = []

    if api_type == "gemini":
        try:
            import google.generativeai as genai
            if api_key:
                genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-2.0-flash")

            for text, lang in tqdm(texts_with_langs, desc="LLM 兜底提取"):
                prompt = build_llm_prompt(text, lang)
                try:
                    response = model.generate_content(prompt)
                    response_text = response.text.strip()
                    # 提取 JSON 部分
                    json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
                    if json_match:
                        entities_raw = json.loads(json_match.group())
                        entities = []
                        for e in entities_raw:
                            if isinstance(e, dict) and "text" in e:
                                entities.append({
                                    "text": e["text"].strip(),
                                    "label": e.get("category", "Unknown"),
                                    "source": "llm",
                                })
                        results.append(entities)
                    else:
                        results.append([])
                except Exception as e:
                    results.append([])

                time.sleep(0.1)  # Rate limit

        except ImportError:
            print("  ⚠ google-generativeai 未安装，跳过 LLM 层")
            results = [[] for _ in texts_with_langs]

    elif api_type == "openai":
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key) if api_key else OpenAI()

            for text, lang in tqdm(texts_with_langs, desc="LLM 兜底提取"):
                prompt = build_llm_prompt(text, lang)
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0,
                        max_tokens=256,
                    )
                    response_text = response.choices[0].message.content.strip()
                    json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
                    if json_match:
                        entities_raw = json.loads(json_match.group())
                        entities = [
                            {"text": e["text"].strip(), "label": e.get("category", "Unknown"), "source": "llm"}
                            for e in entities_raw if isinstance(e, dict) and "text" in e
                        ]
                        results.append(entities)
                    else:
                        results.append([])
                except Exception:
                    results.append([])
                time.sleep(0.05)

        except ImportError:
            print("  ⚠ openai 未安装，跳过 LLM 层")
            results = [[] for _ in texts_with_langs]

    elif api_type == "local":
        # 本地模型兜底（如 Qwen2.5-7B-Instruct）
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            model_name = "Qwen/Qwen2.5-7B-Instruct"
            print(f"  加载本地模型 {model_name}...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float16, device_map="auto"
            )

            for text, lang in tqdm(texts_with_langs, desc="本地 LLM 提取"):
                prompt = build_llm_prompt(text, lang)
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.1)
                response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                # 截取 prompt 之后的部分
                response_text = response_text[len(prompt):].strip()
                try:
                    json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
                    if json_match:
                        entities_raw = json.loads(json_match.group())
                        entities = [
                            {"text": e["text"].strip(), "label": e.get("category", "Unknown"), "source": "llm"}
                            for e in entities_raw if isinstance(e, dict) and "text" in e
                        ]
                        results.append(entities)
                    else:
                        results.append([])
                except Exception:
                    results.append([])

        except ImportError:
            print("  ⚠ transformers 未安装，跳过本地 LLM 层")
            results = [[] for _ in texts_with_langs]

    else:
        results = [[] for _ in texts_with_langs]

    return results


# ============================================================
# 后处理管线 (Post-processing Pipeline)
# ============================================================

def clean_entity(text, lang):
    """单个实体的清洗规则"""
    text = text.strip()

    # 1. 去除首尾标点
    text = re.sub(r'^[\s\.,;:!?、。，；：！？「」『』（）\(\)\[\]]+', '', text)
    text = re.sub(r'[\s\.,;:!?、。，；：！？「」『』（）\(\)\[\]]+$', '', text)

    # 2. 长度检查（CJK vs Latin）
    if lang in ("zh", "ja"):
        if len(text) > 15 or len(text) < 1:
            return None
    else:
        if len(text) > 40 or len(text) < 2:
            return None

    # 3. 代词过滤
    if text.lower() in PRONOUN_BLACKLIST:
        return None

    # 4. 噪声词过滤
    if text.lower() in NOISE_BLACKLIST:
        return None

    # 5. 过滤明显是句子的结果（包含动词助词等）
    if lang == "ja":
        # 日文：如果包含助词 が/は/を/に/で/も/と + 动词结尾，大概率是句子
        sentence_pattern = r'.{5,}[がはをにでもと].{3,}[るたいすくけれ]$'
        if re.search(sentence_pattern, text):
            return None
        # 包含する/して/した 等动词形式
        if re.search(r'(して|した|している|しまくり|なって|グルに)', text):
            return None

    if lang == "zh":
        # 中文：如果包含 "的" + 超过 6 字，很可能是短语
        if "的" in text and len(text) > 8:
            return None
        # 包含明显的动词结构
        if re.search(r'(不住|起来|出来|下去|进去|过来|会|是|在|了|着|过|把|被|让|给)', text) and len(text) > 6:
            return None

    return text


def normalize_entity(text):
    """别名归一化"""
    normalized = ALIAS_MAP.get(text.lower(), text)
    return normalized


def deduplicate_entities(entities):
    """
    去重策略：
    1. 完全相同的 -> 合并
    2. 一个是另一个的子串 -> 保留长的
    """
    if not entities:
        return []

    # 按文本长度降序排列
    sorted_ents = sorted(entities, key=lambda x: len(x["text"]), reverse=True)
    kept = []

    for ent in sorted_ents:
        text_lower = ent["text"].lower()
        # 检查是否是已保留实体的子串
        is_substring = False
        for k in kept:
            if text_lower in k["text"].lower() or k["text"].lower() in text_lower:
                # 保留较长的那个
                if len(text_lower) <= len(k["text"]):
                    is_substring = True
                    break
        if not is_substring:
            kept.append(ent)

    return kept


# ============================================================
# 主管线 (Main Pipeline)
# ============================================================

def extract_targets_hybrid(df,
                           use_llm=True,
                           llm_api_type="gemini",
                           llm_api_key=None,
                           llm_min_threshold=2,
                           use_spacy_fallback=False):
    """
    三层混合实体提取管线

    参数:
        df: 包含 'text', 'lang', 'topic' 列的 DataFrame
        use_llm: 是否启用 LLM 兜底层
        llm_api_type: "gemini" / "openai" / "local"
        llm_api_key: API key (None 则从环境变量读取)
        llm_min_threshold: Layer1+2 提取实体数 < 此值时才触发 LLM
        use_spacy_fallback: 是否使用轻量 spaCy 模型
    """

    print("=" * 60)
    print("RQ1 Target Extraction v3 — Hybrid Pipeline")
    print("=" * 60)

    # --- 加载 spaCy ---
    print("\n[1/4] 加载 spaCy 多语言模型...")
    spacy_models = load_spacy_models(use_fallback=use_spacy_fallback)

    # --- Layer 1 + Layer 2: spaCy + 词典 ---
    print(f"\n[2/4] Layer 1 (spaCy NER) + Layer 2 (领域词典) 提取中...")
    all_entities = []
    llm_needed_indices = []  # 记录需要 LLM 兜底的行索引

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Layer 1+2"):
        text = str(row.get("text", "")).strip()
        lang = str(row.get("lang", "en")).strip().lower()

        if not text:
            all_entities.append([])
            continue

        # Layer 1: spaCy NER
        spacy_ents = extract_spacy_entities(text, lang, spacy_models)

        # Layer 2: 词典匹配
        gaz_ents = extract_gazetteer_entities(text, lang)

        # 合并
        combined = spacy_ents + gaz_ents

        # 后处理
        cleaned = []
        seen_texts = set()
        for ent in combined:
            clean_text = clean_entity(ent["text"], lang)
            if clean_text and clean_text.lower() not in seen_texts:
                norm_text = normalize_entity(clean_text)
                cleaned.append({
                    "text": norm_text,
                    "label": ent["label"],
                    "source": ent["source"],
                })
                seen_texts.add(norm_text.lower())

        # 去重
        cleaned = deduplicate_entities(cleaned)
        all_entities.append(cleaned)

        # 判断是否需要 LLM 兜底
        if len(cleaned) < llm_min_threshold:
            llm_needed_indices.append(idx)

    # --- Layer 3: LLM 兜底 ---
    if use_llm and llm_needed_indices:
        print(f"\n[3/4] Layer 3 (LLM 兜底) — 共 {len(llm_needed_indices)} 条文本需要补充提取...")

        texts_for_llm = [
            (str(df.iloc[i]["text"]), str(df.iloc[i].get("lang", "en")))
            for i in llm_needed_indices
        ]

        llm_results = extract_llm_entities_batch(
            texts_for_llm, api_type=llm_api_type, api_key=llm_api_key
        )

        for i, llm_ents in zip(llm_needed_indices, llm_results):
            existing_texts = {e["text"].lower() for e in all_entities[i]}
            lang = str(df.iloc[i].get("lang", "en"))
            for ent in llm_ents:
                clean_text = clean_entity(ent["text"], lang)
                if clean_text and clean_text.lower() not in existing_texts:
                    norm_text = normalize_entity(clean_text)
                    all_entities[i].append({
                        "text": norm_text,
                        "label": ent["label"],
                        "source": "llm",
                    })
                    existing_texts.add(norm_text.lower())
    else:
        print(f"\n[3/4] LLM 层已跳过 (use_llm={use_llm}, 需兜底: {len(llm_needed_indices)})")

    # --- 写回 DataFrame ---
    print(f"\n[4/4] 整理结果...")
    df["hate_targets"] = [
        [e["text"] for e in ents] for ents in all_entities
    ]
    df["hate_targets_detail"] = [
        [f'{e["text"]}|{e["label"]}|{e["source"]}' for e in ents]
        for ents in all_entities
    ]

    # 统计
    total_entities = sum(len(ents) for ents in all_entities)
    by_source = Counter(
        e["source"] for ents in all_entities for e in ents
    )
    print(f"\n  总提取实体数: {total_entities}")
    print(f"  来源分布: {dict(by_source)}")
    print(f"  LLM 兜底文本数: {len(llm_needed_indices)}")

    return df


# ============================================================
# 按 Topic 汇总统计
# ============================================================

def analyze_topic_targets(df, output_dir="unsupervised_classification/RQ1/data"):
    """按 Topic 统计 Top-N 攻击目标"""
    print("\n--- 汇总每个 Topic 的核心攻击目标 ---")

    valid_df = df[df["topic"] != -1].copy()
    topic_target_summary = []

    for topic_id, group in valid_df.groupby("topic"):
        all_targets = [t for sublist in group["hate_targets"] for t in sublist]
        target_counts = Counter(all_targets)
        top_10 = target_counts.most_common(10)

        # 同时统计详细信息（带类别）
        all_details = [
            d for sublist in group["hate_targets_detail"] for d in sublist
        ]
        label_counts = Counter()
        for d in all_details:
            parts = d.split("|")
            if len(parts) >= 2:
                label_counts[parts[1]] += 1

        topic_target_summary.append({
            "Topic_ID": topic_id,
            "Topic_Size": len(group),
            "Top_Targets": [f"{t[0]}({t[1]})" for t in top_10],
            "Target_Categories": dict(label_counts.most_common(5)),
            "Unique_Targets": len(target_counts),
        })

    summary_df = pd.DataFrame(topic_target_summary)

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "rq1_topic_targets_v3.csv")
    summary_df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ 已保存至 {save_path}")

    # 同时保存文档级详细结果
    detail_path = os.path.join(output_dir, "rq1_document_entities_v3.csv")
    df[["text", "lang", "topic", "hate_targets", "hate_targets_detail"]].to_csv(
        detail_path, index=False, encoding="utf-8-sig"
    )
    print(f"✅ 文档级详情已保存至 {detail_path}")

    return summary_df


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RQ1 Hybrid Target Extraction")
    parser.add_argument("--input", type=str,
                        default="unsupervised_classification/topic_modeling_results/sixth/data/document_topic_mapping.csv",
                        help="输入 CSV 路径")
    parser.add_argument("--output-dir", type=str,
                        default="unsupervised_classification/RQ1/data",
                        help="输出目录")
    parser.add_argument("--no-llm", action="store_true",
                        help="禁用 LLM 兜底层")
    parser.add_argument("--llm-type", type=str, default="gemini",
                        choices=["gemini", "openai", "local"],
                        help="LLM API 类型")
    parser.add_argument("--llm-key", type=str, default=None,
                        help="LLM API Key")
    parser.add_argument("--spacy-fallback", action="store_true",
                        help="使用轻量 spaCy 模型")
    args = parser.parse_args()

    print(f"读取数据: {args.input}")
    df = pd.read_csv(args.input)

    # 确保 lang 列存在
    if "lang" not in df.columns:
        print("⚠ 数据中没有 'lang' 列，将尝试推断...")
        # 简单启发式推断
        def guess_lang(text):
            text = str(text)
            ja_chars = len(re.findall(r'[\u3040-\u309F\u30A0-\u30FF]', text))
            zh_chars = len(re.findall(r'[\u4E00-\u9FFF]', text))
            if ja_chars > 3:
                return "ja"
            elif zh_chars > len(text) * 0.3:
                return "zh"
            return "en"
        df["lang"] = df["text"].apply(guess_lang)

    df_extracted = extract_targets_hybrid(
        df,
        use_llm=not args.no_llm,
        llm_api_type=args.llm_type,
        llm_api_key=args.llm_key,
        use_spacy_fallback=args.spacy_fallback,
    )

    summary = analyze_topic_targets(df_extracted, output_dir=args.output_dir)
    print("\n=== Top 5 Topics ===")
    print(summary.head().to_string())
