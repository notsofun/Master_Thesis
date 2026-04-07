"""
viz_utils.py — RQ1 可视化工具库
=================================
职责：
  1. CJK 字体自动探测 & 注册（跨平台 Mac/Win/Linux）
  2. 静态翻译词典（覆盖全部 DOMAIN_GAZETTEER 实体）
  3. Google Translate API 自动补全 + 本地缓存
  4. 统一的双语标签格式化函数：「統一教会 (Unification Church)」

使用方法（在可视化函数开头调用）：
  from viz_utils import setup_cjk_font, bilingual_label

作者: Zhidian  |  日期: 2026-04
"""

from __future__ import annotations
import json
import os
import re
import sys
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ============================================================
# 一、CJK 字体自动探测 & 配置
# ============================================================

# 按优先级排列的候选字体（覆盖 Mac / Windows / Linux）
_FONT_CANDIDATES = [
    # ── macOS ──
    "PingFang SC",            # 简体中文（macOS）
    "PingFang TC",            # 繁体中文
    "Hiragino Sans GB",       # 中文（macOS）
    "Hiragino Sans",          # 日文首选（macOS）
    "Apple SD Gothic Neo",    # 韩文（顺带支持CJK）
    # ── Windows ──
    "Microsoft YaHei",        # 微软雅黑（Win 中文）
    "SimHei",                 # 黑体
    "SimSun",                 # 宋体
    "Yu Gothic",              # 游ゴシック（Win 日文）
    "Meiryo",                 # メイリオ（Win 日文）
    "MS Gothic",
    # ── Linux / 通用 ──
    "Noto Sans CJK SC",       # Noto 简中
    "Noto Sans CJK TC",       # Noto 繁中
    "Noto Sans CJK JP",       # Noto 日文
    "Noto CJK",
    "Source Han Sans",        # 思源黑体
    "WenQuanYi Micro Hei",    # 文泉驿
    "Droid Sans Fallback",    # Linux 备用（只支持 CJK，不支持 Latin）
]

_CJK_FONT_PATH: Optional[str] = None   # 注册后的字体文件路径
_CJK_FONT_NAME: Optional[str] = None   # 主字体名称
_CJK_FONT_FALLBACKS: list[str] = []   # 完整 fallback chain


def _find_cjk_font_file() -> tuple[Optional[str], Optional[str]]:
    """
    遍历候选列表，返回第一个在系统中找到的 (name, path) 对。
    如果找到的是「只支持 CJK 不支持 Latin」的字体，记录下来但继续
    寻找更好的候选（支持 Latin 的字体优先）。
    """
    import matplotlib.font_manager as fm

    # 建立名称→路径的索引
    name_to_path: dict[str, str] = {
        f.name: f.fname for f in fm.fontManager.ttflist
    }

    # 同时检查文件系统上的常见路径
    extra_paths = []
    if sys.platform == "darwin":
        extra_paths = [
            "/System/Library/Fonts/",
            "/Library/Fonts/",
            str(Path.home() / "Library" / "Fonts"),
        ]
    elif sys.platform == "win32":
        extra_paths = [str(Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts")]
    else:
        extra_paths = [
            "/usr/share/fonts/",
            "/usr/local/share/fonts/",
            str(Path.home() / ".fonts"),
        ]

    # 先用 fontManager 索引查，收集所有可用的候选字体
    found_fonts = []
    fallback_droid = None
    for candidate in _FONT_CANDIDATES:
        if candidate in name_to_path:
            path = name_to_path[candidate]
            if "Droid" in candidate:
                fallback_droid = (candidate, path)   # 保底备用
                continue
            logger.info(f"[Font] 找到 CJK 字体: {candidate} ({path})")
            found_fonts.append((candidate, path))
    
    # 如果找到至少一个字体，返回第一个（但后续会配置所有的作为 fallback）
    if found_fonts:
        return found_fonts[0]

    # 再按文件名在 extra_paths 里查
    cjk_keywords = ["CJK", "Noto", "YaHei", "Gothic", "Hiragino",
                     "PingFang", "SimHei", "Meiryo", "WenQuan", "SourceHan"]
    for font_dir in extra_paths:
        for ext in ("*.ttf", "*.otf", "*.ttc"):
            for fp in Path(font_dir).rglob(ext) if Path(font_dir).exists() else []:
                if any(kw.lower() in fp.name.lower() for kw in cjk_keywords):
                    try:
                        fm.fontManager.addfont(str(fp))
                        prop = fm.FontProperties(fname=str(fp))
                        name = prop.get_name()
                        if "Droid" in name:
                            fallback_droid = (name, str(fp))
                            continue
                        logger.info(f"[Font] 扫描找到: {name} ({fp})")
                        return name, str(fp)
                    except Exception:
                        continue

    if fallback_droid:
        logger.warning(
            f"[Font] 只找到 {fallback_droid[0]}（不支持 Latin），"
            "双语标签将仅显示 CJK 部分。建议安装 Noto Sans CJK："
            "\n  macOS:   brew install --cask font-noto-sans-cjk"
            "\n  Ubuntu:  sudo apt install fonts-noto-cjk"
            "\n  pip:     pip install mplfonts && mplfonts init"
        )
        return fallback_droid

    logger.warning("[Font] 未找到任何 CJK 字体，图表中文/日文将显示为方块。")
    return None, None


def setup_cjk_font() -> Optional[str]:
    """
    配置 matplotlib 使用 CJK 字体，支持多字体 fallback chain。
    返回主字体名称（可传给 FontProperties），None 表示未找到。

    调用方式（可视化函数开头）：
        from viz_utils import setup_cjk_font
        cjk_font_name = setup_cjk_font()
    """
    global _CJK_FONT_PATH, _CJK_FONT_NAME, _CJK_FONT_FALLBACKS

    if _CJK_FONT_NAME:       # 已初始化，直接返回缓存
        return _CJK_FONT_NAME

    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm

    # 收集所有可用的候选字体
    available_fonts: list[str] = []
    name_to_path: dict[str, str] = {
        f.name: f.fname for f in fm.fontManager.ttflist
    }

    # 按優先級查找並收集所有可用字體
    for candidate in _FONT_CANDIDATES:
        if candidate in name_to_path:
            if "Droid" not in candidate:  # 先跳過只支持CJK的字體
                available_fonts.append(candidate)
                logger.info(f"[Font] 找到 CJK 字体: {candidate}")
    
    # 如果沒找到，再加上 Droid Sans Fallback
    if not available_fonts:
        for candidate in _FONT_CANDIDATES:
            if candidate in name_to_path and "Droid" in candidate:
                available_fonts.append(candidate)
                logger.warning(f"[Font] 只找到 {candidate}（仅支持CJK）")
                break

    if not available_fonts:
        logger.warning("[Font] 未找到任何 CJK 字体，图表中文/日文将显示为方块。")
        return None

    # 設定主字體（第一個找到的）
    main_font = available_fonts[0]
    _CJK_FONT_NAME = main_font
    
    # 取得主字體的路徑（用於 FontProperties）
    if main_font in name_to_path:
        _CJK_FONT_PATH = name_to_path[main_font]

    # ★ 核心改進：配置所有可用字体作為 fallback chain
    # 這樣當某個字在主字體中缺失時，會自動嘗試下一個字體
    current = plt.rcParams.get("font.sans-serif", [])
    
    # 构造新的字体列表：所有找到的 CJK 字体 + 现有的 Latin 字体
    fallback_list = available_fonts + [f for f in current if f not in available_fonts]
    
    plt.rcParams["font.sans-serif"] = fallback_list
    plt.rcParams["axes.unicode_minus"] = False

    logger.info(
        f"[Font] Fallback chain 已配置: {', '.join(available_fonts[:3])}"
        f"{'...' if len(available_fonts) > 3 else ''}"
    )
    return main_font



def get_cjk_font_prop():
    """
    返回可用于 ax.set_yticklabels(..., fontproperties=prop) 的 FontProperties 对象。
    ★ 改进：使用 family 列表，让 matplotlib 尝试整个 fallback chain。
    """
    import matplotlib.font_manager as fm
    if _CJK_FONT_FALLBACKS:
        return fm.FontProperties(family=_CJK_FONT_FALLBACKS)
    if _CJK_FONT_NAME:
        return fm.FontProperties(family=_CJK_FONT_NAME)
    if _CJK_FONT_PATH:
        return fm.FontProperties(fname=_CJK_FONT_PATH)
    return None


# ============================================================
# 二、静态翻译词典
# ============================================================

# 覆盖 DOMAIN_GAZETTEER 中的所有实体 + 预分析 top-terms 高频词
# 格式：原文（小写）→ 英文译名
STATIC_TRANSLATIONS: dict[str, str] = {

    # ── 宗教组织 (Organizations) ──
    "統一教会":           "Unification Church",
    "世界平和統一家庭連合": "Family Federation (UC)",
    "旧統一教会":         "Former Unification Church",
    "創価学会":           "Soka Gakkai",
    "日本会議":           "Nippon Kaigi",
    "幸福の科学":         "Happy Science",
    "オウム真理教":       "Aum Shinrikyo",
    "アレフ":             "Aleph (Aum successor)",
    "全能神教会":         "Church of Almighty God",
    "エホバの証人":       "Jehovah's Witnesses",
    "キリスト教福音宣教会": "Christian Gospel Mission (JMS)",
    "カトリック教会":     "Catholic Church",
    "天主教会":           "Catholic Church",
    "梵蒂冈":             "Vatican",
    "教廷":               "Holy See",
    "统一教会":           "Unification Church",
    "全能神教会":         "Church of Almighty God",
    "法轮功":             "Falun Gong",
    "创价学会":           "Soka Gakkai",
    "修道院":             "Monastery",
    "教会":               "Church",
    "教堂":               "Church (building)",
    "主教团":             "Bishops' Conference",
    "耶和华见证人":       "Jehovah's Witnesses",
    "新天地":             "Shincheonji",

    # ── 宗教群体 (Religious Groups) ──
    "カトリック":         "Catholic",
    "カトリック教徒":     "Catholic believers",
    "プロテスタント":     "Protestant",
    "プロテスタント教徒": "Protestant believers",
    "福音派":             "Evangelicals",
    "バプテスト":         "Baptists",
    "メソジスト":         "Methodists",
    "エホバの証人":       "Jehovah's Witnesses",
    "ムスリム":           "Muslims",
    "ユダヤ人":           "Jews",
    "仏教徒":             "Buddhists",
    "信者":               "Believers",
    "教徒":               "Church members",
    "信徒":               "Congregation",
    "カルト信者":         "Cult members",
    "カルト":             "Cult",
    "二世信者":           "Second-generation believers",
    "宗教二世":           "Religious 2nd generation",
    "キリスト教徒":       "Christians",
    "正教会信徒":         "Orthodox believers",
    "保守派クリスチャン": "Conservative Christians",
    "天主教徒":           "Catholics",
    "新教徒":             "Protestants",
    "基督徒":             "Christians",
    "基督教徒":           "Christians",
    "穆斯林":             "Muslims",
    "犹太人":             "Jews",
    "佛教徒":             "Buddhists",
    "信徒":               "Believers",
    "教徒":               "Church members",
    "二世信徒":           "2nd-gen believers",
    "邪教徒":             "Cult followers",
    "保守派基督徒":       "Conservative Christians",
    "保守派天主教徒":     "Conservative Catholics",

    # ── 特定人物 (Specific Persons) ──
    "安倍晋三":           "Abe Shinzo",
    "安倍":               "Abe (Shinzo)",
    "安倍元首相":         "Former PM Abe",
    "岸田":               "Kishida (Fumio)",
    "岸田文雄":           "Kishida Fumio",
    "麻生":               "Aso (Taro)",
    "麻生太郎":           "Aso Taro",
    "山上":               "Yamagami (shooter)",
    "山上徹也":           "Yamagami Tetsuya",
    "トランプ":           "Trump",
    "バイデン":           "Biden",
    "フランシスコ教皇":   "Pope Francis",
    "イエス・キリスト":   "Jesus Christ",
    "イエス":             "Jesus",
    "キリスト":           "Christ",
    "パウロ":             "Paul (Apostle)",
    "ペテロ":             "Peter (Apostle)",
    "豊臣秀吉":           "Toyotomi Hideyoshi",
    "织田信长":           "Oda Nobunaga",
    "耶稣":               "Jesus",
    "教皇":               "Pope",
    "教宗":               "Pope",
    "方济各":             "Francis (Pope)",
    "孔庆东":             "Kong Qingdong",
    "汪海林":             "Wang Hailin",
    "习近平":             "Xi Jinping",
    "洪秀全":             "Hong Xiuquan",

    # ── 社会身份 / 蔑称 (Social Identity) ──
    "カルト":             "Cult",
    "スパイ":             "Spy",
    "工作員":             "Agent (infiltrator)",
    "反日":               "Anti-Japan",
    "売国奴":             "Traitor",
    "売国":               "Traitor / selling out the nation",
    "保守派":             "Conservatives",
    "リベラル":           "Liberals",
    "左翼":               "Left-wing",
    "右翼":               "Right-wing",
    "悪魔":               "Devil / Satan",
    "サタン":             "Satan",
    "異端":               "Heretic",
    "背教者":             "Apostate",
    "神父":               "Priest (Father)",
    "牧師":               "Pastor",
    "修道士":             "Monk",
    "修道女":             "Nun",
    "司教":               "Bishop",
    "連中":               "Those people / bunch",
    "圣母婊":             "Sanctimonious b*tch",
    "神棍":               "Fraudulent cleric",
    "基督狗":             "Christ-dog (slur)",
    "修女":               "Nun",
    "牧师":               "Pastor",
    "传教士":             "Missionary",
    "主教":               "Bishop",
    "异端":               "Heretic",

    # ── 政治群体 (Political Groups) ──
    "自民党":             "LDP (Liberal Democratic Party)",
    "公明党":             "Komeito",
    "立憲民主党":         "Constitutional Democratic Party",
    "日本共産党":         "Japanese Communist Party",
    "維新の会":           "Nippon Ishin (Innovation Party)",
    "共産党":             "Communist Party",
    "中国共产党":         "Chinese Communist Party",
    "国民党":             "Kuomintang (KMT)",
    "韓国":               "South Korea",
    "中国":               "China",
    "ロシア":             "Russia",
    "ユダヤ":             "Jews / Jewish",
    "日本人":             "Japanese people",
    "ユダヤ国際金融資本": "International Jewish Finance Capital",
    "ユダヤ共産主義":     "Judeo-Communism",

    # ── 英文实体（已是英文的不需翻译，但提供规范形式）──
    "catholic church":    "Catholic Church",
    "vatican":            "Vatican",
    "jesuits":            "Jesuits",
    "opus dei":           "Opus Dei",
    "jehovah's witnesses": "Jehovah's Witnesses",
    "planned parenthood": "Planned Parenthood",
    "southern baptist convention": "Southern Baptist Convention",
}

# 缓存文件（避免重复调用 API）
_CACHE_FILE = Path(__file__).parent / "data" / "translation_cache.json"


def _load_cache() -> dict:
    if _CACHE_FILE.exists():
        try:
            with open(_CACHE_FILE, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_cache(cache: dict):
    _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def _gemini_translate(texts: list[str], api_key: str | None = None) -> dict[str, str]:
    """
    批量调用 Gemini API 做翻译，返回 {原文: 译文}。
    若无 API key 或请求失败则返回空字典。
    """
    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return {}

    try:
        from google import genai
    except ImportError:
        logger.warning("[Translate] google-genai 未安装，跳过 Gemini 翻译")
        return {}

    results: dict[str, str] = {}
    BATCH = 50  # Gemini 批量翻译，保守估计

    for i in range(0, len(texts), BATCH):
        batch = texts[i: i + BATCH]
        prompt = (
            "Translate the following Chinese/Japanese/mixed text entities to English. "
            "Return ONLY the translations, one per line, in the exact same order as input. "
            "Do NOT add explanations, numbers, or any other text.\n\n"
            + "\n".join(batch)
        )

        try:
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            trans_lines = response.text.strip().split("\n")

            # 逐行对应
            for orig, trans in zip(batch, trans_lines):
                trans = trans.strip()
                if trans:
                    results[orig] = trans
                else:
                    results[orig] = orig  # 回退：若翻译为空则用原文

        except Exception as e:
            logger.warning(f"[Translate] Gemini 批次失败 (行 {i}-{i+len(batch)}): {e}")
            # 单条重试
            for orig in batch:
                if orig not in results:
                    try:
                        prompt_single = f"Translate to English:\n{orig}\n\nReturn ONLY the translation."
                        response = client.models.generate_content(
                            model="gemini-2.5-flash",
                            contents=prompt_single,
                        )
                        trans = response.text.strip().split("\n")[0].strip()
                        results[orig] = trans if trans else orig
                    except Exception:
                        results[orig] = orig

    return results


def translate_entities(
    entities: list[str],
    api_key: str | None = None,
    use_api: bool = True,
) -> dict[str, str]:
    """
    将实体列表翻译成英文。
    优先级：静态词典 → 本地缓存 → Gemini API

    返回：{原文: 英文译名}
    """
    translations: dict[str, str] = {}
    need_api: list[str] = []

    cache = _load_cache()

    for ent in entities:
        ent_stripped = ent.strip()
        key = ent_stripped.lower()

        # 1. 静态词典
        if key in STATIC_TRANSLATIONS:
            translations[ent] = STATIC_TRANSLATIONS[key]
        # 2. 本地缓存
        elif key in cache:
            translations[ent] = cache[key]
        # 3. 已是纯英文 → 直接返回原文
        elif re.match(r'^[A-Za-z0-9\s\-\'\.]+$', ent_stripped):
            translations[ent] = ent_stripped.title()
        else:
            need_api.append(ent)

    # 4. 批量调用 Gemini API
    if need_api and use_api:
        api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if api_key:
            api_results = _gemini_translate(need_api, api_key=api_key)
            for orig, trans in api_results.items():
                translations[orig] = trans
                cache[orig.lower()] = trans
            if api_results:
                _save_cache(cache)
                logger.info(f"[Translate] Gemini API 翻译了 {len(api_results)} 个新词，已缓存")
        else:
            logger.debug("[Translate] 无 GEMINI_API_KEY，跳过 API 翻译")
            for ent in need_api:
                translations[ent] = ent   # 回退：原文

    # 填充剩余未翻译的
    for ent in entities:
        if ent not in translations:
            translations[ent] = ent

    return translations


# ============================================================
# 三、双语标签格式化
# ============================================================

def bilingual_label(
    entity: str,
    translation: str | None = None,
    max_cjk_len: int = 12,
    max_en_len: int = 22,
    show_translation: bool = True,
) -> str:
    """
    生成双语标签：
      「統一教会 (Unification Church)」
      「神父 (Priest)」
      「Catholics」  ← 已是英文则不重复

    参数:
        entity          原始实体文本
        translation     英文翻译（None 则从静态词典查）
        max_cjk_len     CJK 部分最大长度（超出截断）
        max_en_len      英文部分最大长度（超出截断）
        show_translation 是否展示翻译（False 则仅返回原文）
    """
    entity = entity.strip()

    # 判断是否需要翻译（纯 ASCII 不需要）
    is_cjk = bool(re.search(r'[\u3000-\u9FFF\uAC00-\uD7FF\uF900-\uFAFF]', entity))

    if not is_cjk or not show_translation:
        # 英文实体直接返回（截断超长）
        return entity[:max_en_len + 3] + "…" if len(entity) > max_en_len else entity

    # 取翻译
    if translation is None:
        key = entity.lower()
        translation = STATIC_TRANSLATIONS.get(key, entity)

    # 截断
    cjk_part = entity[:max_cjk_len] + ("…" if len(entity) > max_cjk_len else "")
    en_part   = translation[:max_en_len] + ("…" if len(translation) > max_en_len else "")

    return f"{cjk_part}\n({en_part})"


def bilingual_labels_batch(
    entities: list[str],
    api_key: str | None = None,
    use_api: bool = True,
    **kwargs,
) -> list[str]:
    """
    批量生成双语标签（自动调用 translate_entities）。
    """
    trans_map = translate_entities(entities, api_key=api_key, use_api=use_api)
    return [bilingual_label(e, trans_map.get(e), **kwargs) for e in entities]


# ============================================================
# 四、统一 matplotlib 初始化入口
# ============================================================

def setup_matplotlib(style: str = "seaborn-v0_8-whitegrid") -> str | None:
    """
    一键初始化：设置 CJK 字体 + 图表风格。
    返回 CJK 字体名（可能为 None）。

    使用：
        from viz_utils import setup_matplotlib
        cjk = setup_matplotlib()
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        plt.style.use(style)
    except Exception:
        plt.style.use("ggplot")

    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 150

    return setup_cjk_font()
