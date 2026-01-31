import json
import random
import itertools

# ==========================================
# 1. 定义核心词库 (基于 Tieba/5ch/Gab 文件提取)
# ==========================================
subjects = [
    {"en": "That christcuck", "zh": "那个基督狗", "ja": "あのキリスト厨"},
    {"en": "The priest", "zh": "那个神棍", "ja": "あの司祭"},
    {"en": "This believer", "zh": "这个信徒", "ja": "この信者"},
    {"en": "Your god", "zh": "你们的上帝", "ja": "お前らの神"},
]

predicates = [
    {"en": "is just a brainwashed puppet.", "zh": "不过是个被洗脑的傀儡。", "ja": "はただの洗脳されたパペットだ。"},
    {"en": "spreads nothing but lies.", "zh": "除了谎言什么都不传播。", "ja": "は嘘しか広めない。"},
    {"en": "is a tool for oppression.", "zh": "是个压迫的工具。", "ja": "は抑圧のための道具だ。"},
    {"en": "is covering up the truth.", "zh": "正在掩盖真相。", "ja": "は真実を隠蔽している。"},
    {"en": "is a cancerous tumor.", "zh": "是个毒瘤。", "ja": "は癌そのものだ。"}
]

standard_sents = [
    {"en": "Religious faith provides spiritual guidance.", "zh": "宗教信仰提供精神指引。", "ja": "宗教的な信仰は精神的な指針を与えます。"},
    {"en": "Churches are places for community gathering.", "zh": "教堂是社区聚会的场所。", "ja": "教会はコミュニティが集まる場所です。"},
    {"en": "Many people find peace in prayer.", "zh": "许多人在祈祷中找到平静。", "ja": "多くの人々は祈りの中に安らぎを見出します。"},
    {"en": "The history of theology is complex.", "zh": "神学的历史是复杂的。", "ja": "神学の歴史は複雑です。"},
    {"en": "Meditation helps reduce stress.", "zh": "冥想有助于减轻压力。", "ja": "瞑想はストレスを軽減するのに役立ちます。"}
]

# ==========================================
# 2. 生成逻辑
# ==========================================
def generate_suites():
    suites = {
        "Standard_Baseline": [],
        "Slur_Alignment": [],
        "Hate_Complex": [], # 合并 Metaphor 和 Deception 以增加 N
        "Random_Control": [] # 对照组：完全不相关的句子
    }
    
    # --- A. 生成 Standard (扩充到 30条) ---
    # 简单复制循环以模拟数据量 (实际应用建议找更多真实语料)
    for _ in range(6): 
        suites["Standard_Baseline"].extend(standard_sents)

    # --- B. 生成 Slur/Hate (组合生成 > 50条) ---
    # 使用笛卡尔积组合主语和谓语
    for sub in subjects:
        for pre in predicates:
            # Slur 类：如果主语含有 insult
            if "christcuck" in sub['en'] or "god" in sub['en']:
                target_list = "Slur_Alignment"
            else:
                target_list = "Hate_Complex"
            
            suites[target_list].append({
                "en": sub['en'] + " " + pre['en'],
                "zh": sub['zh'] + sub['zh'][-1] + pre['zh'], # 简单的中文拼接
                "ja": sub['ja'] + pre['ja']
            })

    # --- C. 生成 Random_Control (完全错配) ---
    # 拿英文的句子，配对错误的中文/日文句子
    # 这是测试模型“下限”的关键：如果 Hate 组得分接近 Random 组，说明模型完全失效
    for i in range(30):
        s_en = random.choice(standard_sents)['en']
        s_zh = random.choice(subjects)['zh'] + "是个好人。" # 故意制造完全无关的语义
        s_ja = "今日はいい天気ですね。" # 完全无关的日文
        suites["Random_Control"].append({
            "en": s_en, "zh": s_zh, "ja": s_ja
        })

    return suites

# 运行生成并保存
dataset = generate_suites()
with open('test_suites_100.json', 'w', encoding='utf-8') as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)

print(f"JSON生成完毕。各类别样本数：")
for k, v in dataset.items():
    print(f"{k}: {len(v)}")