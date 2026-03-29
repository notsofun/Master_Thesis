import pandas as pd
import spacy
import plotly.graph_objects as go
from collections import defaultdict

print("正在加载多语言 NLP 模型，这可能需要几秒钟...")
nlp_en = spacy.load("en_core_web_sm")
nlp_zh = spacy.load("zh_core_web_sm")
nlp_ja = spacy.load("ja_core_news_sm")

# --- 1. 配置你的核心靶子与无效词过滤 ---
# 为了让桑基图清晰，我们选出最具代表性的几个靶子实体
CORE_TARGETS = ['church', 'catholic', '統一教会', '安倍', 'jesus', '圣母']

# 过滤掉没有社会学意义的“轻动词”和“系动词”
STOP_VERBS = {'be', 'is', 'are', 'was', 'were', 'have', 'has', 'do', 'does', 'say', 'said', 
              'go', 'make', 'think', 'know', 'see', 'する', 'いる', 'ある', 'なる', 'いう', 
              '是', '有', '说', '做', '去', '想', '看', '让'}

def get_action_verb(text, lang):
    """提取文本中与靶子相关的核心动词"""
    if lang == 'en':
        doc = nlp_en(text)
    elif lang == 'zh':
        doc = nlp_zh(text)
    elif lang == 'jp':
        doc = nlp_ja(text)
    else:
        return []

    verbs = []
    text_lower = text.lower()
    
    # 如果句子包含我们的核心靶子
    for target in CORE_TARGETS:
        if target.lower() in text_lower:
            for token in doc:
                # 找到句子里的动词
                if token.pos_ == 'VERB':
                    verb_lemma = token.lemma_ if lang == 'en' else token.text
                    if verb_lemma not in STOP_VERBS and len(verb_lemma) > 1:
                        verbs.append((target, verb_lemma))
    return verbs

def process_and_plot_sankey(doc_mapping_csv):
    print("开始读取数据并提取依存句法关系...")
    df = pd.read_csv(doc_mapping_csv)
    
    # 过滤掉噪声 (-1)
    df = df[df['topic'] != -1]
    
    # 记录流向频次：Topic -> Target, Target -> Verb
    topic_target_weights = defaultdict(int)
    target_verb_weights = defaultdict(int)
    
    # 为了演示速度，这里假设我们只抽取带有核心词的行进行分析
    # 你可以加入 tqdm 来看进度
    for _, row in df.iterrows():
        topic_name = f"Topic {row['topic']}" # 例如 "Topic 0"
        lang = row.get('lang', 'en') # 确保你的csv里有lang列
        text = str(row['text'])
        
        extracted_pairs = get_action_verb(text, lang)
        
        for target, verb in extracted_pairs:
            topic_target_weights[(topic_name, target)] += 1
            target_verb_weights[(target, verb)] += 1

    print("数据提取完毕，准备生成桑基图...")
    
    # --- 2. 剪枝：防止桑基图变成“毛线球” ---
    # 只保留高频的动词 (比如频次 >= 3 的)
    # 并且只取 Top 15 的动词，保证图片美观
    sorted_verbs = sorted(target_verb_weights.items(), key=lambda x: x[1], reverse=True)
    top_target_verbs = dict(sorted_verbs[:20]) # 控制右侧节点的数量
    
    # 筛选出有效的目标实体 (存在于高频动词中的)
    valid_targets = set([k[0] for k in top_target_verbs.keys()])
    
    # 构建绘图所需的 Source, Target, Value 列表
    nodes = list(set([k[0] for k in topic_target_weights.keys()] + 
                     list(valid_targets) + 
                     [k[1] for k in top_target_verbs.keys()]))
    node_indices = {node: i for i, node in enumerate(nodes)}
    
    source = []
    target = []
    value = []
    
    # 添加 Topic -> Target 的连线
    for (t, tgt), weight in topic_target_weights.items():
        if tgt in valid_targets:
            source.append(node_indices[t])
            target.append(node_indices[tgt])
            value.append(weight)
            
    # 添加 Target -> Verb 的连线
    for (tgt, v), weight in top_target_verbs.items():
        source.append(node_indices[tgt])
        target.append(node_indices[v])
        value.append(weight)

    # --- 3. 使用 Plotly 绘制桑基图 ---
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=nodes,
            # 可以自定义颜色：Topic用蓝色，Target用红色，Verb用绿色等
            color="rgba(31, 119, 180, 0.8)" 
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color="rgba(169, 169, 169, 0.4)" # 半透明灰色连线
        )
    )])

    fig.update_layout(title_text="Cross-Topic Frame Contrast: Topic -> Target -> Tactic", font_size=12)
    fig.write_html("data/sankey_network_rq2.html")
    print("✅ 桑基图已生成！请在浏览器中打开 sankey_network_rq2.html 查看结果。")

# 运行代码
# 替换为你的真实路径
process_and_plot_sankey('unsupervised_classification/topic_modeling_results/sixth/data/document_topic_mapping.csv')