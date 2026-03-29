import pandas as pd
import spacy
import plotly.graph_objects as go
import re
from collections import defaultdict
from tqdm import tqdm

# --- 1. 初始化模型 ---
print("正在加载多语言模型...")
nlp_en = spacy.load("en_core_web_sm")
nlp_zh = spacy.load("zh_core_web_sm")
nlp_ja = spacy.load("ja_core_news_sm")

# --- 2. 核心词蒸馏映射 (关键优化) ---
# 无论 GLiNER 抽出的句子多长，只要包含这些词，就归一化为核心靶子
ANCHOR_KEYWORDS = {
    '統一教会': 'Unification Church', '教会': 'Church (General)', 
    '安倍': 'Abe (Politician)', '自民党': 'LDP (Japan)',
    '耶稣': 'Jesus', 'jesus': 'Jesus', 'christ': 'Christianity',
    '圣母': 'Holy Mother/Mary', '神父': 'Priests/Clergy',
    'catholic': 'Catholic Church', 'church': 'Church (General)',
    'trump': 'Trump', 'obama': 'Obama', '孔子': 'Confucius'
}

# 停止词
STOP_WORDS = {'i', 'you', 'he', 'she', 'they', 'we', 'it', 'me', 'my', 'your', 'this', 'that'}
COPULA_VERBS = {'be', 'is', 'are', 'was', 'were', '是', '就是', 'だ', 'です', 'である'}

def distill_targets(raw_target_list):
    """
    将 GLiNER 抽出的长难句实体‘蒸馏’为干净的核心词。
    例如：'安倍は統一教会とズブズブ' -> ['Abe (Politician)', 'Unification Church']
    """
    distilled = []
    for raw in raw_target_list:
        found = False
        for key, norm in ANCHOR_KEYWORDS.items():
            if key in raw:
                distilled.append(norm)
                found = True
        # 如果长句里没包含预设核心词，但它本身比较短，也留着
        if not found and len(raw) < 10 and raw.lower() not in STOP_WORDS:
            distilled.append(raw)
    return list(set(distilled))

def parse_rq1_targets_refined(rq1_csv_path):
    """解析 RQ1，并应用蒸馏逻辑"""
    rq1_df = pd.read_csv(rq1_csv_path)
    topic_targets_map = {}
    for _, row in rq1_df.iterrows():
        # 提取 '实体(频次)' 里的实体部分
        raw_targets = re.findall(r"\'(.*?)\(\d+\)\'", row['Top_Targets'])
        # 蒸馏
        clean_list = distill_targets(raw_targets)
        topic_targets_map[int(row['Topic_ID'])] = clean_list
    return topic_targets_map

def extract_tactic_lenient(text, lang, targets_to_search):
    """利用蒸馏后的靶子进行宽容提取"""
    if lang == 'jp': nlp = nlp_ja
    elif lang == 'zh': nlp = nlp_zh
    else: nlp = nlp_en

    # 日文字节截断处理
    if lang == 'jp' and len(text.encode('utf-8')) > 48000:
        text = text.encode('utf-8')[:48000].decode('utf-8', errors='ignore')

    try: doc = nlp(text)
    except: return []

    found_tactics = []
    for token in doc:
        token_text = token.text.lower()
        
        # 匹配靶子：检查当前 token 是否命中任何蒸馏后的靶子核心词
        matched_target = None
        for t in targets_to_search:
            # 这里用双向包含来增加宽容度
            if t.lower() in token_text or token_text in t.lower():
                matched_target = t
                break
        
        if matched_target:
            head = token.head
            # 抓取逻辑：主语 -> 动作/标签，或者宾语 -> 被针对的动作
            if token.dep_ in ('nsubj', 'nsubjpass', 'top', 'compound', 'dep'):
                if head.lemma_ in COPULA_VERBS or head.text in COPULA_VERBS:
                    for child in head.children:
                        if child.dep_ in ('attr', 'acomp', 'dobj') and child.text != token.text:
                            found_tactics.append((matched_target, f"Is: {child.text}"))
                elif head.pos_ == 'VERB' and head.text not in STOP_WORDS:
                    verb = head.lemma_ if lang != 'zh' else head.text
                    found_tactics.append((matched_target, f"Do: {verb}"))
            elif token.dep_ in ('obj', 'dobj', 'pobj') and head.pos_ == 'VERB':
                verb = head.lemma_ if lang != 'zh' else head.text
                found_tactics.append((matched_target, f"Act_on: {verb}"))
                
    return found_tactics

def main():
    doc_path = r'unsupervised_classification\topic_modeling_results\sixth\data\document_topic_mapping.csv'
    rq1_path = r'unsupervised_classification\RQ1\data\rq1_topic_targets_summary.csv'
    
    doc_df = pd.read_csv(doc_path)
    # 使用优化后的解析器
    target_map = parse_rq1_targets_refined(rq1_path)
    links = defaultdict(int)
    
    # 选取前 15 个 Topic 展示
    selected_topics = sorted(target_map.keys())[:20]
    
    for _, row in tqdm(doc_df.iterrows(), total=len(doc_df)):
        tid = int(row['topic'])
        if tid not in selected_topics: continue
        
        lang = row.get('lang', 'en')
        text = str(row['text'])
        targets = target_map[tid]
        
        results = extract_tactic_lenient(text, lang, targets)
        for target_node, tactic_node in results:
            links[(f"Topic {tid}", target_node)] += 1
            links[(target_node, tactic_node)] += 1

    # --- 保存 links 到 CSV 进行后续分析 ---
    links_data = []
    for (src, tgt), w in links.items():
        # 判断关系类型，方便后期在 Excel 里过滤
        link_type = "Topic-to-Target" if src.startswith("Topic") else "Target-to-Tactic"
        links_data.append({
            "Source": src,
            "Target": tgt,
            "Weight": w,
            "Type": link_type
        })
    links_data = [{"Source": k[0], "Target": k[1], "Weight": v} for k, v in links.items()]
    links_df = pd.DataFrame(links_data).sort_values(by="Weight", ascending=False)
    links_df.to_csv(r"unsupervised_classification\RQ2\data\rq2_links_distilled.csv", index=False, encoding='utf-8-sig')
    

    print(f"✅ 原始连线数据已保存至: unsupervised_classification\RQ2\data\rq2_links_distilled.csv")

    # 绘图逻辑 (过滤 weight >= 2)
    filtered_links = {k: v for k, v in links.items() if v >= 2}
    
    nodes = sorted(list(set([k[0] for k in filtered_links.keys()] + [k[1] for k in filtered_links.keys()])))
    node_idx = {node: i for i, node in enumerate(nodes)}
    
    node_colors = []
    for node in nodes:
        if node.startswith("Topic"): node_colors.append("rgba(31, 119, 180, 0.7)")
        elif node.startswith("Do:"): node_colors.append("rgba(44, 160, 44, 0.7)")
        elif node.startswith("Act_on:"): node_colors.append("rgba(148, 103, 189, 0.7)") # 紫色：针对性动作
        elif node.startswith("Is:"): node_colors.append("rgba(214, 39, 40, 0.7)")
        else: node_colors.append("rgba(255, 127, 14, 0.7)") 

    sources = [node_idx[k[0]] for k in filtered_links.keys()]
    targets = [node_idx[k[1]] for k in filtered_links.keys()]
    values = [v for v in filtered_links.values()]

    fig = go.Figure(data=[go.Sankey(
        node=dict(pad=15, thickness=20, label=nodes, color=node_colors),
        link=dict(source=sources, target=targets, value=values, color="rgba(200, 200, 200, 0.4)")
    )])

    fig.update_layout(title_text="RQ2: Lenient Multilingual Analysis (Topic -> Target -> Tactic)", font_size=10, height=900)
    save_path = r"unsupervised_classification\RQ2\data\rq2_sankey_lenient.html"
    fig.write_html(save_path)
    print(f"\n✅ 宽容版桑基图已生成：{save_path}")
    print("✅ 蒸馏版分析完成，CSV 已保存。")

if __name__ == "__main__":
    main()