import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re

# --- 准备工作：定义你的 Topic 语种映射（根据你之前的报告手工填入） ---
# 比如 Topic 0 主要是 EN，Topic 4 主要是 JP，Topic 2 主要是 ZH
topic_to_lang = {
    0: 'EN', 1: 'EN', 3: 'EN', 20: 'EN', 21: 'EN', 26: 'EN', 30: 'EN', 35: 'EN', 39: 'EN', # 英语系
    4: 'JP', 8: 'JP', 12: 'JP', 16: 'JP', 18: 'JP', 31: 'JP', 37: 'JP', 45: 'JP', 49: 'JP', 52: 'JP', # 日语系
    2: 'ZH', 7: 'ZH', 11: 'ZH', 14: 'ZH', 23: 'ZH', 25: 'ZH', 27: 'ZH', 28: 'ZH', 33: 'ZH', 38: 'ZH', 50: 'ZH' # 中文系
}

# --- 准备工作：定义社会学类别映射 ---
category_map = {
    'Individual': ['jesus', 'pope', 'francis', 'trump', 'obama', '安倍', '孔子', '山上', '汪海林', '孔庆东'],
    'Institution': ['church', 'vatican', 'bishops', '統一教会', '自民党', '創価学会', '日本会議', 'monastery'],
    'Social Group': ['catholics', 'christians', 'priests', 'pastor', 'baptist', 'mormons', '二世信徒', '新教', '穆斯林'],
    'Pejorative Label': ['圣母婊', '神棍', '基督狗', 'cult', 'カルト', 'スパイ', '詐欺集团', 'sb', '畜生', '迷信']
}

def get_category(word):
    word = word.lower()
    for cat, keywords in category_map.items():
        if any(k.lower() in word for k in keywords):
            return cat
    return None

# --- 主处理函数 ---
def generate_heatmap_from_summary(csv_path):
    df = pd.read_csv(csv_path)
    data_list = []

    for _, row in df.iterrows():
        t_id = int(row['Topic_ID'])
        lang = topic_to_lang.get(t_id, 'Other')
        if lang == 'Other': continue # 忽略未归类的话题
        
        # 解析字符串列表 "['church(74)', 'jesus(23)']"
        targets_str = row['Top_Targets']
        # 正则提取: 单词 和 括号里的数字
        matches = re.findall(r"\'(.*?)\((\d+)\)\'", targets_str)
        
        for word, count in matches:
            cat = get_category(word)
            if cat:
                data_list.append({
                    'Language': lang,
                    'Category': cat,
                    'Count': int(count)
                })

    # 转为 DataFrame 并聚合
    plot_df = pd.DataFrame(data_list)
    heatmap_data = plot_df.groupby(['Language', 'Category'])['Count'].sum().unstack(fill_value=0)
    
    # 转为比例（按行归一化）
    heatmap_data_pct = heatmap_data.div(heatmap_data.sum(axis=1), axis=0)
    
    # 绘图
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data_pct, annot=True, cmap='YlGnBu', fmt='.2%')
    plt.title('RQ1: Hate Target Categories Distribution (by Language)', fontsize=14)
    plt.show()

generate_heatmap_from_summary(r'unsupervised_classification\RQ1\data\rq1_topic_targets_summary.csv')