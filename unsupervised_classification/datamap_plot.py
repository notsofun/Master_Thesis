import pandas as pd
import numpy as np
import os, sys
import umap
from bertopic import BERTopic
import matplotlib.pyplot as plt
from matplotlib import font_manager
import matplotlib

# --- 1. 设置项目根目录路径 ---
# 获取当前脚本所在目录的父目录（即项目根目录）
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# 定义字体文件的绝对路径
font_path = os.path.join(project_root, "SourceHanSansSC-Regular.otf")

def fast_plot_only(model_path, csv_path, embeddings_path, output_path):
    # ==================== 新增：配置中文字体 ====================
    try:
        # 1. 将自定义字体注册到 matplotlib 的 fontManager
        font_manager.fontManager.addfont(font_path)
        # 2. 获取该字体的内部名称（例如 'Source Han Sans SC'）
        custom_font_name = font_manager.FontProperties(fname=font_path).get_name()
        # 3. 设置 matplotlib 全局默认字体
        plt.rcParams['font.family'] = custom_font_name
        # 4. 解决坐标轴负号 '-' 显示为方块的问题
        plt.rcParams['axes.unicode_minus'] = False
        print(f"✅ 成功加载中文字体: {custom_font_name}")
    except Exception as e:
        print(f"❌ 字体加载失败，请检查字体文件是否存在于: {font_path}")
        print(f"错误信息: {e}")
    # ============================================================

    # 1. 加载数据与对齐向量
    df = pd.read_csv(csv_path)
    all_embeddings = np.load(embeddings_path)
    df['temp_idx'] = range(len(df))
    df['topic'] = pd.to_numeric(df['topic'], errors='coerce')
    clean_df = df.dropna(subset=['topic', 'text']).copy()
    
    valid_indices = clean_df['temp_idx'].values
    embeddings = all_embeddings[valid_indices]
    texts = clean_df['text'].tolist()
    topics = clean_df['topic'].astype(int).tolist()

# 2. 加载已经命名好的模型
    topic_model = BERTopic.load(model_path, embedding_model=None)

    # ==================== 新增：应用英文标签 ====================
    english_topic_labels = {
    -1: "Inclusion and Reform Challenges in the Church",
    0: "Separation of Church & State and Corruption in Japan",
    1: "Christians and Trump's Politics",
    2: "Women and Ordination",
    3: "LGBTQ Issues and Youth Alienation",
    4: "Religion, Sexuality, and Violence",
    5: "Catholic-Lutheran Communion Disputes",
    6: "Church Abuse Victims and Support Mechanisms",
    7: "Religion and Natural Disasters",
    8: "Pro-Life and Women's Autonomy",
    9: "Divine-Human Nature and Teachings of Jesus",
    10: "Conflicts and Reflections on the Bible",
    11: "2017 Pakistan Religious Violence & Media Bias",
    12: "Communion for the Divorced and Remarried",
    13: "Religion in Public vs. Catholic Schools",
    14: "Public Prayer and Religious Freedom",
    15: "Authenticity and Faith in Catholicism",
    16: "Canadian Immigration and Cultural Disputes",
    17: "Pagan-Christian Conflicts in Online Debates",
    18: "Internal Discussions in Catholicism",
    19: "Doctrines of Religion and Salvation",
    20: "Nuns and Religious Culture",
    21: "Adventist Church: Status and Challenges",
    22: "Social Observations and Religious Critique",
    23: "Historical Critiques of Christian Violence",
    24: "Reforms and Divisions in the Catholic Church",
    25: "Social Division and Racial Antagonism",
    26: "Priests and Complexities of Faith",
    27: "Bakery and Religious Freedom Legal Conflicts",
    28: "Burke vs. Pope Francis & Future of the Church",
    29: "Christian Principles and Criticisms",
    30: "Christian Development and Denominational Differences",
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
    43: "Missionaries and Encounter with Confucianism",
    44: "Pope Francis: Internal Conflicts and Mercy",
    45: "Life and Responsibility",
    46: "Religious Beliefs and Social Controversies"
}
    # 强制将模型的自定义标签更新为你提供的英文标签
    topic_model.set_topic_labels(english_topic_labels)
    # ============================================================

    # 3. 2D 降维 (绘图必备)
    print("正在进行 2D 降维...")
    reducer = umap.UMAP(n_components=2, random_state=42)
    embeddings_2d = reducer.fit_transform(embeddings)

    # 4. 绘图
    print("正在生成 DataMap...")
    viz_topics = [t for t in sorted(list(set(topics))) if t >= 0][:30]
    
    try:
        # custom_labels=True 现在会读取你刚刚 set 进去的英文标签
        fig = topic_model.visualize_document_datamap(
            texts,
            reduced_embeddings=embeddings_2d,
            topics=viz_topics,
            custom_labels=True, 
        )
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"恭喜！英文版图片已成功保存至: {output_path}")
        
    except Exception as e:
        print(f"绘图失败了: {e}")

if __name__ == "__main__":
    BASE_DIR = r"unsupervised_classification\topic_modeling_results\sixth"
    fast_plot_only(
        model_path=os.path.join(BASE_DIR, "models", "bertopic_model"),
        csv_path=os.path.join(BASE_DIR, "data", "document_topic_mapping.csv"),
        embeddings_path=os.path.join(BASE_DIR, "models", "embeddings.npy"),
        output_path=os.path.join(BASE_DIR, "visualizations", "datamap_final_final.png")
    )