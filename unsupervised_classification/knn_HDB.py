# python unsupervised_classification/knn_HDB.py
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import umap
import hdbscan
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
import seaborn as sns
import os, sys, re
import logging
from tqdm import tqdm  # 引入进度条
import jieba
from janome.tokenizer import Tokenizer as JanomeTokenizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

# 初始化分词器
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

jt = JanomeTokenizer()

# Get the directory where the script is located (unsupervised_classification)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory (Master_Thesis)
project_root = os.path.dirname(current_dir)

# Add the project root to sys.path
if project_root not in sys.path:
    sys.path.append(project_root)

from scripts.set_logger import setup_logging

import warnings

# --- 1. 屏蔽 Python 级别的 UserWarning ---
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*TBB threading layer.*")

# --- 2. 针对第三方库，强制锁定日志级别，拒收 DEBUG 和 WARNING ---
# 这一步要在 setup_logging 之前或之后紧接着做
log_levels = {
    'matplotlib': logging.ERROR,
    'numba': logging.ERROR,
    'hdbscan': logging.ERROR,
    'umap': logging.ERROR,
    'transformers': logging.ERROR,
    'torch': logging.ERROR,
    'urllib3': logging.ERROR,
    'nltk': logging.ERROR
}

for lib, level in log_levels.items():
    logging.getLogger(lib).setLevel(level)

# --- 3. 初始化你的日志 ---
# 确保你的 setup_logging 内部没有把全局级别设为 logging.DEBUG
logger, _ = setup_logging()

# 如果 setup_logging 之后还是有 DEBUG，可以强制全局 INFO
logging.getLogger().setLevel(logging.INFO)

logger.info("日志系统已精简：已过滤 UserWarning 与第三方库 DEBUG 信息。")

# --- 新增：手动加载支持中日文的字体 ---
font_path = '/root/autodl-tmp/Master_Thesis/SourceHanSansSC-Regular.otf'
if os.path.exists(font_path):
    my_font = font_manager.FontProperties(fname=font_path)
    # 设置全局字体（可选，但有时对某些组件无效）
    plt.rcParams['font.family'] = my_font.get_name()
    plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题
    logger.info(f"已加载字体: {font_path}")
else:
    logger.warning("未找到中文字体文件，图片可能显示乱码。")
    

# ---------------------------
# 1. 配置与模型初始化
# ---------------------------
model_name = 'intfloat/multilingual-e5-large-instruct'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"正在使用设备: {device}")

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    logger.info(f"模型 {model_name} 加载成功。")
except Exception as e:
    logger.error(f"模型加载失败: {e}")
    raise

CORPUS_PATHS = {
    'zh': 'data_detect/finetuned_detection/chinese_final_religious_hate.csv',
    'en': 'data_collection/English_Existing/merged_deduped.csv',
    'jp': 'data_detect/finetuned_detection/japanese_final_religious_hate.csv'
}


def get_embeddings(texts, instruction, batch_size=16):
    processed_texts = [f"Instruct: {instruction}\nQuery: {t}" for t in texts]
    all_embeddings = []
    
    # 使用 tqdm 显示推理进度
    logger.info("开始生成语义嵌入 (Embedding)...")
    for i in tqdm(range(0, len(processed_texts), batch_size), desc="Inferencing"):
        batch = processed_texts[i : i + batch_size]
        inputs = tokenizer(batch, max_length=512, padding=True, truncation=True, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            # 提取 Mean Pooling 并归一化
            embeddings = outputs.last_hidden_state.mean(dim=1)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu().numpy())
            
    return np.vstack(all_embeddings)

def basic_tokenizer(text, lang):
    text = str(text).lower()
    if lang == 'zh':
        return " ".join(jieba.cut(text))
    if lang == 'jp':
        return " ".join([token.surface for token in jt.tokenize(text)])
    if lang == 'en':
        tokens = word_tokenize(text)
        return " ".join(tokens)
    return text


# 确保下载了 NLTK 停用词库
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

import requests

def load_authoritative_stopwords():
    """整合中、日、英权威停用词库"""
    all_stops = set()

    # --- 1. 英文：直接用 NLTK (权威标准) ---
    try:
        from nltk.corpus import stopwords
        all_stops.update(stopwords.words('english'))
    except:
        pass

    # --- 2. 中文：加载哈工大 (HIT) 停用词表 (权威) ---
    # 这里直接提供一个常用的中文基础停用词集合，涵盖百度/HIT的核心部分
    zh_hit_core = {
        '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这', '那', '们', '之', '与', '及', '等', '被', '从', '而', '但', '由于', '因此', '如果', '可以', '这样', '那个', '这里', '那里', '什么', '他们', '不是', '一个', "、","。","〈","〉","《","》","一","一个","一些","一何","一切","一则","一方面","一旦","一来","一样","一种","一般","一转眼","七","万一","三","上","上下","下","不","不仅","不但","不光","不单","不只","不外乎","不如","不妨","不尽","不尽然","不得","不怕","不惟","不成","不拘","不料","不是","不比","不然","不特","不独","不管","不至于","不若","不论","不过","不问","与","与其","与其说","与否","与此同时","且","且不说","且说","两者","个","个别","中","临","为","为了","为什么","为何","为止","为此","为着","乃","乃至","乃至于","么","之","之一","之所以","之类","乌乎","乎","乘","九","也","也好","也罢","了","二","二来","于","于是","于是乎","云云","云尔","五","些","亦","人","人们","人家","什","什么","什么样","今","介于","仍","仍旧","从","从此","从而","他","他人","他们","他们们","以","以上","以为","以便","以免","以及","以故","以期","以来","以至","以至于","以致","们","任","任何","任凭","会","似的","但","但凡","但是","何","何以","何况","何处","何时","余外","作为","你","你们","使","使得","例如","依","依据","依照","便于","俺","俺们","倘","倘使","倘或","倘然","倘若","借","借傥然","假使","假如","假若","做","像","儿","先不先","光","光是","全体","全部","八","六","兮","共","关于","关于具体地说","其","其一","其中","其二","其他","其余","其它","其次","具体地说","具体说来","兼之","内","再","再其次","再则","再有","再者","再者说","再说","冒","冲","况且","几","几时","凡","凡是","凭","凭借","出于","出来","分","分别","则","则甚","别","别人","别处","别是","别的","别管","别说","到","前后","前此","前者","加之","加以","区","即","即令","即使","即便","即如","即或","即若","却","去","又","又及","及","及其","及至","反之","反而","反过来","反过来说","受到","另","另一方面","另外","另悉","只","只当","只怕","只是","只有","只消","只要","只限","叫","叮咚","可","可以","可是","可见","各","各个","各位","各种","各自","同","同时","后","后者","向","向使","向着","吓","吗","否则","吧","吧哒","含","吱","呀","呃","呕","呗","呜","呜呼","呢","呵","呵呵","呸","呼哧","咋","和","咚","咦","咧","咱","咱们","咳","哇","哈","哈哈","哉","哎","哎呀","哎哟","哗","哟","哦","哩","哪","哪个","哪些","哪儿","哪天","哪年","哪怕","哪样","哪边","哪里","哼","哼唷","唉","唯有","啊","啐","啥","啦","啪达","啷当","喂","喏","喔唷","喽","嗡","嗡嗡","嗬","嗯","嗳","嘎","嘎登","嘘","嘛","嘻","嘿","嘿嘿","四","因","因为","因了","因此","因着","因而","固然","在","在下","在于","地","基于","处在","多","多么","多少","大","大家","她","她们","好","如","如上","如上所述","如下","如何","如其","如同","如是","如果","如此","如若","始而","孰料","孰知","宁","宁可","宁愿","宁肯","它","它们","对","对于","对待","对方","对比","将","小","尔","尔后","尔尔","尚且","就","就是","就是了","就是说","就算","就要","尽","尽管","尽管如此","岂但","己","已","已矣","巴","巴巴","年","并","并且","庶乎","庶几","开外","开始","归","归齐","当","当地","当然","当着","彼","彼时","彼此","往","待","很","得","得了","怎","怎么","怎么办","怎么样","怎奈","怎样","总之","总的来看","总的来说","总的说来","总而言之","恰恰相反","您","惟其","慢说","我","我们","或","或则","或是","或曰","或者","截至","所","所以","所在","所幸","所有","才","才能","打","打从","把","抑或","拿","按","按照","换句话说","换言之","据","据此","接着","故","故此","故而","旁人","无","无宁","无论","既","既往","既是","既然","日","时","时候","是","是以","是的","更","曾","替","替代","最","月","有","有些","有关","有及","有时","有的","望","朝","朝着","本","本人","本地","本着","本身","来","来着","来自","来说","极了","果然","果真","某","某个","某些","某某","根据","欤","正值","正如","正巧","正是","此","此地","此处","此外","此时","此次","此间","毋宁","每","每当","比","比及","比如","比方","没奈何","沿","沿着","漫说","点","焉","然则","然后","然而","照","照着","犹且","犹自","甚且","甚么","甚或","甚而","甚至","甚至于","用","用来","由","由于","由是","由此","由此可见","的","的确","的话","直到","相对而言","省得","看","眨眼","着","着呢","矣","矣乎","矣哉","离","秒","称","竟而","第","等","等到","等等","简言之","管","类如","紧接着","纵","纵令","纵使","纵然","经","经过","结果","给","继之","继后","继而","综上所述","罢了","者","而","而且","而况","而后","而外","而已","而是","而言","能","能否","腾","自","自个儿","自从","自各儿","自后","自家","自己","自打","自身","至","至于","至今","至若","致","般的","若","若夫","若是","若果","若非","莫不然","莫如","莫若","虽","虽则","虽然","虽说","被","要","要不","要不是","要不然","要么","要是","譬喻","譬如","让","许多","论","设使","设或","设若","诚如","诚然","该","说","说来","请","诸","诸位","诸如","谁","谁人","谁料","谁知","贼死","赖以","赶","起","起见","趁","趁着","越是","距","跟","较","较之","边","过","还","还是","还有","还要","这","这一来","这个","这么","这么些","这么样","这么点儿","这些","这会儿","这儿","这就是说","这时","这样","这次","这般","这边","这里","进而","连","连同","逐步","通过","遵循","遵照","那","那个","那么","那么些","那么样","那些","那会儿","那儿","那时","那样","那般","那边","那里","都","鄙人","鉴于","针对","阿","除","除了","除外","除开","除此之外","除非","随","随后","随时","随着","难道说","零","非","非但","非徒","非特","非独","靠","顺","顺着","首先","︿","！","＃","＄","％","＆","（","）","＊","＋","，","０","１","２","３","４","５","６","７","８","９","：","；","＜","＞","？","＠","［","］","｛","｜","｝","～","￥"
    }
    all_stops.update(zh_hit_core)

    # --- 3. 日语：加载 SlothLib 核心停用词 (权威) ---
    jp_sloth_core = {
        'あそこ', 'あたり', 'あっち', 'あて', 'あと', 'アドバイス', 'あまり', 'いい', 'いう', 'いく', 'いくら', 'いっ', 'いったい', 'いつも', 'いわゆる', 'うわ', 'えっ', 'える', 'おおよそ', 'おかげ', 'おまえ', 'おれ', 'から', 'が', 'き', 'ここ', 'こと', 'この', 'これ', 'これら', 'さま', 'さらに', 'し', 'しかし', 'する', 'ず', 'せ', 'ぜ', 'そ', 'そこ', 'そっち', 'そて', 'そに', 'その', 'その後', 'それ', 'それぞれ', 'それなり', 'た', 'たい', 'だ', 'たび', 'ため', 'だめ', 'だら', 'つ', 'て', 'で', 'でき', 'できる', 'です', 'では', 'でも', 'と', 'という', 'とき', 'どこ', 'どこか', 'ところ', 'どこまで', 'どの', 'どのよう', 'とも', 'と共に', 'どう', 'な', 'ない', 'ながら', 'なく', 'なっ', 'など', 'なに', 'なな', 'なにしろ', 'なにも', 'なる', 'なん', 'なんらか', 'に', 'における', 'において', 'について', 'によって', 'により', 'による', 'に対して', 'に対し', 'に関する', 'に関する', 'にて', 'れば', 'を', 'てる', "あそこ","あっ","あの","あのかた","あの人","あり","あります","ある","あれ","い","いう","います","いる","う","うち","え","お","および","おり","おります","か","かつて","から","が","き","ここ","こちら","こと","この","これ","これら","さ","さらに","し","しかし","する","ず","せ","せる","そこ","そして","その","その他","その後","それ","それぞれ","それで","た","ただし","たち","ため","たり","だ","だっ","だれ","つ","て","で","でき","できる","です","では","でも","と","という","といった","とき","ところ","として","とともに","とも","と共に","どこ","どの","な","ない","なお","なかっ","ながら","なく","なっ","など","なに","なら","なり","なる","なん","に","において","における","について","にて","によって","により","による","に対して","に対する","に関する","の","ので","のみ","は","ば","へ","ほか","ほとんど","ほど","ます","また","または","まで","も","もの","ものの","や","よう","より","ら","られ","られる","れ","れる","を","ん","何","及び","彼","彼女","我々","特に","私","私達","貴方","貴方方", 
    }
    all_stops.update(jp_sloth_core)

    # --- 4. 领域特有停用词 (根据你的日志反馈添加) ---
    # 这些词虽然是宗教词汇，但在你的语料中是“底色”，无助于区分“攻击类型”
    domain_stops = {
        # 中文
        '圣经', '耶稣', '上帝', '基督教', '基督', '修女', '神父', '教会', '教徒', '信仰', '宗教', '福音',
        # 日文
        'キリスト', '教会', '神', '聖書', 'クリスチャン', 'イエス', '教団', '宗教', 
        # 英文
        'christian', 'christians', 'church', 'jesus', 'bible', 'religion', 'god', 'pastor', 'faith'
    }
    all_stops.update(domain_stops)

    return list(all_stops)

# 最终在 TfidfVectorizer 中调用
custom_stopwords = load_authoritative_stopwords()

def get_keywords(df, cluster_col, n_words=6):
    keywords_dict = {}
    unique_clusters = [c for c in df[cluster_col].unique() if c != -1]
    

    for cluster in unique_clusters:
        cluster_texts = df[df[cluster_col] == cluster]['tokenized_text']
        
        # 显式使用我们的停用词表
        vectorizer = TfidfVectorizer(
            stop_words=custom_stopwords,
            max_features=1000,
            # token_pattern=r"(?u)\b\w+\b" # 如果想保留单字标签，取消此行注释
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(cluster_texts)
            words = vectorizer.get_feature_names_out()
            sums = tfidf_matrix.sum(axis=0).A1
            
            # 这里的排序会优先选出该簇独有，且不在停用词表里的词
            top_indices = sums.argsort()[-n_words:][::-1]
            keywords_dict[cluster] = "\n".join([words[i] for i in top_indices])
        except:
            keywords_dict[cluster] = "Processing Error"
            
    return keywords_dict
    
# ---------------------------
# 2. 加载与预处理数据
# ---------------------------
all_dfs = []
for lang, path in CORPUS_PATHS.items():
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            df = df[['text']].dropna().head(1000)
            df['lang'] = lang
            all_dfs.append(df)
            logger.info(f"成功加载 {lang} 语料: {len(df)} 条记录")
        except Exception as e:
            logger.warning(f"读取 {path} 时出错: {e}")
    else:
        logger.error(f"找不到路径: {path}")

if not all_dfs:
    logger.critical("未加载任何数据，程序退出。")
    exit()

df_total = pd.concat(all_dfs, ignore_index=True)
texts = df_total['text'].tolist()
logger.info(f"总计处理文本量: {len(texts)}")

# ---------------------------
# 3. 语义处理与聚类
# ---------------------------
# 预分词用于关键词提取
logger.info("执行多语言预分词...")
df_total['tokenized_text'] = df_total.apply(lambda row: basic_tokenizer(row['text'], row['lang']), axis=1)

instruction = "Identify and cluster hate speech patterns targeting Christianity, its clergy, and belief systems."
embeddings = get_embeddings(df_total['text'].tolist(), instruction)

logger.info("UMAP 降维...")
reducer = umap.UMAP(n_neighbors=15, n_components=2, metric='cosine', random_state=42)
embeddings_2d = reducer.fit_transform(embeddings)
df_total['x'], df_total['y'] = embeddings_2d[:, 0], embeddings_2d[:, 1]

# K-Means
logger.info("执行 K-Means...")
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df_total['cluster_kmeans'] = kmeans.fit_predict(embeddings)
keywords_kmeans = get_keywords(df_total, 'cluster_kmeans')

# HDBSCAN
logger.info("执行 HDBSCAN...")
hdb = hdbscan.HDBSCAN(min_cluster_size=10, metric='euclidean')
df_total['cluster_hdbscan'] = hdb.fit_predict(embeddings_2d)
keywords_hdbscan = get_keywords(df_total, 'cluster_hdbscan')

# ---------------------------
# 5. 可视化输出 (保留原版样式)
# ---------------------------
logger.info("正在生成对比图并标注关键词...")
plt.figure(figsize=(22, 7))
lang_palette = {'zh': '#e74c3c', 'en': '#3498db', 'jp': '#2ecc71'} # 红、蓝、绿

def plot_with_keywords(subplot_idx, title, cluster_col, keywords_dict):
    plt.subplot(1, 3, subplot_idx)
    sns.scatterplot(data=df_total, x='x', y='y', hue='lang', palette=lang_palette, s=15, alpha=0.4)
    
    for cluster, tags in keywords_dict.items():
        center_x = df_total[df_total[cluster_col] == cluster]['x'].mean()
        center_y = df_total[df_total[cluster_col] == cluster]['y'].mean()
        
        if not np.isnan(center_x):
            plt.text(
                center_x, center_y, tags,
                fontsize=8, 
                fontproperties=my_font,
                fontweight='bold', 
                color='black',
                ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.3')
            )
    plt.title(title, fontproperties=my_font) # 标题也加上

# 图1：原始语言分布（不标词）
plt.subplot(1, 3, 1)
sns.scatterplot(data=df_total, x='x', y='y', hue='lang', palette=lang_palette, s=15, alpha=0.6)
plt.title('Language Distribution (Ground Truth)')

# 图2：K-means 结果与词标签
plot_with_keywords(2, 'K-Means Clusters & Keywords', 'cluster_kmeans', keywords_kmeans)

# 图3：HDBSCAN 结果与词标签
plot_with_keywords(3, 'HDBSCAN Clusters & Keywords', 'cluster_hdbscan', keywords_hdbscan)

plt.tight_layout()
pic_path = 'unsupervised_classification/pics/clustering_keywords_comparison.png'
plt.savefig(pic_path, dpi=300)
logger.info(f"可视化图片已保存至: {pic_path}")

# 存储结果
df_total.to_csv('unsupervised_classification/result/clustering_results.csv', index=False, encoding='utf-8-sig')
logger.info("Process Completed.")
plt.show()