配置项目相同环境的方法：

# 1. 创建全新的虚拟环境
python3 -m venv venv

# 2. 激活环境
source venv/bin/activate

# 3. 根据清单安装依赖
pip install -r requirements.txt

# 模型文件已写入gitignore
需要通过`model/train/classifier/upload_hf.py`上传到hf来保存

基于上传的图片和这几个文件

unsupervised_classification\topic_modeling_results\sixth\data\topic_info.csv

unsupervised_classification\topic_modeling_results\sixth\data\cluster_quality_detailed.csv

为我撰写一章关于berTopic基础分类的分析，在我的论文中属于第一个实验。撰写在这个文件内，保持简单易懂的英文，latex格式，需要插入图片的地方给我空出来，我理解上面四张图都需要加入。

thesis_draft\chapters\berTopic.tex



基本框架就是，先简单overview以下，说我使用了bertopic，然后设置了一些关键参数，比如umap的值等等，这些你可以空出来我来填。得到了以下结果，我将会挑出关键话题来论述。

第一小节，就是说我选择了三个单语言话题，就是每种语言占比最高的那个话题，然后选了三个跨语言的话题，你挑几个representative doc讲一下内容，结合标签讲讲这个分类到底是说明了什么。

第二小节，主要是结合hierarchy的那个图，讲讲各个分类之间是什么关系。

第三小节，结合similarity那个图，讲各个话题的相似度，其实也没有什么相似度:)

第四小节，基本就是小结一下，然后说接下来将开始进入who->how->why的分析