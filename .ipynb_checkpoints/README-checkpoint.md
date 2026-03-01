配置项目相同环境的方法：

# 1. 创建全新的虚拟环境
python3 -m venv venv

# 2. 激活环境
source venv/bin/activate

# 3. 根据清单安装依赖
pip install -r requirements.txt

# 模型文件已写入gitignore
需要通过`model/train/classifier/upload_hf.py`上传到hf来保存