# config.py
import os


# 默认输入、输出路径
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "output"))

# 如果你手上有 CSV 原始文件，可在运行脚本传入；这里给默认文件名
DEFAULT_INPUT_CSV = '../../data_collection/Tieba/all_search_posts.csv'



# 采样默认置信与z值
Z_95 = 1.96
DEFAULT_PROPORTION = 0.5  # p=0.5 为最保守
DEFAULT_MARGIN = 0.03     # 3% margin
DEFAULT_POPULATION = 20000
DEFAULT_MAX_SAMPLE = 5000  # 你想要的上限（可更改）
RANDOM_SEED = 42
