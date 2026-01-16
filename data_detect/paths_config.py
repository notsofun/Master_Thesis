"""
项目路径配置 - 确保在任何执行环境中都能正确引用路径
"""

import os
import sys

# 获取项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 确保项目根目录在 sys.path 中（用于本地开发）
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 数据检测模块路径
DATA_DETECT_DIR = os.path.join(PROJECT_ROOT, 'data_detect')
JAPANESE_DIR = os.path.join(DATA_DETECT_DIR, 'Japanese')
MODELS_DIR = os.path.join(JAPANESE_DIR, 'models')

# 数据收集模块路径
DATA_COLLECTION_DIR = os.path.join(PROJECT_ROOT, 'data_collection')

# 数据预分析模块路径
DATA_PREANALYSIS_DIR = os.path.join(PROJECT_ROOT, 'data_preanalysis')

# 日志目录
LOG_DIR = os.path.join(DATA_DETECT_DIR, 'log')
os.makedirs(LOG_DIR, exist_ok=True)

# 输出目录
OUTPUT_DIR = os.path.join(DATA_DETECT_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

__all__ = [
    'PROJECT_ROOT',
    'DATA_DETECT_DIR',
    'JAPANESE_DIR',
    'MODELS_DIR',
    'DATA_COLLECTION_DIR',
    'DATA_PREANALYSIS_DIR',
    'LOG_DIR',
    'OUTPUT_DIR',
]
