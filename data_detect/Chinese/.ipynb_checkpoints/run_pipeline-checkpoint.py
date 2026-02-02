#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中文仇恨言论检测 Pipeline
使用集成学习方法对中文文本进行仇恨言论检测
"""

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))  # data_detect/Chinese 目录
root_dir = os.path.dirname(os.path.dirname(current_dir))   # Master_Thesis 目录
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import logging
from datetime import datetime
from data_detect.Chinese.config import DEFAULT_INPUT_CSV, OUTPUT_DIR, DEFAULT_POPULATION, DEFAULT_MARGIN, DEFAULT_MAX_SAMPLE
from data_detect.pipeline import HatePipeline
from data_detect.Chinese.constants import ChineseModelName
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "log")

# 创建日志目录
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------ 日志配置 ------------
log_filename = datetime.now().strftime("%Y%m%d_%H%M%S") + "_pipeline_chinese.log"
log_path = os.path.join(LOG_DIR, log_filename)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 文件输出
file_handler = logging.FileHandler(log_path, encoding="utf-8")
file_handler.setLevel(logging.INFO)

# 控制台输出
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter(
    "[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

# 降低一些库的日志级别
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.WARNING)

logger.info("="*60)
logger.info("中文仇恨言论检测 Pipeline 启动")
logger.info("日志路径: %s", log_path)
logger.info("="*60)

if __name__ == "__main__":
    # 指定要使用的模型
    models_to_test = [ChineseModelName.THUCOAI, ChineseModelName.MORIT, ChineseModelName.DAVIDCLIAO]
    # 注: DAVIDCLIAO 需要本地下载模型文件，暂不包含
    
    # 输入数据 CSV (需要包含 'text' 或 'main_content' 列)
    input_csv = '../../data_collection/Tieba/final_cleaned_data.csv'
    
    # 创建 Pipeline
    pipeline = HatePipeline(
        logger,
        input_csv=input_csv,
        models=models_to_test,
        population=DEFAULT_POPULATION,
        margin=DEFAULT_MARGIN,
        output_dir=OUTPUT_DIR,
        device="cuda"  # 根据需要改为 "cuda" 或 "cuda:0"
    )
    
    # 运行检测
    result = pipeline.run_detection(total_annotation_n=4000)
    
    logger.info("="*60)
    logger.info("Pipeline 完成。结果摘要:")
    logger.info("- 评估的文本数: %d", len(result['evaluated']))
    logger.info("- 注释集大小: %d", len(result['annotation_df']))
    logger.info("- 使用的模型数: %d", result['n_models'])
    logger.info("- 输出路径: %s", result['annotation_csv'])
    logger.info("="*60)
