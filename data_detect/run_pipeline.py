# scripts/run_pipeline.py
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__)) # data_detect 目录
root_dir = os.path.dirname(current_dir)                 # Master_Thesis 目录
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from data_detect.Japanese.constants import ModelName

import logging
from datetime import datetime
from data_detect.Japanese.config import DEFAULT_INPUT_CSV
from data_detect.pipeline import HatePipeline
from dotenv import load_dotenv
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "log")

# ------------ 日志配置 ------------
log_filename = datetime.now().strftime("%Y%m%d_%H%M%S") + "pipeline.log"
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

logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("google_genai").setLevel(logging.ERROR)
logging.getLogger("google_genai.models").setLevel(logging.ERROR)

logger.info("="*60)
logger.info("日志系统初始化完成")
logger.info("日志路径: %s", log_path)
logger.info("="*60)

if __name__ == "__main__":
    to_test = [ModelName.KIT]
    # 2) 入口数据 CSV，必须包含列 'text'
    input_csv = DEFAULT_INPUT_CSV
    # 使用 ensemble pipeline（不再直接依赖 Gemini）
    pipeline = HatePipeline(logger, input_csv=input_csv, models=to_test)
    result = pipeline.run_detection(total_annotation_n=4000)
    logger.info("Pipeline finished. Result summary:")
    logger.info(result)
