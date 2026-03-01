import logging, os
import traceback
from datetime import datetime


def setup_logging():
    # 1. 获取当前脚本 (train.py) 的绝对路径
    current_file_path = os.path.abspath(__file__)
    # 2. 获取脚本所在的目录 (即 model_train/classifier/)
    current_dir = os.path.dirname(current_file_path)
    
    # 3. 在脚本同级目录下定义 logs 文件夹
    log_dir = os.path.join(current_dir, "logs")
    
    # 如果不存在 logs 文件夹，则创建一个
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 生成日志文件名
    log_filename = datetime.now().strftime("%Y%m%d_%H%M%S_train.log")
    log_path = os.path.join(log_dir, log_filename)

    # 配置 Logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 清除现有的 handler (防止重复运行产生重复日志)
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # 输出到控制台
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 输出到文件
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger, log_path