import logging, os
import traceback
from datetime import datetime
import inspect


def setup_logging(name:str = 'train'):
    # 1. 获取调用该函数的脚本路径 (例如 train.py)
    caller_frame = inspect.currentframe().f_back
    caller_file = caller_frame.f_globals['__file__']
    caller_dir = os.path.dirname(os.path.abspath(caller_file))
    
    # 2. 在调用脚本的同级目录下定义 logs 文件夹
    log_dir = os.path.join(caller_dir, "logs")
    
    # 如果不存在 logs 文件夹，则创建一个
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 生成日志文件名
    log_filename = datetime.now().strftime(f"%Y%m%d_%H%M%S_{name}.log")
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