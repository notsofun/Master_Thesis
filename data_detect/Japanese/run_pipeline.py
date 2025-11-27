# scripts/run_pipeline.py
import os
import logging
from datetime import datetime
from config import DEFAULT_INPUT_CSV
from pipeline import HatePipeline
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
    # 1) 配置：把模型列表换成你要的 HS 模型
    model_specs = [
        {"name": "luke_offensiveness", "tokenizer_base": "studio-ousia/luke-japanese-base-lite", "model": "TomokiFujihara/luke-japanese-base-lite-offensiveness-estimation"},
        # 增加其他你想评估的模型
        # {"name": "my_hs_model_v1", "tokenizer_base": "bert-base-multilingual-cased", "model": "your-org/your-finetuned-model"},
    ]

    # 2) 入口数据 CSV，必须包含列 'text'
    input_csv = DEFAULT_INPUT_CSV

    # 3) 从环境变量读取 Gemini Key
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        raise RuntimeError("Please set GEMINI_API_KEY in environment before running.")

    pipeline = HatePipeline(logger, input_csv=input_csv, model_specs=model_specs, gemini_api_key=gemini_key)
    result = pipeline.run_detection()
    logger.info("Pipeline finished. Result summary:")
    logger.info(result)
