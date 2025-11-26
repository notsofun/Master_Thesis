import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dataclasses import dataclass
import logging, os
from tqdm import tqdm

from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "log")

# ------------ 日志配置 ------------
log_filename = datetime.now().strftime("%Y%m%d_%H%M%S") + "detect.log"
log_path = os.path.join(LOG_DIR, log_filename)

logger = logging.getLogger("5ch_crawler")
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

logger.info("="*60)
logger.info("日志系统初始化完成")
logger.info("日志路径: %s", log_path)
logger.info("="*60)

@dataclass
class HateScore:
    non_attack: float
    gray_zone: float
    attack: float


class HateSpeechDetector:
    def __init__(self, model_name="TomokiFujihara/luke-japanese-base-lite-offensiveness-estimation"):
        base_model = "studio-ousia/luke-japanese-base-lite"

        logger.info("[INFO] Loading tokenizer and model...")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, trust_remote_code=True
        )
        logger.info("[INFO] Model loaded.")

    def score_text(self, text: str) -> HateScore:
        """Return normalized scores for 3 classes."""
        inputs = self.tokenizer.encode_plus(text, return_tensors="pt")
        logits = self.model(
            inputs["input_ids"],
            inputs["attention_mask"]
        ).detach().numpy()[0][:3]

        # normalization (as your example)
        minimum = np.min(logits)
        if minimum < 0:
            logits = logits - minimum

        score = logits / np.sum(logits)

        return HateScore(
            non_attack=float(score[0]),
            gray_zone=float(score[1]),
            attack=float(score[2])
        )

    @staticmethod
    def is_hate(score: HateScore) -> int:
        """Determine if the text is hate speech."""
        max_score = max(score.non_attack, score.gray_zone, score.attack)
        return 1 if max_score == score.attack else 0

    def run_batch(self, texts):
        """Run detection on a list/Series of texts."""
        results = []
        errors = []

        for idx, text in tqdm(
                enumerate(texts),
                total=len(texts),
                desc="Scoring",
                mininterval=0.5,
                ncols=100
            ):

            try:
                score = self.score_text(text)
                hs = self.is_hate(score)

                results.append({
                    "text": text,
                    "HS": hs
                })

            except Exception as e:
                logger.error(f"Error: {e} when analyzing {idx}th text.")
                errors.append({
                    "text": text,
                    "error": str(e)
                })

        # 转成 DataFrame 以便后处理
        return pd.DataFrame(results), pd.DataFrame(errors)

if __name__ == "__main__":
    df = pd.read_csv('../5ch/20251124_130843_5ch_posts.csv')
    texts = df["text"]

    detector = HateSpeechDetector()

    # 运行批量检测，得到成功与失败两份 DF
    result_df, error_df = detector.run_batch(texts)

    # 保存结果
    result_df.to_csv("result.csv", index=False)
    logger.info("[INFO] Saved to result.csv")

    # 保存错误日志
    if len(error_df) > 0:
        error_df.to_csv("errors.csv", index=False)
        logger.warning("[WARNING] Saved error records to errors.csv")

    # 统计
    total = len(texts)
    success = len(result_df)
    failed = len(error_df)

    logger.info(f"[INFO] Total texts: {total}")
    logger.info(f"[INFO] Successfully processed: {success} ({success/total:.2%})")
    logger.info(f"[INFO] Failed: {failed} ({failed/total:.2%})")

    # hate speech ratio
    if success > 0:
        hs_ratio = result_df["HS"].mean() * 100
        logger.info(f"[INFO] Hate speech ratio: {hs_ratio:.2f}%")
