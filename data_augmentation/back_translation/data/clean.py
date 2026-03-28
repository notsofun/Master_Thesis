import pandas as pd
import os
import logging

# ================= 配置区域 =================
CONFIG = {
    "input_file": "model_train/classifier/data/final_annotated_Japanese_train.csv",       # 原始导出的文件名
    "output_file": "data_augmentation/back_translation/data/Japanese.csv", # 筛选后准备增强的文件名
    "col_hate": "hate_speech",          # 仇恨言论列名
    "col_rel": "christianity_related",  # 宗教相关性列名
    "col_text": "text",                 # 文本列名
}

# 日志配置
log_path = os.path.join(os.path.dirname(__file__), "filter_process_chinese.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()]
)

def filter_dataset():
    if not os.path.exists(CONFIG["input_file"]):
        logging.error(f"未找到输入文件: {CONFIG['input_file']}")
        return

    # 1. 加载数据
    try:
        # 尝试不同编码，防止中文乱码
        df = pd.read_csv(CONFIG["input_file"], encoding='utf-8-sig')
    except:
        df = pd.read_csv(CONFIG["input_file"], encoding='gb18030')

    original_count = len(df)
    logging.info(f"原始数据总量: {original_count} 条")

    # 2. 执行筛选逻辑
    # 目标：hate_speech == '是' 或 christianity_related == '是'
    # 注意：这里使用了 str.strip() 消除可能存在的空格干扰
    condition = (
        (df[CONFIG["col_hate"]] == 1) & 
        (df[CONFIG["col_rel"]] == 1)
    )
    
    filtered_df = df[condition].copy()
    
    # 3. 数据质量检查：去重与空值处理
    filtered_df.dropna(subset=[CONFIG["col_text"]], inplace=True)
    filtered_df.drop_duplicates(subset=[CONFIG["col_text"]], inplace=True)
    
    final_count = len(filtered_df)

    # 4. 保存结果
    if final_count > 0:
        filtered_df.to_csv(CONFIG["output_file"], index=False, encoding='utf-8-sig')
        logging.info(f"筛选完成！")
        logging.info(f"符合条件 (是仇恨 或 宗教相关) 的文本: {final_count} 条")
        logging.info(f"过滤掉不符条件数据: {original_count - final_count} 条")
        logging.info(f"结果已保存至: {CONFIG['output_file']}")
    else:
        logging.warning("警告：筛选后没有符合条件的数据，请检查 CSV 中的‘是/否’字样是否匹配。")

if __name__ == "__main__":
    filter_dataset()