import csv
import time
import os
from datasets import load_dataset

# --- 配置区 ---
TASKS = [
    {
        "lang": "zh",
        "keywords": [
            "上帝", "耶稣", "基督", "基督徒", "教会", "圣经", "圣灵", "圣餐", "圣体",
            "十字架", "旧约", "新约", "福音", "教义", "做礼拜", "宗教", "教区", 
            "教派", "天主", "天主教", "东正教", "新教", "犹太教", "以色列", 
            "传教士", "神父", "牧师", "主教", "教皇", "修道士", "修女", "修道院",
            "耶狗", "基督狗", "神棍", "布道鬼", "布道佬", "皮条教", "圣母婊",
            "脑残信徒", "二毛子", "性罪教", "十字军党", "圣战", "十字军", 
            "东征", "审判", "教义争论", "改革", "宗教改革", "中世纪"
        ],
        "filename": "extract_zh_religious.csv"
    },
    {
        "lang": "ja",
        "keywords": [
            "キリスト教", "キリスト教徒", "キリスト", "イエス", "教会", "カトリック",
            "プロテスタント", "正教会", "ローマ", "ローマ教皇", "司教", "司祭",
            "神父", "枢機卿", "聖書", "旧約", "新約", "信仰", "教義", "福音",
            "殉教", "コルベ", "マザーテレサ", "キリシタン", "統一教会", "フランシスコ"
        ],
        "filename": "extract_ja_religious.csv"
    }
]

MAX_CHAR_LEN = 250
MAX_EXTRACT_PER_LANG = 400000  # 调大一点，100条太快了
CHECKPOINT = 10000

def run_multilingual_extraction():
    for task in TASKS:
        lang = task['lang']
        kws = task['keywords']
        fname = task['filename']
        
        # 检查文件是否存在，决定是否写入表头
        file_exists = os.path.isfile(fname)
        
        print(f"\n{'='*40}")
        print(f"📡 启动任务 | 语言: {lang.upper()} | 模式: {'追加' if file_exists else '新建'}")
        print(f"{'='*40}")

        try:
            dataset = load_dataset("allenai/c4", f"multilingual.{lang}", split="train", streaming=True)
        except:
            dataset = load_dataset("allenai/c4", lang, split="train", streaming=True)

        count = 0
        scanned = 0
        start_time = time.time()

        # 使用 'a' (append) 模式打开文件
        with open(fname, mode='a', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['text', 'char_count']) 

            try:
                for entry in dataset:
                    scanned += 1
                    text = entry['text'].replace("\n", " ").strip()
                    
                    if len(text) <= MAX_CHAR_LEN:
                        if any(kw in text for kw in kws):
                            writer.writerow([text, len(text)])
                            count += 1
                            
                            # --- 稳健性核心：强制写盘 ---
                            if count % 10 == 0: # 每10条强制刷新一次文件缓冲区
                                f.flush()

                            if count % CHECKPOINT == 0:
                                avg_speed = scanned / (max(1, time.time() - start_time))
                                print(f"✅ [{lang.upper()}] 已保存: {count} 条 | 速度: {avg_speed:.1f}条/秒")

                            if count >= MAX_EXTRACT_PER_LANG:
                                break
            except Exception as e:
                print(f"\n⚠️ 运行时中断 (可能是网络问题): {e}")
                print(f"数据已安全保存至 {fname}，你可以重新运行脚本继续加载。")
        
        print(f"🎉 {lang.upper()} 环节处理结束。")

if __name__ == "__main__":
    run_multilingual_extraction()