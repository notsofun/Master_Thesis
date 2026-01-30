import csv
import gzip
import json
import requests
import time
import os
import sys

# ================= 配置区 =================
LANG = "ja"
# 目标条数
TARGET_COUNT = 400000 
# 最大字符长度（防止抓到超长文章）
MAX_CHAR_LEN = 250
# 镜像站基准 URL
BASE_URL = "https://hf-mirror.com/datasets/allenai/c4/resolve/main/multilingual/c4-ja.tfrecord-{:05d}-of-01024.json.gz"
# 输出文件名
SAVE_PATH = "extract_ja_religious.csv"
# 状态记录文件（记录处理到第几个文件了）
STATE_FILE = "extraction_state.txt"

# 全量关键词列表
KEYWORDS = [
    "キリスト教", "キリスト教徒", "キリスト", "イエス", "教会", "カトリック",
    "プロテスタント", "正教会", "ローマ", "ローマ教皇", "司教", "司祭",
    "神父", "枢機卿", "聖書", "旧約", "新約", "信仰", "教義", "福音",
    "殉教", "コルベ", "マザーテレサ", "キリシタン", "統一教会", "フランシスコ"
]
# ==========================================

def get_progress():
    """获取进度：已保存条数和当前处理的文件索引"""
    saved_count = 0
    start_file_idx = 0
    
    if os.path.exists(SAVE_PATH):
        with open(SAVE_PATH, 'r', encoding='utf-8') as f:
            saved_count = sum(1 for _ in f) - 1 # 扣除表头
            
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            start_file_idx = int(f.read().strip())
            
    return max(0, saved_count), start_file_idx

def save_state(file_idx):
    """保存当前处理到的文件编号"""
    with open(STATE_FILE, 'w') as f:
        f.write(str(file_idx))

def stream_from_url(url):
    """流式下载并解压"""
    # 增加重试机制
    for attempt in range(3):
        try:
            response = requests.get(url, stream=True, timeout=60)
            if response.status_code == 200:
                with gzip.open(response.raw, 'rt', encoding='utf-8') as f:
                    for line in f:
                        yield json.loads(line)
                return # 成功处理完一个文件
            elif response.status_code == 404:
                print(f"❌ 文件不存在(404): {url}")
                return
        except Exception as e:
            print(f"⚠️ 连接失败 (尝试 {attempt+1}/3): {e}")
            time.sleep(5)
    print(f"🔥 无法处理文件: {url}")

def run():
    saved_count, start_file_idx = get_progress()
    
    print(f"{'='*40}")
    print(f"🚀 任务启动 | 目标: {TARGET_COUNT} 条")
    print(f"📈 当前进度: {saved_count} 条 | 起始分片: {start_file_idx:05d}")
    print(f"{'='*40}")

    # 检查是否已完成
    if saved_count >= TARGET_COUNT:
        print("✅ 任务检测到已完成，无需运行。")
        return

    # 打开文件（追加模式）
    file_exists = os.path.exists(SAVE_PATH)
    with open(SAVE_PATH, mode='a', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(['text', 'char_count'])

        # 遍历 C4 的 1024 个分片
        for file_idx in range(start_file_idx, 1024):
            url = BASE_URL.format(file_idx)
            print(f"\n📡 正在处理分片 [{file_idx:05d}/1023]")
            
            start_time = time.time()
            file_processed_docs = 0
            
            for entry in stream_from_url(url):
                text = entry.get('text', '').replace('\n', ' ').strip()
                file_processed_docs += 1
                
                # 过滤逻辑
                if 10 < len(text) <= MAX_CHAR_LEN:
                    if any(kw in text for kw in KEYWORDS):
                        writer.writerow([text, len(text)])
                        saved_count += 1
                        
                        # 每 10 条强制刷盘，防止断电丢失
                        if saved_count % 10 == 0:
                            csvfile.flush()
                        
                        # 打印实时进度
                        if saved_count % 100 == 0:
                            print(f"\r   进度: {saved_count}/{TARGET_COUNT} | 当前分片已扫描: {file_processed_docs}", end="")

                        if saved_count >= TARGET_COUNT:
                            print(f"\n\n🎯 恭喜！已达到目标数量 {TARGET_COUNT} 条。")
                            save_state(file_idx)
                            return

            # 完成一个分片后的处理
            elapsed = time.time() - start_time
            print(f"\n✅ 分片 {file_idx:05d} 处理完毕 | 耗时: {elapsed:.1f}s")
            save_state(file_idx + 1) # 成功后记录下一个文件索引
            
            # 适当休眠，避免镜像站封锁
            time.sleep(1.5)

if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print("\n\n👋 用户手动停止。输入 'python stable_extract_ja.py' 可随时恢复。")