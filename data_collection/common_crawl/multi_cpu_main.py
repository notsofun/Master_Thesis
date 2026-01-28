import csv
import os
import time
from multiprocessing import Pool, cpu_count
from datasets import load_dataset

# --- 配置区 ---
TASKS = [
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
    # 你可以继续在这里添加其他语言任务
]

MAX_CHAR_LEN = 250
MAX_EXTRACT_PER_LANG = 400000 
NUM_PROC = cpu_count()  # 自动获取服务器核心数

def worker_task(args):
    """
    单个核心执行的扫描任务
    """
    lang, kws, shard_idx, num_shards, quota_per_shard = args
    
    # 尝试加载数据集
    try:
        dataset = load_dataset("allenai/c4", f"multilingual.{lang}", split="train", streaming=True)
    except:
        dataset = load_dataset("allenai/c4", lang, split="train", streaming=True)

    # 关键：对流式数据集进行分片
    dataset = dataset.shard(num_shards=num_shards, index=shard_idx)

    local_results = []
    count = 0
    
    for entry in dataset:
        text = entry['text'].replace("\n", " ").strip()
        
        if len(text) <= MAX_CHAR_LEN:
            if any(kw in text for kw in kws):
                local_results.append([text, len(text)])
                count += 1
                
        # 达到该核心的分摊配额就停止
        if count >= quota_per_shard:
            break
            
    return local_results

def run_multilingual_extraction():
    for task in TASKS:
        lang = task['lang']
        kws = task['keywords']
        fname = task['filename']
        
        print(f"\n{'='*40}")
        print(f"🚀 任务启动 | 语言: {lang.upper()} | 核心数: {NUM_PROC}")
        print(f"{'='*40}")

        start_time = time.time()
        
        # 计算每个核心应负担的任务量 (稍微多给 10% 缓冲，防止有的分片数据不够)
        quota_per_shard = (MAX_EXTRACT_PER_LANG // NUM_PROC) + (MAX_EXTRACT_PER_LANG // 100)
        
        # 准备进程池参数
        worker_args = [
            (lang, kws, i, NUM_PROC, quota_per_shard) 
            for i in range(NUM_PROC)
        ]

        # 启动多进程并行处理
        with Pool(processes=NUM_PROC) as pool:
            # imap_unordered 会在结果出来时就返回，稍微快一点点
            shards_results = pool.map(worker_task, worker_args)

        # 汇总所有进程的结果
        final_data = []
        for res in shards_results:
            final_data.extend(res)

        # 精准截取前 MAX_EXTRACT_PER_LANG 条
        final_data = final_data[:MAX_EXTRACT_PER_LANG]

        # 写入 CSV
        with open(fname, mode='w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['text', 'char_count'])
            writer.writerows(final_data)

        end_time = time.time()
        avg_speed = len(final_data) / (end_time - start_time)
        print(f"✅ [{lang.upper()}] 完成！抓取: {len(final_data)} 条")
        print(f"⏱️ 总耗时: {end_time - start_time:.1f}s | 平均速度: {avg_speed:.1f}条/秒")

if __name__ == "__main__":
    # Windows 环境下必须在 if __name__ == "__main__": 后运行进程池
    run_multilingual_extraction()