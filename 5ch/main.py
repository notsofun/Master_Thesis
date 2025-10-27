import os
import time
import json
import random
import pickle
import urllib.parse
import re
from typing import List, Dict, Any
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from webdriver_manager.chrome import ChromeDriverManager

# ========== 基本配置 ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COOKIE_FILE = os.path.join(BASE_DIR, "5ch_cookies.json")
PROGRESS_DIR = os.path.join(BASE_DIR, "progress")
os.makedirs(PROGRESS_DIR, exist_ok=True)

# 搜索配置
KEYWORDS = ["キリスト教"]  # 搜索关键词
OUTPUT_FILE = "5ch_full_posts.jsonl"  # 输出文件
DELAY_BETWEEN_REQUESTS = random.uniform(2, 4)  # 请求间隔（随机）
MAX_RETRIES = 3  # 最大重试次数
BACKOFF_BASE = 5  # 重试等待基础时间（秒）

# ========== 工具函数 ==========
def init_browser() -> webdriver.Chrome:
    """初始化Chrome浏览器"""
    print("[INFO] 初始化浏览器...")
    options = Options()
    options.add_argument("--start-maximized")
    # options.add_argument("--headless")  # 无头模式，取消注释以启用
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    
    # 如果有保存的cookies，加载它们
    if os.path.exists(COOKIE_FILE):
        print("[INFO] 加载已保存的cookies...")
        try:
            driver.get("https://itest.5ch.net")
            with open(COOKIE_FILE, "r", encoding="utf-8") as f:
                cookies = json.load(f)
            for cookie in cookies:
                driver.add_cookie(cookie)
            print("[INFO] Cookies加载完成")
        except Exception as e:
            print(f"[WARN] 加载cookies失败: {e}")
    
    return driver

def clean_text(s: str) -> str:
    """清理文本中的多余空白"""
    return re.sub(r"\s+", " ", s).strip()

def uid_from_text(*parts) -> str:
    """生成文本内容的唯一标识"""
    import hashlib
    return hashlib.sha1("||".join(map(str, parts)).encode("utf-8")).hexdigest()

def safe_filename(s: str) -> str:
    """生成安全的文件名"""
    return "".join(c if c.isalnum() else "_" for c in s)[:200]

def human_like_sleep(min_s=2.0, max_s=4.0):
    """模拟人工操作的随机等待"""
    time.sleep(random.uniform(min_s, max_s))

def save_progress(keyword: str, state: Dict[str, Any]):
    """保存断点续传状态"""
    path = os.path.join(PROGRESS_DIR, f"{safe_filename(keyword)}_progress.pkl")
    with open(path, "wb") as f:
        pickle.dump(state, f)

def load_progress(keyword: str):
    """加载断点续传状态"""
    path = os.path.join(PROGRESS_DIR, f"{safe_filename(keyword)}_progress.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

# ========== Selenium 配置 ==========
def init_driver():
    """初始化浏览器驱动"""
    options = Options()
    options.add_argument("--start-maximized")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--lang=ja-JP")
    # options.add_argument("--headless=new")  # 无头模式（取消注释启用）
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    wait = WebDriverWait(driver, 12)
    
    # 加载 cookie（如果存在）
    if os.path.exists(COOKIE_FILE):
        try:
            print("[INFO] 尝试加载已保存的 cookies...")
            driver.get("https://5ch.net/")
            with open(COOKIE_FILE, "r", encoding="utf-8") as f:
                cookies = json.load(f)
            for cookie in cookies:
                try:
                    driver.add_cookie(cookie)
                except Exception:
                    pass
            driver.refresh()
        except Exception as e:
            print(f"[WARN] 加载 cookies 失败: {e}")
    
    return driver, wait

driver, wait = init_driver()
wait = WebDriverWait(driver, 12)

# ========== 搜索页抓取 ==========
def parse_search_page(driver, keyword: str):
    """解析搜索结果页面"""
    search_url = f"https://itest.5ch.net/find?q={urllib.parse.quote(keyword)}"
    print(f"[INFO] 搜索关键词: {keyword}")
    
    retry_count = 0
    while retry_count < MAX_RETRIES:
        try:
            driver.get(search_url)
            human_like_sleep(2, 4)  # 等待页面加载
            
            # 保存当前进度
            if retry_count == 0:
                cookies = driver.get_cookies()
                with open(COOKIE_FILE, "w", encoding="utf-8") as f:
                    json.dump(cookies, f, ensure_ascii=False, indent=2)

            soup = BeautifulSoup(driver.page_source, "lxml")
            thread_urls = []

            items = soup.find_all("li")
            for li in items:
                a_tag = li.find("a", class_="subback_link", href=True)
                if not a_tag:
                    continue
                thread_url = a_tag["href"]
                thread_urls.append(thread_url)

            if thread_urls:  # 如果成功找到链接就返回
                print(f"[INFO] 找到 {len(thread_urls)} 个帖子链接")
                return thread_urls
            
            print(f"[WARN] 未找到任何帖子链接，尝试重试 ({retry_count + 1}/{MAX_RETRIES})")
            retry_count += 1
            human_like_sleep(BACKOFF_BASE * (2 ** retry_count))
            
        except Exception as e:
            print(f"[ERROR] 搜索页面解析出错: {e}")
            retry_count += 1
            if retry_count >= MAX_RETRIES:
                print("[ERROR] 达到最大重试次数，放弃该关键词")
                break
            human_like_sleep(BACKOFF_BASE * (2 ** retry_count))
    
    return []  # 如果多次重试后仍然失败，返回空列表

# ========== 线程抓取 ==========
def parse_thread_page(driver, thread_url: str, keyword: str):
    # (您的 driver.get 和 sleep 逻辑是正确的)
    driver.get(thread_url)
    time.sleep(2) # 等待页面加载
    
    soup = BeautifulSoup(driver.page_source, "lxml")
    posts = []

    # 1. 获取帖子标题 (使用您的方法，但目标更精确)
    # 标题在 <h1 id="title"> 标签中
    title_elem = soup.find("h1", id="title") 
    title = clean_text(title_elem.get_text()) if title_elem else "无标题"

    # 2. 遍历所有回帖 (使用正确的 <li> 选择器)
    # 查找所有 class 包含 "threadview_response" 的 <li> 标签
    li_items = soup.find_all("li", class_=re.compile(r"threadview_response"))
    
    # (!!!!) 您之前的循环 for post in soup.find_all("dl", class_="post"): 将不会在这里工作
    
    for li in li_items:
        # 3. 提取内容 (使用正确的 <div> class)
        # 帖子正文在 "threadview_response_body" 或 "threadview_response_detail"
        body_div = li.find("div", class_=re.compile(r"threadview_response_body|threadview_response_detail"))
        if not body_div:
            continue
        text = clean_text(body_div.get_text(" ", strip=True))

        # 4. 提取元数据 (作者、时间)
        info_div = li.find("div", class_=re.compile(r"threadview_response_info"))
        author, post_time = None, None
        if info_div:
            info_text = info_div.get_text(" ", strip=True)
            
            # 使用正则表达式从 "0001 神も仏も名無しさん 2025/10/27(月) 18:36:16.48" 中提取信息
            m_author = re.search(r"(名前|名無し|ID)\s*[:：]?\s*([^\s]+)", info_text)
            if m_author:
                author = m_author.group(2)
            else:
                # 备用方案：尝试提取第一个看起来像名字的部分
                name_match = re.search(r'\d+\s+(.*?)\s+\d{4}/', info_text)
                if name_match:
                    author = name_match.group(1).strip()

            m_time = re.search(r"(\d{4}[/-]\d{1,2}[/-]\d{1,2}[^0-9]{0,10}\d{1,2}:\d{2}:\d{2})", info_text)
            if m_time:
                post_time = m_time.group(1)

        post = {
            "thread_url": thread_url,
            "title": title, # (添加标题到数据中)
            "keyword": keyword,
            "text": text,
            "author": author,
            "time": post_time,
            "uid": uid_from_text(thread_url, text[:200])
        }
        posts.append(post)

    return posts

# ========== 主流程 ==========
def crawl_keywords(keywords: List[str]) -> None:
    """抓取所有关键词对应的帖子"""
    driver = init_browser()
    seen_uids = set()
    count = 0
    
    try:
        for keyword in keywords:
            print(f"\n[INFO] 开始抓取关键词: {keyword}")
            progress_file = os.path.join(PROGRESS_DIR, f"{keyword}_progress.json")
            
            # 检查进度文件
            if os.path.exists(progress_file):
                with open(progress_file, "r", encoding="utf-8") as f:
                    progress = json.load(f)
                print(f"[INFO] 找到进度文件，已完成 {len(progress['completed_urls'])} 个帖子")
                completed_urls = set(progress["completed_urls"])
                seen_uids.update(progress.get("seen_uids", []))
            else:
                completed_urls = set()
            
            # 获取搜索结果
            thread_urls = parse_search_page(driver, keyword)
            if not thread_urls:
                print(f"[WARN] 关键词 '{keyword}' 未找到任何结果")
                continue
                
            # 过滤已完成的URL
            thread_urls = [url for url in thread_urls if url not in completed_urls]
            print(f"[INFO] 找到 {len(thread_urls)} 个新帖子需要抓取")
            
            # 抓取每个帖子
            with open(OUTPUT_FILE, "a", encoding="utf-8") as fout:
                for i, url in enumerate(thread_urls, 1):
                    print(f"\n[INFO] 正在抓取第 {i}/{len(thread_urls)} 个帖子: {url}")
                    try:
                        posts = parse_thread_page(driver, url, keyword)
                        new_posts = 0
                        for post in posts:
                            if post["uid"] not in seen_uids:
                                seen_uids.add(post["uid"])
                                fout.write(json.dumps(post, ensure_ascii=False) + "\n")
                                count += 1
                                new_posts += 1
                                
                        print(f"[INFO] {url} -> 新增 {new_posts} 个帖子")
                        
                        # 更新进度
                        completed_urls.add(url)
                        progress = {
                            "keyword": keyword,
                            "completed_urls": list(completed_urls),
                            "seen_uids": list(seen_uids),
                            "last_update": datetime.now().isoformat()
                        }
                        with open(progress_file, "w", encoding="utf-8") as f:
                            json.dump(progress, f, ensure_ascii=False, indent=2)
                            
                    except Exception as e:
                        print(f"[ERROR] 帖子抓取失败: {url}")
                        print(f"[ERROR] 错误信息: {e}")
                        continue
                        
                    human_like_sleep(DELAY_BETWEEN_REQUESTS)
                    
    finally:
        print(f"[DONE] 累计保存 {count} 条帖子 -> {OUTPUT_FILE}")
        driver.quit()

# ========== 执行 ==========
if __name__ == "__main__":
    crawl_keywords(KEYWORDS)
