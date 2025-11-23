import os
import time
import csv
import json
import random
import logging
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
LOG_DIR = os.path.join(BASE_DIR, "log")
MAX_POSTS_PER_THREAD = 5000

os.makedirs(PROGRESS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ------------ 日志配置 ------------
log_filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".log"
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

# 搜索配置
KEYWORD_JSON = "../fianl_keywors.json"
OUTPUT_CSV = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_5ch_posts.csv"  # CSV输出文件
DELAY_BETWEEN_REQUESTS = random.uniform(2, 4)  # 请求间隔（随机）
MAX_RETRIES = 3  # 最大重试次数
BACKOFF_BASE = 5  # 重试等待基础时间（秒）

# ========== 关键词与CSV管理函数 ==========
def load_keywords_from_json(path=KEYWORD_JSON, key="Japanese"):
    """从JSON文件加载关键词"""
    try:
        logger.info("📖 正在加载关键词文件: %s", path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        keywords = data.get(key, [])
        logger.info("✅ 成功加载 %d 个关键词", len(keywords))
        return keywords
    except Exception as e:
        logger.error("❌ 加载关键词失败: %s", e)
        return []

def append_to_csv(rows: List[Dict], filename=OUTPUT_CSV, header=None):
    """追加数据到CSV文件"""
    if not rows:
        logger.warning("⚠️ 无数据，跳过CSV保存")
        return
    
    try:
        file_exists = os.path.exists(filename)
        filepath = os.path.join(BASE_DIR, filename)
        
        with open(filepath, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            if not file_exists:
                writer.writeheader()
                logger.info("📝 创建新CSV文件: %s", filename)
            writer.writerows(rows)
            logger.info("✅ 已保存 %d 条记录到 %s", len(rows), filename)
    except Exception as e:
        logger.error("❌ CSV保存失败: %s", e)
        raise

# ========== 工具函数 ==========
def init_browser() -> webdriver.Chrome:
    """初始化Chrome浏览器"""
    logger.info("🌐 初始化浏览器...")
    try:
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
        logger.info("✅ 浏览器初始化成功")
        
        # 如果有保存的cookies，加载它们
        if os.path.exists(COOKIE_FILE):
            logger.info("🍪 加载已保存的cookies...")
            try:
                driver.get("https://itest.5ch.net")
                with open(COOKIE_FILE, "r", encoding="utf-8") as f:
                    cookies = json.load(f)
                for cookie in cookies:
                    try:
                        driver.add_cookie(cookie)
                    except Exception:
                        pass
                logger.info("✅ Cookies加载完成")
            except Exception as e:
                logger.warning("⚠️ 加载cookies失败: %s", e)
        
        return driver
    except Exception as e:
        logger.error("❌ 浏览器初始化失败: %s", e)
        raise

def clean_text(s: str) -> str:
    """清理文本中的多余空白"""
    if not s:
        return ""
    return re.sub(r"\s+", " ", str(s)).strip()

def uid_from_text(*parts) -> str:
    """生成文本内容的唯一标识"""
    import hashlib
    text = "||".join(map(str, parts))
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

def safe_filename(s: str) -> str:
    """生成安全的文件名"""
    return "".join(c if c.isalnum() else "_" for c in s)[:200]

def human_like_sleep(min_s=2.0, max_s=4.0):
    """模拟人工操作的随机等待"""
    sleep_time = random.uniform(min_s, max_s)
    time.sleep(sleep_time)

def save_progress(keyword: str, state: Dict[str, Any]):
    """保存断点续传状态"""
    try:
        path = os.path.join(PROGRESS_DIR, f"{safe_filename(keyword)}_progress.pkl")
        with open(path, "wb") as f:
            pickle.dump(state, f)
        logger.info("💾 进度已保存: %s", keyword)
    except Exception as e:
        logger.error("❌ 进度保存失败: %s", e)

def load_progress(keyword: str):
    """加载断点续传状态"""
    try:
        path = os.path.join(PROGRESS_DIR, f"{safe_filename(keyword)}_progress.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                state = pickle.load(f)
            logger.info("📂 已加载进度: %s", keyword)
            return state
    except Exception as e:
        logger.warning("⚠️ 进度加载失败: %s", e)
    return None

# ========== 搜索页抓取 ==========
def parse_search_page(driver: webdriver.Chrome, keyword: str) -> List[str]:
    """解析搜索结果页面"""
    search_url = f"https://itest.5ch.net/find?q={urllib.parse.quote(keyword)}"
    logger.info("🔍 开始搜索关键词: %s", keyword)
    logger.info("🌐 搜索URL: %s", search_url)
    
    retry_count = 0
    while retry_count < MAX_RETRIES:
        try:
            logger.info("📬 发送搜索请求... (尝试 %d/%d)", retry_count + 1, MAX_RETRIES)
            driver.get(search_url)
            human_like_sleep(2, 4)  # 等待页面加载
            
            # 保存当前进度
            if retry_count == 0:
                try:
                    cookies = driver.get_cookies()
                    with open(COOKIE_FILE, "w", encoding="utf-8") as f:
                        json.dump(cookies, f, ensure_ascii=False, indent=2)
                    logger.info("💾 Cookies已更新")
                except Exception as e:
                    logger.warning("⚠️ 保存cookies失败: %s", e)

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
                logger.info("✅ 找到 %d 个帖子链接", len(thread_urls))
                return thread_urls
            
            logger.warning("⚠️ 未找到任何帖子链接，尝试重试 (%d/%d)", retry_count + 1, MAX_RETRIES)
            retry_count += 1
            wait_time = BACKOFF_BASE * (2 ** retry_count)
            logger.info("⏳ 等待 %d 秒后重试...", wait_time)
            human_like_sleep(wait_time)
            
        except Exception as e:
            logger.error("❌ 搜索页面解析出错: %s", e)
            retry_count += 1
            if retry_count >= MAX_RETRIES:
                logger.error("❌ 达到最大重试次数，放弃该关键词")
                break
            wait_time = BACKOFF_BASE * (2 ** retry_count)
            logger.info("⏳ 错误恢复，等待 %d 秒后重试...", wait_time)
            human_like_sleep(wait_time)
    
    logger.warning("⚠️ 关键词搜索失败，返回空列表")
    return []

# ========== 线程抓取 ==========
def parse_thread_page(driver: webdriver.Chrome, thread_url: str, keyword: str) -> List[Dict[str, Any]]:
    """解析线程/帖子页面"""
    logger.info("📖 开始解析线程页面: %s", thread_url)
    posts = []
    
    retry_count = 0
    while retry_count < MAX_RETRIES:
        try:
            logger.info("📥 加载页面... (尝试 %d/%d)", retry_count + 1, MAX_RETRIES)
            driver.get(thread_url)
            human_like_sleep(2, 4)  # 等待页面加载
            
            soup = BeautifulSoup(driver.page_source, "lxml")

            # 1. 获取帖子标题
            title_elem = soup.find("h1", id="title")
            title = clean_text(title_elem.get_text()) if title_elem else "无标题"
            logger.info("📋 线程标题: %s", title[:50])

            # 2. 遍历所有回帖
            li_items = soup.find_all("li", class_=re.compile(r"threadview_response"))
            logger.info("🔍 找到 %d 条回帖", len(li_items))
            
            if not li_items:
                logger.warning("⚠️ 未找到任何回帖内容")
                return []
            
            for idx, li in enumerate(li_items, 1):
                try:
                    if idx > MAX_POSTS_PER_THREAD:
                        break

                    # 3. 提取内容
                    body_div = li.find("div", class_=re.compile(r"threadview_response_body|threadview_response_detail"))
                    if not body_div:
                        logger.debug("⚠️ 第 %d 条回帖: 未找到内容", idx)
                        continue
                    text = clean_text(body_div.get_text(" ", strip=True))

                    # 4. 提取元数据
                    info_div = li.find("div", class_=re.compile(r"threadview_response_info"))
                    author, post_time = "未知", "未知"
                    
                    if info_div:
                        info_text = info_div.get_text(" ", strip=True)
                        
                        # 提取作者
                        m_author = re.search(r"(名前|名無し|ID)\s*[:\uff1a]?\s*([^\s]+)", info_text)
                        if m_author:
                            author = m_author.group(2)
                        else:
                            name_match = re.search(r'\d+\s+(.*?)\s+\d{4}/', info_text)
                            if name_match:
                                author = name_match.group(1).strip()

                        # 提取时间
                        m_time = re.search(r"(\d{4}[/-]\d{1,2}[/-]\d{1,2}[^0-9]{0,10}\d{1,2}:\d{2}:\d{2})", info_text)
                        if m_time:
                            post_time = m_time.group(1)

                    # 创建帖子数据
                    post = {
                        "keyword": keyword,
                        "thread_url": thread_url,
                        "title": title,
                        "author": author,
                        "time": post_time,
                        "text": text,
                        "uid": uid_from_text(thread_url, text[:200])
                    }
                    posts.append(post)
                    logger.debug("✅ 第 %d 条回帖已提取", idx)
                    
                except Exception as e:
                    logger.warning("⚠️ 第 %d 条回帖提取失败: %s", idx, e)
                    continue

            if posts:
                logger.info("✅ 成功提取 %d 条回帖", len(posts))
                return posts
            else:
                logger.warning("⚠️ 未成功提取任何回帖")
                retry_count += 1
                if retry_count < MAX_RETRIES:
                    wait_time = BACKOFF_BASE * (2 ** retry_count)
                    logger.info("⏳ 等待 %d 秒后重试...", wait_time)
                    human_like_sleep(wait_time)
                continue
                
        except Exception as e:
            logger.error("❌ 页面解析出错: %s", e)
            retry_count += 1
            if retry_count >= MAX_RETRIES:
                logger.error("❌ 达到最大重试次数，放弃该页面")
                break
            wait_time = BACKOFF_BASE * (2 ** retry_count)
            logger.info("⏳ 错误恢复，等待 %d 秒后重试...", wait_time)
            human_like_sleep(wait_time)
    
    logger.warning("⚠️ 页面解析最终失败，返回空列表")
    return []

# ========== 主流程 ==========
def crawl_keywords(keywords: List[str]) -> None:
    """抓取所有关键词对应的帖子"""
    logger.info("="*60)
    logger.info("🚀 开始爬虫任务")
    logger.info("="*60)
    
    driver = None
    try:
        driver = init_browser()
        seen_uids = set()
        total_count = 0
        
        for kw_idx, keyword in enumerate(keywords, 1):
            logger.info("")
            logger.info("="*60)
            logger.info("📌 处理关键词 [%d/%d]: %s", kw_idx, len(keywords), keyword)
            logger.info("="*60)
            
            try:
                # 检查进度文件
                progress = load_progress(keyword)
                if progress:
                    completed_urls = set(progress.get("completed_urls", []))
                    seen_uids.update(progress.get("seen_uids", []))
                    logger.info("📂 从进度恢复: 已完成 %d 个线程", len(completed_urls))
                else:
                    completed_urls = set()
                
                # 获取搜索结果
                logger.info("")
                logger.info("🔎 正在搜索...")
                thread_urls = parse_search_page(driver, keyword)
                if not thread_urls:
                    logger.warning("⚠️ 关键词 '%s' 未找到任何结果", keyword)
                    continue
                    
                # 过滤已完成的URL
                new_urls = [url for url in thread_urls if url not in completed_urls]
                logger.info("📊 搜索结果: 总计 %d 个线程，其中 %d 个为新线程", 
                           len(thread_urls), len(new_urls))
                
                if not new_urls:
                    logger.info("✅ 所有线程都已处理")
                    continue
                
                # 抓取每个线程
                batch_data = []
                for i, url in enumerate(new_urls, 1):
                    logger.info("")
                    logger.info("📰 [%d/%d] 正在抓取线程: %s", i, len(new_urls), url)
                    
                    try:
                        posts = parse_thread_page(driver, url, keyword)
                        new_posts_count = 0
                        
                        for post in posts:
                            if post["uid"] not in seen_uids:
                                seen_uids.add(post["uid"])
                                batch_data.append(post)
                                new_posts_count += 1
                                total_count += 1
                        
                        logger.info("✅ 线程完成: 新增 %d 条记录 (累计: %d)", 
                                   new_posts_count, total_count)
                        
                        # 每处理完一个线程就保存批量数据
                        if batch_data and len(batch_data) >= 10:
                            header = list(batch_data[0].keys())
                            append_to_csv(batch_data, filename=OUTPUT_CSV, header=header)
                            batch_data = []
                            
                    except Exception as e:
                        logger.error("❌ 线程抓取失败: %s", url)
                        logger.error("   错误信息: %s", e)
                        # 继续处理其他线程
                        continue
                    
                    # 更新进度（每处理完一个线程）
                    completed_urls.add(url)
                    state = {
                        "keyword": keyword,
                        "completed_urls": list(completed_urls),
                        "seen_uids": list(seen_uids),
                        "last_update": datetime.now().isoformat()
                    }
                    save_progress(keyword, state)
                    
                    human_like_sleep(DELAY_BETWEEN_REQUESTS)
                
                # 保存剩余的批量数据
                if batch_data:
                    header = list(batch_data[0].keys())
                    append_to_csv(batch_data, filename=OUTPUT_CSV, header=header)
                
                logger.info("")
                logger.info("✅ 关键词 '%s' 处理完成", keyword)
                
            except Exception as e:
                logger.error("❌ 关键词 '%s' 处理失败: %s", keyword, e)
                # 不影响后续关键词处理
                continue
        
        logger.info("")
        logger.info("="*60)
        logger.info("✅ 爬虫任务完成")
        logger.info("📊 统计: 总计保存 %d 条记录", total_count)
        logger.info("📁 输出文件: %s", OUTPUT_CSV)
        logger.info("="*60)
        
    except Exception as e:
        logger.error("❌ 爬虫任务失败: %s", e)
        logger.error("   请检查日志获取详细信息")
    finally:
        if driver:
            try:
                logger.info("🚪 关闭浏览器...")
                driver.quit()
                logger.info("✅ 浏览器已关闭")
            except Exception as e:
                logger.warning("⚠️ 关闭浏览器失败: %s", e)

# ========== 执行 ==========
if __name__ == "__main__":
    logger.info("")
    logger.info("="*60)
    logger.info("📌 5ch爬虫启动")
    logger.info("="*60)
    
    try:
        keywords = load_keywords_from_json(KEYWORD_JSON, key="Japanese")
        if keywords:
            crawl_keywords(keywords)
        else:
            logger.error("❌ 未能加载任何关键词")
    except Exception as e:
        logger.error("❌ 程序异常: %s", e)
        import traceback
        logger.error(traceback.format_exc())

