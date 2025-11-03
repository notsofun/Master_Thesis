import time
import csv, random
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# ===== 配置部分 =====
import os
import json
import pickle
from datetime import datetime
from typing import Dict, List, Any, Optional

# 基本配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COOKIES_FILE = os.path.join(BASE_DIR, "twitter_cookie.txt")
PROGRESS_DIR = os.path.join(BASE_DIR, "progress")
os.makedirs(PROGRESS_DIR, exist_ok=True)

KEYWORDS = ["Christianity", "church", "キリスト教"]
LANGUAGES = ["en", "ja"]
SINCE = "2025-07-01"
UNTIL = "2025-10-31"
MAX_SCROLL = 200  # 最大滚动次数
OUTPUT_FILE = "tweets_keyword_scraped.csv"

# 代理配置（可选）
USE_PROXY = False
PROXIES = [
    # "http://user:pass@ip:port",
    # "socks5://ip:port"
]

# 请求配置
RATE_LIMIT_WAIT = 900  # 频率限制时等待时间（秒）
MAX_RETRIES = 3  # 最大重试次数
# ==================

def save_cookies(driver):
    """保存 cookies 到文件"""
    cookies = driver.get_cookies()
    with open(COOKIES_FILE, 'w', encoding='utf-8') as f:
        json.dump(cookies, f)
    print("[INFO] Cookies 已保存")

def load_cookies(driver, cookies_file = COOKIES_FILE):
    """从文本 cookie 加载到 Selenium driver（兼容 key=value; key=value 格式）"""
    if not os.path.exists(cookies_file):
        print("[WARN] cookies 文件不存在")
        return False

    try:
        with open(cookies_file, "r", encoding="utf-8") as f:
            raw = f.read().strip()
        
        if not raw:
            print("[WARN] cookies 文件为空")
            return False

        # 分割 key=value
        cookies = []
        for item in raw.split("; "):
            if "=" in item:
                name, value = item.split("=", 1)
                cookies.append({"name": name, "value": value})
        
        # 添加到 driver
        for cookie in cookies:
            try:
                driver.add_cookie(cookie)
            except Exception as e:
                print(f"[WARN] 添加 cookie 失败: {cookie['name']}: {e}")
        
        print(f"[INFO] 成功加载 {len(cookies)} 条 cookies")
        return True

    except Exception as e:
        print(f"[WARN] 加载 cookies 失败: {e}")
        return False

def init_driver():
    options = Options()
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--start-maximized")

    driver = webdriver.Chrome(options=options)

    # 打开与 cookie 域名一致的页面
    driver.get("https://x.com/home")

    # 尝试加载 cookie
    if load_cookies(driver):
        print("[INFO] 已加载保存的 cookies，尝试登录...")
        driver.refresh()
        time.sleep(5)
        if not is_logged_in(driver):
            print("[WARN] Cookies 无效，需重新登录")
    else:
        print("[INFO] 未找到 cookies 文件")

    # 手动登录
    if not is_logged_in(driver):
        print("\n[INFO] 请在浏览器中手动登录 Twitter/X ...")
        input("[等待输入] 登录完成后请按 Enter 继续...")
        save_cookies(driver)
        print("[INFO] Cookies 已保存，继续执行任务...\n")

    return driver

def is_logged_in(driver):
    """通过检测个人主页按钮判断是否已登录"""
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "//a[contains(@href,'/home')]"))
        )
        return True
    except:
        return False

def build_query(keyword, lang, since, until):
    parts = [keyword]
    if lang:
        parts.append(f"lang:{lang}")
    if since:
        parts.append(f"since:{since}")
    if until:
        parts.append(f"until:{until}")
    return " ".join(parts)

def get_progress_file(keyword: str, lang: str) -> str:
    """获取断点续传文件路径"""
    safe_name = "".join(c if c.isalnum() else "_" for c in f"{keyword}_{lang}")
    return os.path.join(PROGRESS_DIR, f"{safe_name}_progress.pkl")

def load_progress(keyword: str, lang: str) -> tuple:
    """加载断点续传数据"""
    progress_file = get_progress_file(keyword, lang)
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'rb') as f:
                data = pickle.load(f)
                print(f"[INFO] 已加载断点数据: {len(data.get('data', []))} 条记录")
                return data.get('data', []), data.get('seen_texts', set())
        except Exception as e:
            print(f"[WARN] 加载断点数据失败: {e}")
    return [], set()

def save_progress(keyword: str, lang: str, data: list, seen_texts: set):
    """保存断点续传数据"""
    progress_file = get_progress_file(keyword, lang)
    try:
        with open(progress_file, 'wb') as f:
            pickle.dump({'data': data, 'seen_texts': seen_texts}, f)
        print(f"[INFO] 已保存断点数据: {len(data)} 条记录")
    except Exception as e:
        print(f"[WARN] 保存断点数据失败: {e}")

def handle_rate_limit(driver):
    """处理频率限制"""
    print(f"[WARN] 检测到频率限制，等待 {RATE_LIMIT_WAIT} 秒...")
    time.sleep(RATE_LIMIT_WAIT)
    driver.refresh()
    time.sleep(5)

def scrape_keyword(driver, keyword, lang):
    """抓取指定关键词和语言的推文数据"""
    query = build_query(keyword, lang, SINCE, UNTIL)
    search_url = f"https://twitter.com/search?q={query}&src=typed_query&f=live"
    print(f"\n[INFO] 开始抓取: {keyword} ({lang})")
    
    # 加载断点数据
    data, seen_texts = load_progress(keyword, lang)
    scroll_count = 0
    last_height = 0
    consecutive_fails = 0
    
    while scroll_count < MAX_SCROLL and consecutive_fails < 3:
        try:
            # 首次加载或需要刷新页面
            if scroll_count == 0:
                driver.get(search_url)
                time.sleep(5)
            
            # 检查登录状态和频率限制
            if not is_logged_in(driver):
                print("[警告] 需要重新登录")
                input("请在浏览器中登录后按 Enter 继续...")
                save_cookies(driver)
                time.sleep(3)
                driver.get(search_url)
                continue
            
            if "rate limit exceeded" in driver.page_source.lower():
                handle_rate_limit(driver)
                continue
            
            # 页面加载和解析
            time.sleep(2)
            html = driver.page_source
            soup = BeautifulSoup(html, "html.parser")
            articles = soup.find_all("article")
            new_count = 0
            
            # 处理每条推文
            for article in articles:
                try:
                    content_div = article.find("div", attrs={"lang": True})
                    if not content_div:
                        continue
                        
                    text = content_div.get_text(strip=True)
                    if text in seen_texts:
                        continue
                    seen_texts.add(text)
                    
                    # 解析推文数据
                    tweet_data = {
                        "keyword": keyword,
                        "lang": lang,
                        "content": text,
                        "datetime": "",
                        "url": "",
                        "author": "未知",
                        "likes": "0",
                        "retweets": "0",
                        "replies": "0",
                        "scrape_time": datetime.now().isoformat()
                    }
                    
                    # 提取元数据
                    time_tag = article.find("time")
                    if time_tag:
                        tweet_data["datetime"] = time_tag["datetime"]
                        
                    user_tag = article.find("a", href=lambda h: h and "/status/" in h)
                    if user_tag:
                        tweet_data["url"] = "https://twitter.com" + user_tag["href"]
                    
                    # 提取作者信息
                    author_div = article.find("div", attrs={"data-testid": "User-Name"})
                    if author_div:
                        tweet_data["author"] = author_div.get_text(strip=True)
                    
                    # 提取互动数据
                    stats = article.find_all("span", attrs={"data-testid": lambda x: x and "text" in x})
                    for stat in stats:
                        stat_id = stat.get("data-testid", "")
                        value = stat.get_text(strip=True) or "0"
                        if "like" in stat_id:
                            tweet_data["likes"] = value
                        elif "retweet" in stat_id:
                            tweet_data["retweets"] = value
                        elif "reply" in stat_id:
                            tweet_data["replies"] = value
                    
                    data.append(tweet_data)
                    new_count += 1
                    
                except Exception as e:
                    print(f"[WARN] 解析推文时出错: {str(e)[:100]}")
                    continue
            
            print(f"  第{scroll_count+1}次滚动，新推文 {new_count} 条，总计 {len(data)} 条")
            
            # 定期保存进度
            if new_count > 0 and len(data) % 50 == 0:
                save_progress(keyword, lang, data, seen_texts)
            
            # 向下滚动
            driver.find_element(By.TAG_NAME, "body").send_keys(Keys.END)
            time.sleep(2)
            
            # 检查是否到达底部
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                consecutive_fails += 1
                if consecutive_fails >= 3:
                    print("  已连续多次未发现新内容，可能已到达底部")
                    break
            else:
                consecutive_fails = 0
                last_height = new_height
            
            scroll_count += 1
            
        except Exception as e:
            print(f"[ERROR] 滚动过程出错: {str(e)[:100]}")
            consecutive_fails += 1
            time.sleep(5)
            
            if consecutive_fails >= 3:
                print("[WARN] 连续失败次数过多，保存当前进度并退出")
                break
    
    # 保存最终进度
    if data:
        save_progress(keyword, lang, data, seen_texts)
    
    return data

def main():
    driver = None
    all_data = []
    current_proxy_index = 0

    try:
        driver = init_driver()
        
        for kw in KEYWORDS:
            for lang in LANGUAGES:
                retry_count = 0
                while retry_count < MAX_RETRIES:
                    try:
                        # 如果配置了代理，在失败时切换
                        if USE_PROXY and PROXIES and retry_count > 0:
                            current_proxy_index = (current_proxy_index + 1) % len(PROXIES)
                            if driver:
                                driver.quit()
                            options = Options()
                            options.add_argument(f'--proxy-server={PROXIES[current_proxy_index]}')
                            driver = webdriver.Chrome(options=options)
                            load_cookies(driver)
                        
                        part = scrape_keyword(driver, kw, lang)
                        if part:  # 如果成功获取数据
                            all_data.extend(part)
                            # 每完成一个关键词就保存一次数据
                            df = pd.DataFrame(all_data)
                            df.to_csv(OUTPUT_FILE, index=False, quoting=csv.QUOTE_ALL)
                            print(f"[INFO] 已保存 {len(df)} 条记录到 {OUTPUT_FILE}")
                            break  # 成功后跳出重试循环
                        
                    except Exception as e:
                        retry_count += 1
                        print(f"[WARN] 抓取 {kw}-{lang} 时出错 (第{retry_count}次): {e}")
                        if "rate limit" in str(e).lower():
                            handle_rate_limit(driver)
                        elif retry_count < MAX_RETRIES:
                            time.sleep(10 * retry_count)  # 递增等待时间
                        else:
                            print(f"[ERROR] 达到最大重试次数，跳过 {kw}-{lang}")
                            break
                            
    except Exception as e:
        print(f"[ERROR] 程序执行出错: {e}")
        
    finally:
        if driver:
            try:
                driver.quit()
            except:
                pass
        
        # 保存最终数据
        if all_data:
            try:
                df = pd.DataFrame(all_data)
                df.to_csv(OUTPUT_FILE, index=False, quoting=csv.QUOTE_ALL)
                print(f"\n✅ 抓取完毕，共 {len(df)} 条推文，已保存至 {OUTPUT_FILE}")
            except Exception as e:
                print(f"[ERROR] 保存数据失败: {e}")
                # 尝试保存为备份文件
                try:
                    backup_file = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    df.to_csv(backup_file, index=False, quoting=csv.QUOTE_ALL)
                    print(f"[INFO] 数据已保存至备份文件: {backup_file}")
                except:
                    print("[ERROR] 备份保存也失败，建议检查数据并手动处理")

    driver.quit()

    df = pd.DataFrame(all_data)
    df.to_csv(OUTPUT_FILE, index=False, quoting=csv.QUOTE_ALL)
    print(f"\n✅ 抓取完毕，共 {len(df)} 条推文，已保存至 {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
