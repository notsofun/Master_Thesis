#!/usr/bin/env python3
# coding: utf-8
"""
undetected-chromedriver-based Twitter/X keyword scraper
- 支持加载/保存 cookie（JSON 或 key=value; 文本）
- 手动登录一次后会保存 cookie，便于下次复用
- 按关键词+语言+since/until 搜索，自动滚动抽取 article 内容
- 增量写入 CSV（断点可续）
注意：请在 headful 模式下首次手动登录（headless=False）
"""

import os
import time
import json
import csv
import random
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import pandas as pd

# ========== 配置区 ==========
COOKIES_FILE = "twitter_cookie.json"   # 保存的 cookie（JSON list）或 key=value 文本
OUTPUT_CSV = "tweets_uc_output.csv"
PROGRESS_PKL = "progress_uc.pkl"

KEYWORDS = ["Christianity", "church", "キリスト教"]
LANGUAGES = ["en", "ja"]   # lang codes
SINCE = "2025-07-01"       # 可设为空 ""
UNTIL = "2025-10-31"       # 可设为空 ""
MAX_TWEETS_PER_QUERY = 2000    # 单个 keyword+lang 的上限（None 表示不限）

HEADLESS = False           # 首次登录请设为 False
SCROLL_PAUSE = 1.0         # 每次滚动等待时间
MAX_SCROLL_ITERS = 300
WRITE_BATCH = 50           # 每抓到多少条写入一次 CSV
# ==========================

def build_query(keyword: str, lang: str, since: str, until: str) -> str:
    parts = [keyword]
    if lang:
        parts.append(f"lang:{lang}")
    if since:
        parts.append(f"since:{since}")
    if until:
        parts.append(f"until:{until}")
    return " ".join(parts)

def save_cookies_from_driver(driver, path: str = COOKIES_FILE):
    cookies = driver.get_cookies()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cookies, f, indent=2, ensure_ascii=False)
    print(f"[INFO] 已保存 {len(cookies)} cookies -> {path}")

def read_text_cookie_string(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
            return text if text else None
    except:
        return None

def load_cookies_to_driver(driver, path: str = COOKIES_FILE) -> bool:
    """
    支持两种文件形式：
      - JSON array (driver.get_cookies() 导出)
      - 单行 key=value; key2=value2 文本 (浏览器复制的 cookie)
    返回 True 表示尝试加载（但不一定有效）
    """
    if not os.path.exists(path):
        print("[WARN] cookie 文件不存在:", path)
        return False
    # 尝试 JSON 格式
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read().strip()
            if not raw:
                print("[WARN] cookie 文件为空")
                return False
            # detect if key=value style (contains ';' and not starting with '[')
            if raw.startswith("[") and raw.endswith("]"):
                cookies = json.loads(raw)
            elif ";" in raw and "=" in raw:
                # parse key=value; key2=value2
                cookies = []
                for item in raw.split(";"):
                    item = item.strip()
                    if not item or "=" not in item:
                        continue
                    k, v = item.split("=", 1)
                    cookies.append({"name": k, "value": v})
            else:
                # try JSON parse fallback
                cookies = json.loads(raw)
    except Exception as e:
        print("[WARN] 解析 cookie 文件出错:", e)
        return False

    # add cookies; must be on same domain before adding
    # we assume driver already opened x.com/twitter.com
    added = 0
    for c in cookies:
        c2 = c.copy()
        # Selenium add_cookie 不接受 keys like 'sameSite' sometimes; sanitize
        for bad in ("sameSite", "same_site", "_expires", "hostOnly"):
            c2.pop(bad, None)
        # expiry -> expiry as int (if present, convert)
        if "expiry" in c2 and isinstance(c2["expiry"], float):
            try:
                c2["expiry"] = int(c2["expiry"])
            except:
                c2.pop("expiry", None)
        try:
            driver.add_cookie(c2)
            added += 1
        except Exception as e:
            # often domain mismatch or HttpOnly cookie cannot be added
            # ignore silently but could print small debug
            # print("cookie add fail", c2.get("name"), e)
            pass
    print(f"[INFO] 尝试加载 cookie，总共尝试 {len(cookies)} 条, 成功添加 {added} 条 (注意：HttpOnly cookie 不能被注入)")
    return added > 0

def is_logged_in(driver, timeout: int = 8) -> bool:
    # 检测登录后特有的元素：Compose tweet 链接或 Profile 链接（/home 通常存在）
    try:
        time.sleep(1)
        # quick source check: profile link or compose
        src = driver.page_source
        if "/compose/tweet" in src or "aria-label=\"Profile\"" in src or "/home" in src:
            return True
        # fallback: look for tweet box presence by xpath
        try:
            # small sleep for dynamic render
            time.sleep(1)
            if "Log in" in src and "Sign in" in src:
                return False
        except:
            pass
        return not ("Log in" in src or "Log in" in src or "Sign in" in src)
    except:
        return False

def start_driver(headless: bool = HEADLESS):
    opts = uc.ChromeOptions()
    if headless:
        opts.add_argument("--headless=new")
    # make it less detectable
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    # you may add user-data-dir for persistent profile if desired
    driver = uc.Chrome(options=opts)
    driver.set_page_load_timeout(60)
    return driver

def extract_from_article_html(article_html) -> Optional[Dict[str, str]]:
    try:
        # article_html is bs4 element
        # content
        content_div = article_html.find("div", attrs={"lang": True})
        if not content_div:
            return None
        text = content_div.get_text(" ", strip=True)
        # url / id
        a_status = article_html.find("a", href=lambda h: h and "/status/" in h)
        url = ""
        tweet_id = ""
        if a_status:
            href = a_status["href"]
            if href.startswith("/"):
                url = "https://x.com" + href
            else:
                url = href
            # parse id
            parts = href.split("/")
            for p in reversed(parts):
                if p.isdigit():
                    tweet_id = p
                    break
        # time
        time_tag = article_html.find("time")
        dt = time_tag["datetime"] if time_tag and time_tag.has_attr("datetime") else ""
        # author
        author_handle = ""
        try:
            # often anchor with href="/username"
            user_a = article_html.find("a", href=lambda h: h and h.count("/") == 2 and h.startswith("/"))
            if user_a:
                author_handle = user_a["href"].split("/")[1]
        except:
            pass

        return {
            "tweet_id": tweet_id,
            "url": url,
            "datetime": dt,
            "author_handle": author_handle,
            "content": text
        }
    except Exception:
        return None

def scrape_search(driver, query: str, lang: str, max_items: Optional[int]=None) -> List[Dict[str, Any]]:
    enc = query.replace(" ", "%20")
    search_url = f"https://x.com/search?q={enc}&src=typed_query&f=live"
    print("[INFO] 打开搜索页：", search_url)
    driver.get(search_url)
    time.sleep(3)

    seen_ids = set()
    rows = []
    scroll_iters = 0
    empty_streak = 0

    while True:
        scroll_iters += 1
        # page source -> bs4
        html = driver.page_source
        soup = BeautifulSoup(html, "html.parser")
        articles = soup.find_all("article")
        new_found = 0
        for art in articles:
            info = extract_from_article_html(art)
            if not info:
                continue
            tid = info.get("tweet_id") or info.get("url")
            if not tid or tid in seen_ids:
                continue
            seen_ids.add(tid)
            info["lang"] = lang
            info["scrape_time"] = datetime.utcnow().isoformat()
            rows.append(info)
            new_found += 1
            # cap
            if max_items and len(rows) >= max_items:
                break
        if new_found == 0:
            empty_streak += 1
        else:
            empty_streak = 0

        if len(rows) > 0 and len(rows) % WRITE_BATCH == 0:
            # append to CSV incrementally
            append_to_csv(rows, OUTPUT_CSV)
            rows = []  # flush buffer

        print(f"  滚动 {scroll_iters}: 本次新增 {new_found} 条, 累计 {len(seen_ids)} 条")
        # stop conditions
        if max_items and len(seen_ids) >= max_items:
            print("[INFO] 达到单 query 上限，停止")
            break
        if scroll_iters >= MAX_SCROLL_ITERS:
            print("[INFO] 达到最大滚动次数，停止")
            break
        if empty_streak >= 6:
            print("[INFO] 连续多次无新内容，可能已加载完毕，停止")
            break

        # scroll
        try:
            driver.find_element(By.TAG_NAME, "body").send_keys(Keys.END)
        except Exception:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(SCROLL_PAUSE + random.random() * 0.6)
    # final flush
    if rows:
        append_to_csv(rows, OUTPUT_CSV)
    # return nothing (results saved to CSV)
    return []

def append_to_csv(rows: List[Dict[str, Any]], path: str):
    if not rows:
        return
    p = Path(path)
    write_header = not p.exists()
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["tweet_id","url","datetime","author_handle","content","lang","scrape_time"])
        if write_header:
            writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"[INFO] 写入 {len(rows)} 条到 {path}")

def ensure_driver_and_login(driver):
    # open x.com to match cookie domain
    try:
        driver.get("https://x.com")
    except:
        driver.get("https://twitter.com")
    time.sleep(3)
    loaded = load_cookies_to_driver(driver)
    if loaded:
        # refresh to apply
        driver.refresh()
        time.sleep(5)
    if not is_logged_in(driver):
        print("[INFO] cookie 未能自动登录或未提供 cookie，请手动在当前窗口登录 X/Twitter（完成后回终端按回车）")
        input("手动登录完成后按 Enter 继续...")
        # attempt to save fresh cookies
        try:
            save_cookies_from_driver(driver)
        except Exception as e:
            print("[WARN] 保存 cookie 失败:", e)
    else:
        print("[INFO] cookie 有效，已登录。")

def main():
    driver = start_driver(headless=HEADLESS)
    try:
        ensure_driver_and_login(driver)
        # 主抓取循环
        for kw in KEYWORDS:
            for lang in LANGUAGES:
                q = build_query(kw, lang, SINCE, UNTIL)
                # 每个关键词语言限定单独文件或合并到同一个 OUTPUT_CSV（这里合并）
                print(f"\n=== 开始抓取: '{kw}' lang={lang} ===")
                scrape_search(driver, q, lang, max_items=MAX_TWEETS_PER_QUERY)
                # small break between queries
                time.sleep(2 + random.random() * 2)
        print("\n[INFO] 所有关键词抓取完成。")
    finally:
        try:
            driver.quit()
        except:
            pass

if __name__ == "__main__":
    main()
