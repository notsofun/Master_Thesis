# tieba_crawler.py
import os
import time
import csv
import json
import random
import pickle
from typing import List, Dict, Any

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

import logging
from datetime import datetime

# ------------ 日志配置 ------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "log")
os.makedirs(LOG_DIR, exist_ok=True)

log_filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".log"
log_path = os.path.join(LOG_DIR, log_filename)

logger = logging.getLogger("tieba_crawler")
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

logger.info("日志系统初始化完成：%s", log_path)


KEYWORD_JSON = "../fianl_keywors.json"
OUTPUT_CSV = "all_search_posts.csv"

def load_keywords_from_json(path=KEYWORD_JSON, key="Chinese"):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get(key, [])


def append_to_big_csv(rows, filename=OUTPUT_CSV, header=None):
    file_exists = os.path.exists(filename)
    with open(filename, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


# ---------------- 基本配置 ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COOKIE_FILE = os.path.join(BASE_DIR, "tieba_cookies.json")
PROGRESS_DIR = os.path.join(BASE_DIR, "progress")
os.makedirs(PROGRESS_DIR, exist_ok=True)

# 可替换为你的关键词（请使用受控词表或占位符）
KEYWORDS = [
    # "耶狗",     # (yē gǒu) "Jesus dog" - 一種非常粗俗的侮辱性稱呼，將信徒比作狗，是直接的人身攻擊。
    # "基督狗",   # (jīdū gǒu) "Christ dog" - 同上，更為直接。
    # "洋教",     # (yáng jiào) "Foreign religion" - 歷史上帶有排外和貶義的色彩，尤其在清末民初，暗示其非本土、與西方帝國主義侵略相關。
    "二毛子",   # (èr máozi) 歷史用語，"毛子"是舊時對俄國人（後泛指外國人）的蔑稱。"二毛子"侮辱性地稱呼那些皈依基督教並與外國人為伍的中國人，暗含「漢奸」或「走狗」的意思。
    "神棍",     # (shén gùn) "God swindler" / "Divine charlatan" - 泛指所有利用宗教或迷信進行欺騙、斂財的人。雖然不專指基督教，但在反宗教語境中，常用來攻擊神職人員或傳教士。
    "聖母婊"    # (shèngmǔ biǎo) "Saintly Mother Bitch" - 這是一個現代網絡俚語，並不專門針對基督徒。它用來嘲諷那些表現得過於寬容、虛偽、不切實際、"聖母般"的人（常與基督教的「博愛」、「寬恕」觀念相聯繫）。
]

# Selenium/Chrome 配置
options = Options()
options.add_argument("--start-maximized")
# 推荐使用真实 profile（取消注释并按需设置路径）
# options.add_argument("--user-data-dir=C:/Users/youruser/AppData/Local/Google/Chrome/User Data")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
wait = WebDriverWait(driver, 12)

# ---------------- 登录与 cookie ----------------
def login_with_cookie():
    driver.get("https://tieba.baidu.com/")
    if os.path.exists(COOKIE_FILE):
        try:
            with open(COOKIE_FILE, "r", encoding="utf-8") as f:
                cookies = json.load(f)
            for cookie in cookies:
                # 某些 cookie 可能缺少 domain 等字段，按需过滤
                try:
                    driver.add_cookie(cookie)
                except Exception:
                    pass
            driver.refresh()
            logger.info("✅ 已加载 cookie 并刷新页面")
        except Exception as e:
            logger.error(f"⚠️ 加载 cookie 失败: {e}")
    else:
        logger.warning("请手动在打开的浏览器中完成登录（扫码等），登录后按 Enter 保存 cookie 并继续...")
        input()
        cookies = driver.get_cookies()
        with open(COOKIE_FILE, "w", encoding="utf-8") as f:
            json.dump(cookies, f, ensure_ascii=False, indent=2)
        logger.info("✅ 登录成功，cookie 已保存")


# ---------------- 工具函数 ----------------
def save_progress(keyword: str, state: Dict[str, Any]):
    path = os.path.join(PROGRESS_DIR, f"{safe_filename(keyword)}_progress.pkl")
    with open(path, "wb") as f:
        pickle.dump(state, f)

def load_progress(keyword: str):
    path = os.path.join(PROGRESS_DIR, f"{safe_filename(keyword)}_progress.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

def safe_filename(s: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in s)[:200]

def human_like_sleep(min_s=1.2, max_s=3.5):
    time.sleep(random.uniform(min_s, max_s))

def human_scroll(driver):
    # 小段滚动模拟人工浏览
    try:
        driver.execute_script("window.scrollBy(0, document.body.scrollHeight*0.25);")
        time.sleep(random.uniform(0.4, 1.0))
        driver.execute_script("window.scrollBy(0, document.body.scrollHeight*0.5);")
        time.sleep(random.uniform(0.6, 1.2))
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    except Exception:
        pass

def safe_switch_to_main(driver):
    try:
        if driver.window_handles:
            driver.switch_to.window(driver.window_handles[0])
    except Exception as e:
        logger.error(f"⚠️ 切换主窗口失败: {e}")

# ---------------- 帖子详情抓取 ----------------
def parse_post_detail(driver, link: str):
    post_data = []
    try:
        driver.execute_script("window.open(arguments[0])", link)
        driver.switch_to.window(driver.window_handles[-1])
        time.sleep(random.uniform(0.8, 1.5))

        # 获取标题（多选择器兜底）
        title = None
        title_selectors = [
            (By.CSS_SELECTOR, "h3.core_title_txt"),
            (By.CSS_SELECTOR, "h1.core_title_txt"),
            (By.CSS_SELECTOR, "h1#j_core_title"),
            (By.CSS_SELECTOR, "div.thread_title h1"),
        ]
        for sel in title_selectors:
            try:
                elem = WebDriverWait(driver, 6).until(EC.presence_of_element_located(sel))
                title = elem.text.strip()
                break
            except Exception:
                title = None

        if not title:
            try:
                og = driver.find_elements(By.XPATH, "//meta[@property='og:title']")
                if og:
                    title = og[0].get_attribute("content")
            except Exception:
                title = None
        if not title:
            try:
                title = driver.title or "未知标题"
            except Exception:
                title = "未知标题"

        # 抓取楼层元素（若不存在则返回空）
        try:
            WebDriverWait(driver, 6).until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.l_post")))
            all_posts = driver.find_elements(By.CSS_SELECTOR, "div.l_post")
        except Exception:
            all_posts = []

        for idx, p in enumerate(all_posts, start=1):
            try:
                author = p.find_element(By.CSS_SELECTOR, "li.d_name a").text
            except:
                author = "未知"
            try:
                time_text = p.find_element(By.CSS_SELECTOR, "div.post-tail-wrap span.tail-info:last-child").text
            except:
                time_text = "未知"
            try:
                main_content = p.find_element(By.CSS_SELECTOR, "div.d_post_content").text.strip()
            except:
                main_content = ""

            lzl_contents = []
            try:
                lzls = p.find_elements(By.CSS_SELECTOR, "div.lzl_content_main")
                for lzl in lzls:
                    lzl_contents.append(lzl.text.strip())
            except:
                pass

            post_data.append({
                "title": title,
                "floor": idx,
                "author": author,
                "time": time_text,
                "main_content": main_content,
                "lzl_replies": " || ".join(lzl_contents),
                "link": link
            })
    except Exception as e:
        logger.error(f"❌ 抓取帖子详情失败 ({link}): {e}")
    finally:
        # 只关闭当前标签页
        try:
            if len(driver.window_handles) > 1:
                driver.close()
        except Exception as e:
            logger.error(f"⚠️ 关闭标签页失败: {e}")
        safe_switch_to_main(driver)

    return post_data

# ---------------- 搜索与抓取主逻辑 ----------------
def is_captcha_present(driver) -> bool:
    """检测常见的安全验证痕迹（标题或页面源码中包含关键字）"""
    try:
        title = driver.title.lower()
        src = driver.page_source.lower()
        indicators = ["安全验证", "验证码", "请滑动", "请完成验证"]
        return any(ind in title or ind in src for ind in indicators)
    except Exception:
        return False

def crawl_tieba_search(keyword: str, pages: int = 3, start_page: int = 0):
    logger.info(f"🔍 开始关键词：{keyword} 从第 {start_page+1} 页，共 {pages} 页")
    url = f"https://tieba.baidu.com/f/search/res?ie=utf-8&qw={keyword}"
    driver.get(url)
    human_like_sleep(1.2, 2.5)

    results = []
    consecutive_failures = 0
    backoff_base = 10

    for page in range(start_page, pages):
        try:
            logger.info(f"📄 正在抓取第 {page+1} 页...")
            human_scroll(driver)
            human_like_sleep(1.0, 2.5)

            # 验证检测
            if is_captcha_present(driver):
                logger.warning("⚠️ 检测到安全验证（captcha）。请在浏览器中完成验证后按 Enter 继续。")
                save_progress(keyword, {"page": page, "results": results})
                input("完成验证后按 Enter 继续...")
                driver.refresh()
                human_like_sleep(1.5, 3)
                # 继续重试当前页
                continue

            # 触发懒加载
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            human_like_sleep(0.8, 1.6)

            # 获取搜索结果条目（s_post）
            posts = driver.find_elements(By.CSS_SELECTOR, "div.s_post")
            links = []
            for post in posts:
                try:
                    a = post.find_element(By.CSS_SELECTOR, "a.bluelink")
                    href = a.get_attribute("href")
                    if href:
                        links.append(href)
                except Exception:
                    continue

            # 降低打开详情页频率，按采样比随机抽取
            sample_rate = 0.45
            sample_links = [l for l in links if random.random() < sample_rate]

            for link in sample_links:
                # 抓取详情
                post_details = parse_post_detail(driver, link)
                if post_details:
                    results.extend(post_details)
                human_like_sleep(0.8, 2.0)

            # 翻页（若不可点则退出）
            try:
                next_btn = wait.until(EC.element_to_be_clickable((By.LINK_TEXT, '下一页>')))
                next_btn.click()
            except Exception:
                logger.error("🚫 未能点击下一页或已到最后一页。")
                break

            # 低概率短暂停留，模拟人工行为
            human_like_sleep(1.5, 4.5)
            consecutive_failures = 0

        except Exception as e:
            logger.error(f"❌ 第 {page+1} 页异常: {e}")
            consecutive_failures += 1
            wait_time = backoff_base * (2 ** (consecutive_failures - 1))
            logger.error(f"⏳ 出错后等待 {wait_time} 秒重试（连续失败 {consecutive_failures} 次）")
            save_progress(keyword, {"page": page, "results": results})
            time.sleep(wait_time)
            if consecutive_failures >= 5:
                logger.error("✋ 多次失败，建议人工检查浏览器或稍后重试。")
                break

    return results

# ---------------- 保存 CSV ----------------
def save_to_csv(data: List[Dict[str, Any]], filename: str):
    if not data:
        logger.error("⚠️ 无数据，跳过保存。")
        return
    path = os.path.join(BASE_DIR, filename)
    with open(path, "w", newline='', encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0].keys()))
        writer.writeheader()
        writer.writerows(data)
    logger.info(f"✅ 已保存 {len(data)} 条记录到 {path}")

# ---------------- 主流程 ----------------
def main():
    all_keywords = load_keywords_from_json(KEYWORD_JSON, key="Chinese")
    logger.info(f"Loaded {len(all_keywords)} keywords.")
    try:
        login_with_cookie()
        for kw in all_keywords:
            logger.info(f"\n===== Processing keyword: {kw} =====")

            try:
            # 加载断点
                prog = load_progress(kw)
                start_page = prog["page"] if prog and "page" in prog else 0
                existing_results = prog.get("results", []) if prog else []

                # 抓取
                posts = crawl_tieba_search(kw, pages=8, start_page=start_page)

                # 给每条记录加上 keyword 字段
                for p in posts:
                    p["keyword"] = kw

                # 合并并保存
                all_results = existing_results + posts

                # --- 写入总 CSV ---
                if all_results:
                    header = list(all_results[0].keys())
                    append_to_big_csv(all_results, filename=OUTPUT_CSV, header=header)

                # --- 清理断点 ---
                safe_name = safe_filename(kw)
                prog_path = os.path.join(PROGRESS_DIR, f"{safe_name}_progress.pkl")
                if os.path.exists(prog_path):
                    os.remove(prog_path)
            except Exception as e:
                # 不影响后续关键词
                logger.error(f"[Error] keyword '{kw}' failed: {e}")
                continue
    finally:
        try:
            driver.quit()
        except Exception:
            pass

if __name__ == "__main__":
    main()
