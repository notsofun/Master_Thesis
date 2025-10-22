import os
import time
import csv
import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager


# ========== 基本配置 ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COOKIE_FILE = os.path.join(BASE_DIR, "tieba_cookies.json")

options = Options()
options.add_argument("--start-maximized")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
wait = WebDriverWait(driver, 15)


# ========== 登录逻辑 ==========
def login_with_cookie():
    driver.get("https://tieba.baidu.com/")
    if os.path.exists(COOKIE_FILE):
        with open(COOKIE_FILE, "r", encoding="utf-8") as f:
            cookies = json.load(f)
        for cookie in cookies:
            driver.add_cookie(cookie)
        driver.refresh()
        print("✅ 已加载 cookie 登录")
    else:
        input("请手动扫码登录后按 Enter 继续...")
        cookies = driver.get_cookies()
        with open(COOKIE_FILE, "w", encoding="utf-8") as f:
            json.dump(cookies, f)
        print("✅ 登录成功，cookie 已保存")


# ========== 工具函数 ==========
def safe_switch_to_main(driver):
    """安全地切换回主窗口"""
    try:
        if driver.window_handles:
            driver.switch_to.window(driver.window_handles[0])
    except Exception as e:
        print(f"⚠️ 无法切换回主窗口: {e}")


# ========== 获取细节帖子数据 ==========
def parse_post_detail(driver, link):
    """打开帖子详情并抓取"""
    post_data = []
    try:
        driver.execute_script("window.open(arguments[0])", link)
        driver.switch_to.window(driver.window_handles[-1])
        time.sleep(3)

        title = driver.find_element(By.CSS_SELECTOR, "h3.core_title_txt").text
        all_posts = driver.find_elements(By.CSS_SELECTOR, "div.l_post")

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

            # 抓取楼中楼内容
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
        print(f"❌ 抓取帖子失败: {e}")

    finally:
        # 只在这里关闭当前标签页
        try:
            if len(driver.window_handles) > 1:
                driver.close()
        except Exception as e:
            print(f"⚠️ 关闭帖子标签页失败: {e}")
        safe_switch_to_main(driver)

    return post_data


# ========== 全贴搜索逻辑 ==========
def crawl_tieba_search(keyword, pages=2):
    print(f"🔍 开始搜索关键词：{keyword}")
    url = f"https://tieba.baidu.com/f/search/res?ie=utf-8&qw={keyword}"
    driver.get(url)
    time.sleep(3)

    results = []

    for page in range(pages):
        print(f"📄 正在抓取第 {page + 1} 页结果...")

        # 滚动加载更多结果
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)

        # 获取帖子链接
        posts = driver.find_elements(By.CSS_SELECTOR, "div.s_post")
        links = []
        for post in posts:
            try:
                link = post.find_element(By.CSS_SELECTOR, "a.bluelink").get_attribute("href")
                links.append(link)
            except:
                continue

        # 抓取每个帖子详情
        for link in links:
            post_details = parse_post_detail(driver, link)
            results.extend(post_details)

        # 翻页
        try:
            next_btn = wait.until(EC.element_to_be_clickable((By.LINK_TEXT, '下一页>')))
            next_btn.click()
            time.sleep(3)
        except:
            print("🚫 没有更多搜索结果页。")
            break

    return results


# ========== 保存为 CSV ==========
def save_to_csv(data, filename="tieba_search_posts.csv"):
    if not data:
        print("⚠️ 没有数据可保存。")
        return
    with open(filename, "w", newline='', encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
    print(f"✅ 已保存 {len(data)} 条结果到 {filename}")


# ========== 主程序入口 ==========
if __name__ == "__main__":
    login_with_cookie()
    posts = crawl_tieba_search("耶狗", pages=5)
    save_to_csv(posts)
    driver.quit()
