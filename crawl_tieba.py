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

# ========== 初始化部分 ==========
options = Options()
options.add_argument("--start-maximized")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
wait = WebDriverWait(driver, 15)
COOKIE_FILE = "tieba_cookies.json"

# ========== 登录逻辑 ==========
def login_with_cookie():
    driver.get("https://tieba.baidu.com/")
    if os.path.exists(COOKIE_FILE):
        # 加载 cookie
        with open(COOKIE_FILE, "r", encoding="utf-8") as f:
            cookies = json.load(f)
        for cookie in cookies:
            driver.add_cookie(cookie)
        driver.refresh()
        print("✅ 已加载本地 cookie 登录")
    else:
        input("请手动扫码登录后按 Enter 继续...")
        cookies = driver.get_cookies()
        with open(COOKIE_FILE, "w", encoding="utf-8") as f:
            json.dump(cookies, f)
        print("✅ 登录成功，cookie 已保存")

# ========== 爬取逻辑 ==========
def crawl_tieba(keyword, pages=1):
    print(f"开始搜索：{keyword}")
    driver.get(f"https://tieba.baidu.com/f?kw={keyword}")
    time.sleep(3)

    results = []

    for page in range(pages):
        print(f"📄 正在爬取第 {page + 1} 页...")
        threads = driver.find_elements(By.CSS_SELECTOR, 'a.j_th_tit')
        links = [t.get_attribute('href') for t in threads]

        for link in links:
            try:
                driver.execute_script("window.open(arguments[0])", link)
                driver.switch_to.window(driver.window_handles[-1])
                time.sleep(3)

                title = driver.find_element(By.CSS_SELECTOR, 'h3.core_title_txt').text
                author = driver.find_element(By.CSS_SELECTOR, 'li.d_name a').text
                post_time = driver.find_element(By.CSS_SELECTOR, 'div.post-tail-wrap span.tail-info').text
                content = driver.find_element(By.CSS_SELECTOR, 'div.d_post_content').text

                # 收集回复（示例只取前几条）
                replies = driver.find_elements(By.CSS_SELECTOR, 'div.l_post')
                reply_texts = []
                for r in replies[1:5]:  # 取前4条回复
                    try:
                        reply_texts.append(r.find_element(By.CSS_SELECTOR, 'div.d_post_content').text)
                    except:
                        continue

                results.append({
                    "title": title,
                    "author": author,
                    "time": post_time,
                    "content": content,
                    "replies": " | ".join(reply_texts)
                })

                driver.close()
                driver.switch_to.window(driver.window_handles[0])

            except Exception as e:
                print(f"❌ 处理帖子失败: {e}")
                driver.close()
                driver.switch_to.window(driver.window_handles[0])

        # 下一页
        try:
            next_btn = wait.until(EC.element_to_be_clickable((By.LINK_TEXT, '下一页>')))
            next_btn.click()
            time.sleep(3)
        except:
            print("没有更多页面。")
            break

    return results

# ========== 保存 CSV ==========
def save_to_csv(data, filename="tieba_posts.csv"):
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
    posts = crawl_tieba("耶狗", pages=2)
    save_to_csv(posts)
    driver.quit()
