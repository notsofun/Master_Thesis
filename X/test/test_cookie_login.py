import os
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from main import load_cookies, is_logged_in  # 确认 main.py 中的 load_cookies 已改成兼容 key=value 文本

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COOKIES_FILE = os.path.join(BASE_DIR, "twitter_cookie.txt")
print(COOKIES_FILE)

def main():
    options = Options()
    options.add_argument("--start-maximized")
    options.add_argument("--disable-blink-features=AutomationControlled")

    driver = webdriver.Chrome(options=options)
    driver.get("https://twitter.com")
    time.sleep(5)  # 等页面完全加载

    print("[STEP 1] 页面加载完毕，尝试加载 cookies...")
    _ = load_cookies(driver, COOKIES_FILE)

    print("[STEP 2] 刷新页面以验证登录状态...")
    driver.refresh()
    time.sleep(5)

    if is_logged_in(driver):
        print("[✅ RESULT] cookies 加载成功，已自动登录 Twitter！")
    else:
        print("[❌ RESULT] cookies 无效或未登录，请重新登录并重新保存。")

    input("\n按 Enter 退出并关闭浏览器...")
    driver.quit()

if __name__ == "__main__":
    main()
