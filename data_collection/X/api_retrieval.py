from twarc import Twarc2, expansions
import datetime
import json, urllib
import re

# Replace your bearer token here
encoded_token = "AAAAAAAAAAAAAAAAAAAAACAx5AEAAAAAdhdK9656qDIvOmX7YcpyVKMST0Q%3Dok4cOR2uDgNm0yA58Z6zZrjrh4yyPcKBoQ91fZTRk5evVHARuV"

client = Twarc2(bearer_token=encoded_token)

def sanitize_filename(s):
    """去掉文件名中可能非法的字符"""
    return re.sub(r"[^\w\-_. ]", "_", s)

def search_and_save(keywords, lang="ja", start_time=None, end_time=None):
    """
    keywords: list of strings, 每个关键词单独抓取
    lang: string, 语言代码
    start_time, end_time: datetime.datetime 对象，UTC
    """
    for kw in keywords:
        # 拼 query
        query = f"{kw} lang:{lang}"

        # 文件名
        filename = f"{sanitize_filename(kw)}_{lang}.json"

        print(f"Searching for '{kw}' from {start_time} to {end_time}...")

        search_results = client.search_recent(query=query, start_time=start_time, end_time=end_time, max_results=100)

        # 流式写入文件
        with open(filename, "w", encoding="utf-8") as f:
            f.write("[")
            first = True
            for page in search_results:
                result = expansions.flatten(page)
                for tweet in result:
                    if not first:
                        f.write(",\n")
                    else:
                        first = False
                    json.dump(tweet, f, ensure_ascii=False)
            f.write("]")

        print(f"Saved tweets to {filename}")

if __name__ == "__main__":
    keywords = ["キリスト教"]  # 这里可以放多个关键词
    lang = "ja"
    # Specify the start time in UTC for the time period you want Tweets from
    start_time = datetime.datetime(2025, 10, 23, 0, 0, 0, 0, datetime.timezone.utc)

    # Specify the end time in UTC for the time period you want Tweets from
    end_time = datetime.datetime(2025, 10, 26, 0, 0, 0, 0, datetime.timezone.utc)
    search_and_save(keywords, lang, start_time, end_time)
