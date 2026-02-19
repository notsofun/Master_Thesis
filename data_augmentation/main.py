import json
from pipeline import AugmentationPipeline
from vector_engine import VectorEngine
from models import BaseLLMClient, OpenAIClient, MockClient

# ==========================================
# 4. 执行入口 (Usage)
# ==========================================

if __name__ == "__main__":
    # 1. 准备你的种子数据 (真实场景请从文件读取)
    seeds = [
        "邪恶的蛇把禁果给了亚当和夏娃，而在中国大家互送苹果，说明中国淫民不信邪。",
        "キリスト教は愛を説くが、歴史を見れば十字軍や植民地支配など、血塗られた歴史しかない。", # 基督教讲爱，但看历史全是十字军和殖民
        "那些天天喊着主啊主啊的人，遇到事还不是要去医院求医生，虚伪的信徒。",
        "日本の伝統を否定して、西洋の神を崇めるのは精神的な侵略だ。", # 否定日本传统崇拜西方神是精神侵略
        "吃着祖宗的饭，砸着祖宗的碗，去信一个中东的木匠儿子，可笑。",
    ]

    # 2. 初始化向量引擎 (这步比较慢，因为要下载/加载模型)
    # 如果没有 GPU，这可能会花几秒钟
    ve = VectorEngine() 

    # 3. 选择并初始化 LLM 客户端
    # 选项 A: 使用 OpenAI (需要 API KEY)
    # llm = OpenAIClient(api_key="sk-xxxx", base_url="https://api.openai.com/v1")
    
    # 选项 B: 使用 DeepSeek / Moonshot (通过 OpenAI 兼容协议)
    # llm = OpenAIClient(api_key="sk-xxxx", base_url="https://api.deepseek.com", model="deepseek-chat")

    # 选项 C: 使用 Mock 客户端 (测试代码逻辑用)
    llm = MockClient()

    # 4. 运行管道
    pipeline = AugmentationPipeline(llm_client=llm, vector_engine=ve)
    final_data = pipeline.run(seed_data=seeds, iterations=2) # 跑2轮测试

    # 5. 保存结果
    with open("synthetic_hate_speech.json", "w", encoding="utf-8") as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)

    print(f"\n全部完成！共生成 {len(final_data)} 条有效数据，已保存至 synthetic_hate_speech.json")