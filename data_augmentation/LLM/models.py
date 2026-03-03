import abc, json

# ==========================================
# 1. LLM 抽象层 (适配器模式)
# ==========================================

class BaseLLMClient(abc.ABC):
    """
    抽象基类：所有模型调用必须继承此类并实现 generate 方法。
    """
    @abc.abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        pass

class OpenAIClient(BaseLLMClient):
    """
    OpenAI 协议兼容客户端 (支持 GPT-4, DeepSeek-V3, Moonshot 等)
    """
    def __init__(self, api_key: str, base_url: str = None, model: str = "gpt-4o"):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.9, # 高创造性
                top_p=0.95
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return ""

class MockClient(BaseLLMClient):
    """
    用于测试流程的伪造客户端 (不需要花钱)
    """
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        # 模拟返回一个 JSON 列表
        mock_data = [
            {"text": "生成的模拟仇恨文本1", "language": "zh", "strategy": "隐喻", "dimension": "教义"},
            {"text": "生成的模拟仇恨文本2", "language": "ja", "strategy": "历史", "dimension": "政治"}
        ]
        return json.dumps(mock_data, ensure_ascii=False)
