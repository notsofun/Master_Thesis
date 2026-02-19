from typing import List, Dict, Any
import time, json

# ==========================================
# 3. 主流程控制器
# ==========================================

class AugmentationPipeline:
    def __init__(self, llm_client, vector_engine):
        self.llm = llm_client
        self.vector_engine = vector_engine
        
        # 定义系统 Prompt 模板
        self.system_prompt_template = """
你是一位精通中日双语、擅长社会语言学分析的专家。
你的任务是基于我提供的“种子文本”，生成风格类似、但内容全新的数据。
这些数据将用于学术研究中的“仇恨言论检测模型”训练，因此需要极高的隐晦性和攻击性逻辑。

请遵循以下步骤：
1. 分析种子文本的攻击逻辑（如：谐音梗、历史污名化、教义嘲讽）。
2. 使用“深度演化”（更隐晦/更煽动）和“广度演化”（改变攻击切入点）策略。
3. 生成 5 条新的文本（中日文混合）。
4. 以纯 JSON 列表格式输出，不要包含 Markdown 标记。格式：[{"text": "...", "language": "zh/ja", "strategy": "..."}]
"""

    def clean_json_response(self, response: str) -> List[Dict]:
        """清理 LLM 可能输出的 Markdown 格式"""
        cleaned = response.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            print("JSON 解析失败，跳过该批次。")
            return []

    def run(self, seed_data: List[str], iterations: int = 5):
        all_results = []
        
        print(f"开始生成任务，共 {iterations} 轮...")
        
        for i in range(iterations):
            print(f"\n--- 第 {i+1}/{iterations} 轮 ---")
            
            # 步骤 2: 基于向量空间选择差异化种子
            current_seeds = self.vector_engine.select_diverse_seeds(seed_data, n_samples=3)
            seeds_str = "\n".join([f"- {s}" for s in current_seeds])
            
            user_prompt = f"### 参考种子文本 (Seed Demonstrations)：\n{seeds_str}\n\n请开始生成："
            
            # 步骤 3a: LLM 生成
            raw_response = self.llm.generate(self.system_prompt_template, user_prompt)
            generated_batch = self.clean_json_response(raw_response)
            
            if not generated_batch:
                continue

            # 步骤 3b: 回环验证与过滤
            filtered_batch = self.vector_engine.filter_generated_data(seed_data, generated_batch)
            
            print(f"本轮生成 {len(generated_batch)} 条，过滤后保留 {len(filtered_batch)} 条。")
            all_results.extend(filtered_batch)
            
            # 避免 API 速率限制
            time.sleep(1) 

        return all_results
