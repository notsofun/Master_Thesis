# gemini_client.py
import json
import re
import time
import logging
from typing import Dict, Any
import os

logger = logging.getLogger(__name__)

# 下面使用你已有的 genai 客户端（根据你的示例）
try:
    from google import genai
except Exception:
    genai = None
    logger.warning("genai package not available. Ensure your environment has the Gemini SDK installed.")


class GeminiClient:
    def __init__(self, logger, api_key: str, model_name: str = "models/gemini-2.5-flash", max_retries: int = 3):
        if not api_key:
            raise ValueError("GEMINI API key must be provided via environment variable or argument.")
        if genai is None:
            raise RuntimeError("genai package not installed or importable.")
        self.client = genai.Client()
        self.logger = logger
        self.model_name = model_name
        self.max_retries = max_retries

    @staticmethod
    def _make_prompt(text: str) -> str:
        """
        严格要求模型只输出 JSON。输出格式：
        {"is_hate": true/false, "confidence": 0.0-1.0, "explanation": "简短理由"}
        """
        prompt = f"""
You are a content-moderation assistant. For the given TEXT below, answer whether it is HATE SPEECH.
Definition (use this to decide): Hate speech = content that attacks or dehumanizes a target based on protected characteristic (race, nationality, religion, gender, sexual orientation, disability, etc.), or encourages violence/harassment. Be conservative if ambiguous.

INSTRUCTIONS:
- Output ONLY a single JSON object and nothing else.
- Fields:
  - is_hate: true or false
  - confidence: number between 0 and 1
  - explanation: a short (1-2 sentence) justification (what target and why)
TEXT:
\"\"\"{text}\"\"\"
"""
        return prompt

    def classify(self, text: str) -> Dict[str, Any]:
        prompt = self._make_prompt(text)

        try:
            raw = self.ask_gemini_for_json(prompt)
            return self._normalize_response(raw)
        except Exception as e:
            self.logger.warning(f"Gemini classify attempt failed: {e} when classifying {prompt}")
            return {"is_hate": None, "confidence": None, "explanation": "failed"}

    def ask_gemini_for_json(self, prompt):
        """强制要求 Gemini 输出 JSON，不符合格式就重试"""
        for _ in range(self.max_retries):
            response = self.client.models.generate_content(
                model="models/gemini-2.5-flash",
                contents=prompt
            )
            text = response.text.strip()

            # 如果返回内容前后可能有 markdown code fence
            if text.startswith("```"):
                text = text.strip("`")
                text = text.replace("json", "", 1).strip()

            try:
                parsed = json.loads(text)
                return parsed
            except json.JSONDecodeError:
                time.sleep(1)  # 做一点退避，避免频繁请求

        self.logger.error(f"Gemini 多次输出无效 JSON，最后一次返回：\n{text}")
        raise ValueError(f"Gemini 多次输出无效 JSON")
    
    def _normalize_response(self, resp: dict) -> dict:
        def norm_bool(v):
            if isinstance(v, bool): return v
            if isinstance(v, str): return v.lower() in ("true", "yes", "1")
            return None

        def norm_float(v):
            try:
                return float(v)
            except Exception:
                return None

        return {
            "is_hate": norm_bool(resp.get("is_hate")),
            "confidence": norm_float(resp.get("confidence")),
            "explanation": resp.get("explanation", ""),
        }
