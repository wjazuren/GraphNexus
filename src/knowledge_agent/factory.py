import os
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


class OpenAICompatibleEngine:
    """Lightweight chat engine for Agent runs.

    It intentionally avoids importing OneKE's transformer-based model package,
    which otherwise loads Torch/Transformers even for remote API models.
    """

    def __init__(self, model: str, api_key: str, base_url: Optional[str] = None):
        self.name = model
        self.model = model
        self.temperature = 0.2
        self.top_p = 0.9
        self.max_tokens = int(os.getenv("LLM_MAX_TOKENS", "4096"))
        self.client = OpenAI(api_key=api_key, base_url=base_url or None)

    def set_hyperparameter(
        self,
        temperature: float = 0.2,
        top_p: float = 0.9,
        max_tokens: int = 4096,
    ) -> None:
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

    def get_chat_response(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content or ""


def build_llm_from_env() -> OpenAICompatibleEngine:
    """Build a remote or vLLM engine without putting secrets in source."""
    provider = os.getenv("LLM_PROVIDER", "openai_compatible").lower()
    model = os.getenv("LLM_MODEL", "")
    base_url = os.getenv("LLM_BASE_URL", "")
    api_key = os.getenv("LLM_API_KEY", "")
    if not model:
        raise RuntimeError("LLM_MODEL is not configured")
    if provider == "vllm":
        return OpenAICompatibleEngine(
            model=model,
            api_key=api_key or "EMPTY_API_KEY",
            base_url=base_url or "http://localhost:8000/v1",
        )
    if not api_key:
        raise RuntimeError("LLM_API_KEY is not configured")
    if provider == "deepseek" and not base_url:
        base_url = "https://api.deepseek.com"
    return OpenAICompatibleEngine(model=model, api_key=api_key, base_url=base_url or None)
