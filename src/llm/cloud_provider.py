"""
src/llm/cloud_provider.py

Cloud LLM provider implementation supporting Google Gemini and OpenAI models
via the official OpenAI Python client.
"""

import os
from typing import Optional
from openai import OpenAI

from src.llm.base import LLMProvider

class CloudLLMProvider(LLMProvider):
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        gemini_key = api_key or os.environ.get("GEMINI_API_KEY")
        openai_key = api_key or os.environ.get("OPENAI_API_KEY")

        if gemini_key:
            self._model = model or os.environ.get("CLOUD_MODEL", "gemini-3.6-flash")
            self._base_url = base_url or "https://generativelanguage.googleapis.com/v1beta/openai/"
            self._client = OpenAI(api_key=gemini_key, base_url=self._base_url)
        elif openai_key:
            self._model = model or os.environ.get("CLOUD_MODEL", "gpt-4o-mini")
            self._base_url = base_url
            self._client = OpenAI(api_key=openai_key, base_url=self._base_url)
        else:
            raise ValueError(
                "Neither GEMINI_API_KEY nor OPENAI_API_KEY is set in the environment. "
                "Please configure an API key or use LLM_PROVIDER=ollama for offline local inference."
            )

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def provider_type(self) -> str:
        return "cloud"

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1000,
        **kwargs
    ) -> str:
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        import time
        max_retries = 3
        backoff = 2.0

        for attempt in range(max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                content = resp.choices[0].message.content
                return (content or "").strip()
            except Exception as e:
                err_str = str(e).lower()
                if ("rate" in err_str or "429" in err_str or "quota" in err_str) and attempt < max_retries - 1:
                    time.sleep(backoff)
                    backoff *= 2.0
                else:
                    raise e

