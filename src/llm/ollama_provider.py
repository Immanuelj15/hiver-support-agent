"""
src/llm/ollama_provider.py

Local offline LLM provider using Ollama HTTP API (http://localhost:11434).
Requires zero API keys and supports models such as mistral, llama3, phi3, etc.
"""

import json
from typing import Optional, Dict, Any
import requests

from src.llm.base import LLMProvider

class OllamaProvider(LLMProvider):
    def __init__(
        self,
        model: str = "mistral:latest",
        base_url: str = "http://localhost:11434"
    ):
        self._model = model
        # Strip trailing /v1 if present for native /api endpoint
        self._base_url = base_url.rstrip("/").removesuffix("/v1")

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def provider_type(self) -> str:
        return "ollama"

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 512,
        **kwargs
    ) -> str:
        """Call Ollama /api/generate endpoint."""
        url = f"{self._base_url}/api/generate"
        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        if system_instruction:
            payload["system"] = system_instruction

        try:
            resp = requests.post(url, json=payload, timeout=90)
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "").strip()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(
                f"Failed to connect to local Ollama daemon at {url}. "
                f"Ensure Ollama is running (`ollama serve`) and model '{self._model}' is installed (`ollama pull {self._model}`). "
                f"Error: {e}"
            )
