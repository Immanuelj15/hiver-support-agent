"""
src/llm/factory.py

Factory to instantiate LLM providers based on environment variables or explicit parameters.
Supports seamless switching between offline Ollama, Groq, and Cloud (Gemini/OpenAI).
"""

import os
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.llm.base import LLMProvider
from src.llm.ollama_provider import OllamaProvider
from src.llm.cloud_provider import CloudLLMProvider

def get_llm_provider(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs
) -> LLMProvider:
    """
    Returns an instance of LLMProvider.
    Checks parameters first, then environment variables (LLM_PROVIDER, GROQ_API_KEY, etc.).
    """
    resolved_provider = (provider or os.environ.get("LLM_PROVIDER", "")).lower().strip()

    # If not explicitly specified, choose based on available keys or defaults
    if not resolved_provider:
        if os.environ.get("GROQ_API_KEY"):
            resolved_provider = "groq"
        elif os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY"):
            resolved_provider = "cloud"
        else:
            resolved_provider = "ollama"

    if resolved_provider == "ollama":
        target_model = model or os.environ.get("OLLAMA_MODEL", "mistral:latest")
        base_url = kwargs.get("base_url") or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        return OllamaProvider(model=target_model, base_url=base_url)

    elif resolved_provider in ("cloud", "gemini", "openai"):
        target_model = model or os.environ.get("CLOUD_MODEL", "gemini-3.6-flash")
        return CloudLLMProvider(model=target_model, provider_name=resolved_provider, **kwargs)

    elif resolved_provider == "groq":
        target_model = model or os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")
        return CloudLLMProvider(model=target_model, provider_name="groq", **kwargs)

    else:
        raise ValueError(f"Unknown LLM provider '{resolved_provider}'. Supported: 'ollama', 'cloud', 'groq'.")

