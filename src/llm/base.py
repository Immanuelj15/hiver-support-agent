"""
src/llm/base.py

Abstract base class for LLM providers.
Supports offline local execution (Ollama) and cloud APIs (Gemini, OpenAI).
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class LLMProvider(ABC):
    """Abstract interface for large language model inference."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 512,
        **kwargs
    ) -> str:
        """
        Generate completion for a given prompt string.
        Returns the text response.
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Returns the current model identifier."""
        pass

    @property
    @abstractmethod
    def provider_type(self) -> str:
        """Returns 'ollama' or 'cloud'."""
        pass
