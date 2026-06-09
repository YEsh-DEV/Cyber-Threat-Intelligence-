"""
Gemini LLM Implementation

Wraps the Google Generative AI SDK for Gemini model access.

Implementation: Phase 7
"""

from typing import Any, Dict
from models.base_model import BaseLLM


class GeminiLLM(BaseLLM):
    """Google Gemini LLM implementation."""

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        super().__init__(model_name, **kwargs)
        # TODO: Phase 7 — Initialize Gemini client

    def generate_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        raise NotImplementedError("GeminiLLM — Phase 7")
