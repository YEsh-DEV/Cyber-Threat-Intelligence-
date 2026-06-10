"""
Groq LLM Implementation

Thin subclass of OpenAICompatibleLLM — Groq uses the standard
OpenAI-compatible chat completions API format.
"""

import logging
from typing import Any

from models.openai_compatible_model import OpenAICompatibleLLM

logger = logging.getLogger(__name__)


class GroqLLM(OpenAICompatibleLLM):
    """
    Groq LLM implementation.

    Uses the OpenAI-compatible API at https://api.groq.com/openai/v1/chat/completions.
    All retry logic, JSON extraction, and token tracking are inherited from
    OpenAICompatibleLLM.
    """

    GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(
        self,
        model_name: str,
        model_id: str = "llama-3.1-8b-instant",
        api_key: str = "",
        **kwargs: Any,
    ) -> None:
        from config import GROQ_API_KEY

        super().__init__(
            model_name=model_name,
            api_url=self.GROQ_API_URL,
            api_key=api_key or GROQ_API_KEY,
            model_id=model_id,
            **kwargs,
        )

        logger.info(
            "GroqLLM initialized: model_id=%s, url=%s",
            self.model_id,
            self.GROQ_API_URL,
        )
