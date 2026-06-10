"""
Mistral LLM Implementation

Thin subclass of OpenAICompatibleLLM — Mistral uses the standard
OpenAI-compatible chat completions API format.
"""

import logging
from typing import Any

from models.openai_compatible_model import OpenAICompatibleLLM

logger = logging.getLogger(__name__)


class MistralLLM(OpenAICompatibleLLM):
    """
    Mistral AI LLM implementation.

    Uses the OpenAI-compatible API at https://api.mistral.ai/v1/chat/completions.
    All retry logic, JSON extraction, and token tracking are inherited from
    OpenAICompatibleLLM.
    """

    MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"

    def __init__(
        self,
        model_name: str,
        model_id: str = "open-mistral-7b",
        api_key: str = "",
        **kwargs: Any,
    ) -> None:
        from config import MISTRAL_API_KEY

        super().__init__(
            model_name=model_name,
            api_url=self.MISTRAL_API_URL,
            api_key=api_key or MISTRAL_API_KEY,
            model_id=model_id,
            **kwargs,
        )

        logger.info(
            "MistralLLM initialized: model_id=%s, url=%s",
            self.model_id,
            self.MISTRAL_API_URL,
        )
