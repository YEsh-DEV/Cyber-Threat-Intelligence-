"""
Base LLM Abstract Class

All LLM implementations must inherit from this class and implement
the `generate_json` method. This ensures the pipeline never contains
model-specific logic.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class BaseLLM(ABC):
    """Abstract base class for all LLM implementations."""

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        self.model_name = model_name
        self._kwargs = kwargs
        logger.info("Initialized LLM: %s", model_name)

    @abstractmethod
    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> Dict[str, Any]:
        """
        Generate a structured JSON response from the LLM.

        Args:
            system_prompt: System-level instruction for the model.
            user_prompt: The user query / event narrative.

        Returns:
            Parsed JSON dictionary from the model's response.

        Raises:
            ValueError: If the model response cannot be parsed as JSON.
            ConnectionError: If the model API is unreachable.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_name={self.model_name!r})"
