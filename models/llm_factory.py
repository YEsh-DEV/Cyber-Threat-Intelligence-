"""
LLM Factory — Creates model instances from configuration.

Uses the MODEL_REGISTRY from config.py to dynamically import
and instantiate the correct LLM class. Pipeline code never
needs to import specific model classes directly.
"""

import importlib
import logging
from typing import Optional

from models.base_model import BaseLLM

logger = logging.getLogger(__name__)


class LLMFactory:
    """Factory for creating LLM instances from registry configuration."""

    @staticmethod
    def create(model_name: str, **kwargs) -> BaseLLM:
        """
        Create an LLM instance by name.

        Args:
            model_name: Key from MODEL_REGISTRY (e.g., 'gemini', 'ollama_gemma').
            **kwargs: Additional keyword arguments passed to the model constructor.

        Returns:
            An instance of a BaseLLM subclass.

        Raises:
            ValueError: If model_name is not found in the registry.
            ImportError: If the model module cannot be imported.
        """
        from config import MODEL_REGISTRY

        if model_name not in MODEL_REGISTRY:
            available = ", ".join(MODEL_REGISTRY.keys())
            raise ValueError(
                f"Unknown model '{model_name}'. Available: {available}"
            )

        entry = MODEL_REGISTRY[model_name]
        module = importlib.import_module(entry["module"])
        cls = getattr(module, entry["class"])

        # Merge registry config with explicit kwargs
        init_kwargs = {k: v for k, v in entry.items() if k not in ("class", "module")}
        init_kwargs.update(kwargs)

        instance = cls(model_name=model_name, **init_kwargs)
        logger.info("Created LLM: %s (%s)", model_name, cls.__name__)
        return instance
