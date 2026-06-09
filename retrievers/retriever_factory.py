"""
Retriever Factory — Creates retriever instances from configuration.

Uses the RETRIEVER_REGISTRY from config.py to dynamically import
and instantiate the correct retriever class.
"""

import importlib
import logging
from typing import Any

from retrievers.base_retriever import BaseRetriever

logger = logging.getLogger(__name__)


class RetrieverFactory:
    """Factory for creating retriever instances from registry configuration."""

    @staticmethod
    def create(retriever_name: str, **kwargs: Any) -> BaseRetriever:
        """
        Create a retriever instance by name.

        Args:
            retriever_name: Key from RETRIEVER_REGISTRY (e.g., 'llm_only').
            **kwargs: Additional keyword arguments passed to the constructor.

        Returns:
            An instance of a BaseRetriever subclass.

        Raises:
            ValueError: If retriever_name is not found in the registry.
        """
        from config import RETRIEVER_REGISTRY

        if retriever_name not in RETRIEVER_REGISTRY:
            available = ", ".join(RETRIEVER_REGISTRY.keys())
            raise ValueError(
                f"Unknown retriever '{retriever_name}'. Available: {available}"
            )

        entry = RETRIEVER_REGISTRY[retriever_name]
        module = importlib.import_module(entry["module"])
        cls = getattr(module, entry["class"])

        instance = cls(name=retriever_name, **kwargs)
        logger.info("Created retriever: %s (%s)", retriever_name, cls.__name__)
        return instance
