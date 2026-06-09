"""
Base Retriever Abstract Class

All retrieval strategies must inherit from this class and implement
the `get_context` method. This enables the pipeline to swap retrieval
strategies without code changes.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, List

logger = logging.getLogger(__name__)


class BaseRetriever(ABC):
    """Abstract base class for all retrieval strategies."""

    def __init__(self, name: str, **kwargs: Any) -> None:
        self.name = name
        self._kwargs = kwargs
        logger.info("Initialized retriever: %s", name)

    @abstractmethod
    def get_context(self, query_text: str) -> List[str]:
        """
        Retrieve relevant context for the given query.

        Args:
            query_text: The event narrative or query to retrieve context for.

        Returns:
            A list of relevant context strings. Empty list for LLM-only mode.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
