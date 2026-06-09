"""
LLM-Only Retriever

Returns no external context — the LLM works solely from its
pre-trained knowledge and the provided event narrative.
"""

from typing import Any, List
from retrievers.base_retriever import BaseRetriever


class LLMOnlyRetriever(BaseRetriever):
    """No-op retriever — returns empty context."""

    def __init__(self, name: str = "llm_only", **kwargs: Any) -> None:
        super().__init__(name, **kwargs)

    def get_context(self, query_text: str) -> List[str]:
        """Return empty context list."""
        return []
