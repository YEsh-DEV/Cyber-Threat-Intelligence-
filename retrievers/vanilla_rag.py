"""
Vanilla RAG Retriever

Uses vector similarity search (ChromaDB or FAISS) over a
MITRE ATT&CK knowledge base to retrieve relevant context passages.

Implementation: Phase 8
"""

from typing import Any, List
from retrievers.base_retriever import BaseRetriever


class VanillaRAGRetriever(BaseRetriever):
    """Vector-based RAG retriever using MITRE ATT&CK knowledge base."""

    def __init__(self, name: str = "vanilla_rag", **kwargs: Any) -> None:
        super().__init__(name, **kwargs)
        # TODO: Phase 8 — Initialize vector store and embeddings

    def get_context(self, query_text: str) -> List[str]:
        raise NotImplementedError("VanillaRAGRetriever — Phase 8")
