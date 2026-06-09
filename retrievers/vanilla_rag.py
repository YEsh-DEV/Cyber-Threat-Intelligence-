"""
Vanilla RAG Retriever

Uses local vector similarity search over the MITRE ATT&CK knowledge base
to retrieve relevant tactics, techniques, software, and threat groups.
"""

import logging
from typing import Any, List

from retrievers.base_retriever import BaseRetriever
from retrievers.vector_store import VectorStore

logger = logging.getLogger(__name__)


class VanillaRAGRetriever(BaseRetriever):
    """
    Vector-based RAG retriever utilizing a locally cached MITRE ATT&CK index.
    """

    def __init__(self, name: str = "vanilla_rag", **kwargs: Any) -> None:
        super().__init__(name, **kwargs)

        # Retrieve optional top_k config
        top_k = kwargs.get("top_k", 3)

        # Initialize vector store
        self.vector_store = VectorStore(top_k=top_k)
        
        # Build index if not present, otherwise load from cache
        logger.info("Initializing vector store index...")
        self.vector_store.initialize()

    def get_context(self, query_text: str) -> List[str]:
        """
        Retrieve relevant MITRE ATT&CK passages for the cybersecurity narrative.
        """
        logger.info("Retrieving ATT&CK context for narrative: %s...", query_text[:60].replace("\n", " "))
        
        try:
            hits = self.vector_store.search(query_text)
            
            context_passages = []
            for i, (doc, score) in enumerate(hits, 1):
                logger.debug("Hit %d: %s (score=%.3f)", i, doc["name"], score)
                context_passages.append(doc["text"])
                
            return context_passages
        except Exception as e:
            logger.error("RAG retrieval failed: %s", e)
            # Graceful fallback: return empty context so pipeline doesn't crash
            return []
