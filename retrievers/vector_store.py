"""
Vector Store for MITRE ATT&CK Knowledge Base

Downloads the STIX 2.0 representation of MITRE ATT&CK, parses techniques,
software, and groups, and loads them into a persistent ChromaDB instance
using local HuggingFace embeddings.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from chromadb.utils import embedding_functions
import requests

logger = logging.getLogger(__name__)


class VectorStore:
    """Persistent vector store using ChromaDB and local embeddings."""

    MITRE_URL = "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    COLLECTION_NAME = "mitre_attack"

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        embedding_model: str = MODEL_NAME,
        top_k: int = 3,
    ) -> None:
        from config import PROJECT_ROOT

        self.data_dir = data_dir or (PROJECT_ROOT / "cache")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.stix_path = self.data_dir / "enterprise-attack.json"
        
        # New ChromaDB persistence directory
        self.chroma_dir = self.data_dir / "chroma_db"
        
        self.embedding_model = embedding_model
        self.top_k = top_k

        # Initialize Chroma client
        self.client = chromadb.PersistentClient(path=str(self.chroma_dir))
        
        # Initialize default embedding function (uses local ONNX runtime under the hood by default in Chroma)
        # We specify the model to ensure parity with the previous implementation
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.embedding_model
        )
        
        self.collection = None

    def initialize(self) -> None:
        """Initialize the vector store: create collection, and ingest if empty."""
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            embedding_function=self.embedding_fn,
            metadata={"description": "MITRE ATT&CK Knowledge Base"}
        )

        # Check if we need to ingest data
        if self.collection.count() == 0:
            logger.info("Chroma collection is empty. Initializing database...")
            self.download_stix()
            docs = self.parse_stix()
            self.build_index(docs)
        else:
            logger.info("Chroma DB loaded from cache. Index contains %d documents.", self.collection.count())

    def download_stix(self) -> None:
        """Download the enterprise attack STIX 2.0 dataset from MITRE."""
        if self.stix_path.exists():
            logger.info("MITRE ATT&CK dataset found locally: %s", self.stix_path)
            return

        # Fallback: check data/ directory if STIX was downloaded there previously
        from config import PROJECT_ROOT
        fallback_path = PROJECT_ROOT / "data" / "enterprise-attack.json"
        if fallback_path.exists() and fallback_path != self.stix_path:
            logger.info("STIX file not found at expected path: %s. Checking data/ directory...", self.stix_path)
            import shutil
            shutil.copy2(fallback_path, self.stix_path)
            logger.info("Copied STIX file from %s to %s", fallback_path, self.stix_path)
            return

        logger.info("Downloading MITRE ATT&CK dataset from %s...", self.MITRE_URL)
        try:
            # Set verify=False to ignore SSL errors from decrypting firewalls
            response = requests.get(self.MITRE_URL, timeout=60, verify=False)
            response.raise_for_status()
            with open(self.stix_path, "w", encoding="utf-8") as f:
                f.write(response.text)
            logger.info("Successfully downloaded and saved MITRE ATT&CK dataset.")
        except Exception as e:
            logger.error("Failed to download MITRE ATT&CK dataset: %s", e)
            raise ConnectionError(f"Could not fetch MITRE ATT&CK knowledge base: {e}") from e

    def parse_stix(self) -> List[Dict[str, Any]]:
        """Parse the STIX JSON file and extract objects."""
        logger.info("Parsing STIX dataset...")
        with open(self.stix_path, "r", encoding="utf-8") as f:
            bundle = json.load(f)

        objects = bundle.get("objects", [])
        parsed_docs = []

        for obj in objects:
            # Skip revoked or deprecated objects
            if obj.get("revoked", False) or obj.get("x_mitre_deprecated", False):
                continue

            obj_type = obj.get("type")
            if obj_type not in ("attack-pattern", "malware", "tool", "intrusion-set"):
                continue

            name = obj.get("name", "")
            description = obj.get("description", "")
            stix_id = obj.get("id", "")

            # Get external ID (e.g. T1059, S0002, G0007)
            external_id = "unknown"
            for ref in obj.get("external_references", []):
                if ref.get("source_name") in ("mitre-attack", "mitre-enterprise-attack"):
                    external_id = ref.get("external_id", "unknown")
                    break

            if not name or not description:
                continue

            # Format human-readable type
            readable_type = {
                "attack-pattern": "Technique",
                "malware": "Malware",
                "tool": "Tool",
                "intrusion-set": "Threat Actor Group",
            }.get(obj_type, obj_type)

            # Construct clean snippet for RAG injection
            text_snippet = (
                f"MITRE ATT&CK {readable_type}: {name} ({external_id})\n"
                f"Description: {description.strip()}"
            )

            # Limit text snippet length to avoid overwhelming LLM context window
            if len(text_snippet) > 1200:
                text_snippet = text_snippet[:1197] + "..."

            parsed_docs.append({
                "id": stix_id,
                "external_id": external_id,
                "name": name,
                "type": readable_type,
                "text": text_snippet,
            })

        logger.info("Extracted %d objects from STIX bundle.", len(parsed_docs))
        return parsed_docs

    def build_index(self, documents: List[Dict[str, Any]]) -> None:
        """Insert documents into ChromaDB collection."""
        if not documents:
            raise ValueError("No documents to index.")

        logger.info("Ingesting %d ATT&CK objects into ChromaDB. This may take a moment...", len(documents))

        ids = [doc["id"] for doc in documents]
        texts = [doc["text"] for doc in documents]
        metadatas = [
            {
                "external_id": doc["external_id"],
                "name": doc["name"],
                "type": doc["type"],
            }
            for doc in documents
        ]

        # Chroma handles batching internally, but we'll do it explicitly for progress logging
        batch_size = 128
        start_time = time.time()
        
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i : i + batch_size]
            batch_texts = texts[i : i + batch_size]
            batch_metadatas = metadatas[i : i + batch_size]
            
            logger.info("Ingesting batch %d/%d...", (i // batch_size) + 1, (len(ids) // batch_size) + 1)
            
            self.collection.add(
                ids=batch_ids,
                documents=batch_texts,
                metadatas=batch_metadatas
            )

        elapsed = time.time() - start_time
        logger.info("Ingested documents into ChromaDB in %.1fs", elapsed)

    def search(self, query: str, top_k: Optional[int] = None) -> List[Tuple[Dict[str, Any], float]]:
        """
        Perform semantic search for the query using ChromaDB.

        Returns:
            List of (document_dict, distance_score) sorted by closest distance.
        """
        if self.collection is None:
            raise ValueError("Vector store not initialized. Call initialize() first.")

        k = top_k or self.top_k
        
        results = self.collection.query(
            query_texts=[query],
            n_results=k
        )
        
        if not results['documents'][0]:
            return []

        # Reconstruct the original document dictionary structure for compatibility
        out_results = []
        for i in range(len(results['documents'][0])):
            doc_id = results['ids'][0][i]
            text = results['documents'][0][i]
            metadata = results['metadatas'][0][i]
            distance = results['distances'][0][i]
            
            doc_dict = {
                "id": doc_id,
                "text": text,
                "external_id": metadata["external_id"],
                "name": metadata["name"],
                "type": metadata["type"]
            }
            out_results.append((doc_dict, float(distance)))

        return out_results
