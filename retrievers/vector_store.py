"""
Vector Store for MITRE ATT&CK Knowledge Base

Downloads the STIX 2.0 representation of MITRE ATT&CK, parses techniques,
software, and groups, embeds them using a local HuggingFace model, and performs
NumPy-based cosine similarity search.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


class VectorStore:
    """Lightweight vector store using local embeddings and NumPy cosine similarity."""

    MITRE_URL = "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        embedding_model: str = MODEL_NAME,
        top_k: int = 3,
    ) -> None:
        from config import PROJECT_ROOT

        self.data_dir = data_dir or (PROJECT_ROOT / "data")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.stix_path = self.data_dir / "enterprise-attack.json"
        self.index_path = self.data_dir / "vector_index.npz"
        self.embedding_model = embedding_model
        self.top_k = top_k

        # Document and embedding lists
        self.documents: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None

        # Lazy load tokenizer and model
        self._tokenizer = None
        self._model = None

    def initialize(self) -> None:
        """Initialize the vector store: download, parse, and embed if cache not present."""
        if self.index_path.exists():
            self.load_index()
        else:
            logger.info("Vector index not found. Initializing database...")
            self.download_stix()
            self.parse_stix()
            self.build_index()

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

    def parse_stix(self) -> None:
        """Parse the STIX JSON file and extract techniques, malware, tools, and groups."""
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

        self.documents = parsed_docs
        logger.info("Extracted %d objects from STIX bundle.", len(self.documents))

    def _get_model(self) -> Tuple[AutoTokenizer, AutoModel]:
        """Lazy loader for HuggingFace embedding model and tokenizer."""
        if self._tokenizer is None or self._model is None:
            logger.info("Loading embedding model '%s'...", self.embedding_model)
            # Handle SSL warnings/errors for models if corporate network intercepts TLS
            # We trust HuggingFace hub cached download or try system certs
            self._tokenizer = AutoTokenizer.from_pretrained(self.embedding_model)
            self._model = AutoModel.from_pretrained(self.embedding_model)
            # Run model on CPU
            self._model.eval()

        return self._tokenizer, self._model

    def _mean_pooling(self, model_output: Any, attention_mask: torch.Tensor) -> torch.Tensor:
        """Mean Pooling - Take attention mask into account for correct averaging."""
        token_embeddings = model_output[0]  # First element contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate L2-normalized embeddings for a list of texts using local PyTorch.
        """
        tokenizer, model = self._get_model()

        # Tokenize inputs
        encoded_input = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        # Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)

        # Perform pooling and normalize
        sentence_embeddings = self._mean_pooling(model_output, encoded_input["attention_mask"])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        return sentence_embeddings.cpu().numpy()

    def build_index(self) -> None:
        """Generate embeddings for all parsed documents and save to cache file."""
        if not self.documents:
            raise ValueError("No documents to index. Run parse_stix() first.")

        logger.info("Generating embeddings for %d ATT&CK objects. This may take a moment...", len(self.documents))
        texts = [doc["text"] for doc in self.documents]

        # Process in batches to save memory
        batch_size = 128
        embeddings_list = []

        start_time = time.time()
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            logger.info("Embedding batch %d/%d...", (i // batch_size) + 1, (len(texts) // batch_size) + 1)
            batch_embeds = self.get_embeddings(batch_texts)
            embeddings_list.append(batch_embeds)

        self.embeddings = np.vstack(embeddings_list)
        elapsed = time.time() - start_time
        logger.info("Generated embeddings in %.1fs", elapsed)

        # Save index file
        self.save_index()

    def save_index(self) -> None:
        """Save documents and embedding matrix to npz file."""
        if self.embeddings is None:
            return

        # Convert list of dicts to JSON string to save inside NPZ
        docs_json = json.dumps(self.documents, ensure_ascii=False)

        np.savez_compressed(
            self.index_path,
            embeddings=self.embeddings,
            documents=np.array([docs_json], dtype=object)
        )
        logger.info("Saved vector index to %s", self.index_path)

    def load_index(self) -> None:
        """Load documents and embedding matrix from cached npz file."""
        logger.info("Loading vector index from cache: %s", self.index_path)
        data = np.load(self.index_path, allow_pickle=True)
        self.embeddings = data["embeddings"]
        docs_json = str(data["documents"][0])
        self.documents = json.loads(docs_json)
        logger.info("Loaded %d documents from index.", len(self.documents))

    def search(self, query: str, top_k: Optional[int] = None) -> List[Tuple[Dict[str, Any], float]]:
        """
        Perform cosine similarity search for the query against the document database.

        Returns:
            List of (document, similarity_score) sorted by descending similarity.
        """
        if self.embeddings is None:
            raise ValueError("Vector store not initialized. Call initialize() first.")

        k = top_k or self.top_k
        query_vector = self.get_embeddings([query])  # Shape: (1, dim)

        # Compute cosine similarity: dot product of normalized query and database vectors
        # Shape: (num_docs,)
        similarities = np.dot(self.embeddings, query_vector[0])

        # Get top-k indices sorted descending
        top_indices = np.argsort(similarities)[::-1][:k]

        results = []
        for idx in top_indices:
            results.append((self.documents[idx], float(similarities[idx])))

        return results
