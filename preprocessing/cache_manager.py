"""
Cache Manager

Handles caching for LLM responses and Retrieval contexts.
These caches save API quota and massive amounts of time during repeated
experiments or dev runs on the same dataset.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import CACHE_DIR

logger = logging.getLogger(__name__)

# Cache directories
LLM_CACHE_DIR = CACHE_DIR / "llm"
RETRIEVAL_CACHE_DIR = CACHE_DIR / "retrieval"

# Ensure directories exist
LLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
RETRIEVAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)


class CacheManager:
    """Manages reading and writing to various pipeline caches."""

    @staticmethod
    def _hash_prompt(model_id: str, system_prompt: str, user_prompt: str) -> str:
        """Create a stable SHA256 hash for a prompt + model combination."""
        hasher = hashlib.sha256()
        hasher.update(model_id.encode("utf-8"))
        hasher.update(system_prompt.encode("utf-8"))
        hasher.update(user_prompt.encode("utf-8"))
        return hasher.hexdigest()

    @staticmethod
    def get_llm_cache(model_id: str, system_prompt: str, user_prompt: str) -> Optional[Dict[str, Any]]:
        """
        Check if an exact prompt has been run with this model before.
        Returns the cached JSON response if found, else None.
        """
        prompt_hash = CacheManager._hash_prompt(model_id, system_prompt, user_prompt)
        cache_file = LLM_CACHE_DIR / f"{prompt_hash}.json"

        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # We return the response JSON
                    return data.get("response")
            except Exception as e:
                logger.warning("Corrupted LLM cache file %s: %s", cache_file.name, e)
        return None

    @staticmethod
    def set_llm_cache(model_id: str, system_prompt: str, user_prompt: str, response: Dict[str, Any]) -> None:
        """Save an LLM response to the cache."""
        prompt_hash = CacheManager._hash_prompt(model_id, system_prompt, user_prompt)
        cache_file = LLM_CACHE_DIR / f"{prompt_hash}.json"

        data = {
            "prompt_hash": prompt_hash,
            "model_id": model_id,
            # We don't save the full prompts to save space, just the length/metadata
            "prompt_lengths": {"system": len(system_prompt), "user": len(user_prompt)},
            "response": response,
        }

        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    @staticmethod
    def get_retrieval_cache(retriever_name: str, global_id: str) -> Optional[List[str]]:
        """
        Check if we've already retrieved context for this specific event with this retriever.
        """
        if not global_id:
            return None
            
        cache_file = RETRIEVAL_CACHE_DIR / f"{retriever_name}_{global_id}.json"

        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data.get("context", [])
            except Exception as e:
                logger.warning("Corrupted retrieval cache file %s: %s", cache_file.name, e)
        return None

    @staticmethod
    def set_retrieval_cache(retriever_name: str, global_id: str, context: List[str]) -> None:
        """Save retrieved context passages to the cache."""
        if not global_id:
            return
            
        cache_file = RETRIEVAL_CACHE_DIR / f"{retriever_name}_{global_id}.json"

        data = {
            "retriever_name": retriever_name,
            "global_id": global_id,
            "context": context,
        }

        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
