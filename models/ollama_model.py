"""
Ollama LLM Implementation

Communicates with a locally-running Ollama instance via its REST API.
Used for development with models like gemma_e2b:latest.

Implementation: Phase 4
"""

from typing import Any, Dict
from models.base_model import BaseLLM


class OllamaLLM(BaseLLM):
    """Ollama-hosted local LLM implementation."""

    def __init__(self, model_name: str, model_id: str = "", **kwargs: Any) -> None:
        super().__init__(model_name, **kwargs)
        self.model_id = model_id
        # TODO: Phase 4 — Initialize Ollama HTTP client

    def generate_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        raise NotImplementedError("OllamaLLM — Phase 4")
