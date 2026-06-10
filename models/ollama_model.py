"""
Ollama LLM Implementation

Communicates with a locally-running Ollama instance via its REST API.
Used for development with models like gemma_e2b:latest.

Features:
  - JSON-structured output generation
  - Automatic retry with exponential backoff
  - Timeout protection
  - Response parsing and repair
  - Configurable model selection
"""

import json
import logging
import re
import time
from typing import Any, Dict, Optional

import requests

from models.base_model import BaseLLM

logger = logging.getLogger(__name__)


class OllamaLLM(BaseLLM):
    """
    Ollama-hosted local LLM implementation.

    Connects to a running Ollama server via its HTTP API to generate
    structured JSON responses for CTI extraction.
    """

    def __init__(
        self,
        model_name: str,
        model_id: str = "",
        base_url: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name, **kwargs)

        from config import OLLAMA_BASE_URL, OLLAMA_MODEL_NAME, MAX_RETRIES, RETRY_BASE_DELAY, REQUEST_TIMEOUT, TEMPERATURE

        self.model_id = model_id or OLLAMA_MODEL_NAME
        self.base_url = (base_url or OLLAMA_BASE_URL).rstrip("/")
        self.max_retries = MAX_RETRIES
        self.retry_base_delay = RETRY_BASE_DELAY
        self.request_timeout = REQUEST_TIMEOUT
        self.temperature = TEMPERATURE

        # Ollama API endpoints
        self._generate_url = f"{self.base_url}/api/generate"
        self._chat_url = f"{self.base_url}/api/chat"
        self._tags_url = f"{self.base_url}/api/tags"

        logger.info(
            "OllamaLLM initialized: model_id=%s, base_url=%s",
            self.model_id,
            self.base_url,
        )

    def is_available(self) -> bool:
        """
        Check if the Ollama server is reachable and the model is loaded.

        Returns:
            True if server is up and model is available.
        """
        try:
            response = requests.get(self._tags_url, timeout=10)
            if response.status_code != 200:
                logger.warning("Ollama server returned status %d", response.status_code)
                return False

            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            available = self.model_id in model_names

            if not available:
                # Also check without :latest suffix
                base_name = self.model_id.split(":")[0]
                available = any(base_name in name for name in model_names)

            if available:
                logger.info("Model '%s' is available on Ollama", self.model_id)
            else:
                logger.warning(
                    "Model '%s' not found. Available: %s",
                    self.model_id,
                    ", ".join(model_names),
                )

            return available

        except requests.ConnectionError:
            logger.error("Cannot connect to Ollama at %s", self.base_url)
            return False
        except Exception as e:
            logger.error("Error checking Ollama availability: %s", e)
            return False

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> Dict[str, Any]:
        """
        Generate a structured JSON response from the Ollama model.

        Uses the /api/chat endpoint with system and user messages.
        Includes retry logic with exponential backoff for reliability.

        Args:
            system_prompt: System-level instruction for the model.
            user_prompt: The user query / event narrative.

        Returns:
            Parsed JSON dictionary from the model's response.

        Raises:
            ValueError: If the model response cannot be parsed as valid JSON
                        after all retries.
            ConnectionError: If the Ollama server is unreachable.
            TimeoutError: If the request exceeds the timeout limit.
        """
        from preprocessing.cache_manager import CacheManager
        
        # Check LLM Cache first
        cached_response = CacheManager.get_llm_cache(self.model_id, system_prompt, user_prompt)
        if cached_response is not None:
            logger.info("Loaded response from LLM cache for model %s", self.model_id)
            return cached_response

        last_error = None

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug(
                    "Attempt %d/%d for model %s",
                    attempt,
                    self.max_retries,
                    self.model_id,
                )

                # Make the API request
                raw_response = self._call_ollama(system_prompt, user_prompt)

                # Parse and validate JSON from the response
                parsed = self._extract_json(raw_response)

                if parsed is not None:
                    logger.debug("Successfully parsed JSON on attempt %d", attempt)
                    # Save successful parse to cache
                    CacheManager.set_llm_cache(self.model_id, system_prompt, user_prompt, parsed)
                    return parsed

                # JSON extraction failed — will retry
                last_error = ValueError(
                    f"Could not extract valid JSON from response (attempt {attempt})"
                )
                logger.warning(
                    "JSON extraction failed on attempt %d. Response preview: %s",
                    attempt,
                    raw_response[:200],
                )

            except requests.ConnectionError as e:
                last_error = ConnectionError(
                    f"Ollama server unreachable at {self.base_url}: {e}"
                )
                logger.error("Connection error on attempt %d: %s", attempt, e)

            except requests.Timeout as e:
                last_error = TimeoutError(
                    f"Request timed out after {self.request_timeout}s: {e}"
                )
                logger.error("Timeout on attempt %d: %s", attempt, e)

            except Exception as e:
                last_error = e
                logger.error("Unexpected error on attempt %d: %s", attempt, e)

            # Exponential backoff before retry
            if attempt < self.max_retries:
                delay = self.retry_base_delay * (2 ** (attempt - 1))
                logger.info("Retrying in %ds...", delay)
                time.sleep(delay)

        # All retries exhausted
        raise last_error or ValueError("All retries exhausted with no response")

    def _call_ollama(self, system_prompt: str, user_prompt: str) -> str:
        """
        Make a chat completion request to the Ollama API.

        Args:
            system_prompt: System message content.
            user_prompt: User message content.

        Returns:
            The raw text content from the model's response.
        """
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {
                "temperature": self.temperature,  # Centralized from config
                "num_predict": 4096,  # Allow enough tokens for JSON
            },
            "format": "json",  # Request JSON output mode
        }

        response = requests.post(
            self._chat_url,
            json=payload,
            timeout=self.request_timeout,
        )

        if response.status_code != 200:
            error_text = response.text[:500]
            raise RuntimeError(
                f"Ollama API error (HTTP {response.status_code}): {error_text}"
            )

        result = response.json()
        content = result.get("message", {}).get("content", "")

        if not content:
            raise ValueError("Empty response from Ollama model")

        # Log token usage if available
        eval_count = result.get("eval_count", 0)
        total_duration = result.get("total_duration", 0)
        if total_duration:
            duration_sec = total_duration / 1e9  # nanoseconds to seconds
            logger.debug(
                "Ollama response: %d tokens in %.1fs",
                eval_count,
                duration_sec,
            )

        return content


    def generate_raw(self, prompt: str) -> str:
        """
        Generate a raw text response (non-JSON mode).

        Useful for evaluation prompts or free-form generation.

        Args:
            prompt: The full prompt text.

        Returns:
            Raw text response from the model.
        """
        from config import TEMPERATURE_RAW

        payload = {
            "model": self.model_id,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": TEMPERATURE_RAW,
                "num_predict": 2048,
            },
        }

        response = requests.post(
            self._generate_url,
            json=payload,
            timeout=self.request_timeout,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Ollama API error (HTTP {response.status_code}): {response.text[:500]}"
            )

        return response.json().get("response", "")
