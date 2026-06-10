"""
OpenAI-Compatible LLM Base Class

Shared implementation for LLM providers that use the OpenAI-compatible
chat completions API format (Groq, Mistral, and similar).

This eliminates the ~85% code duplication between groq_model.py and
mistral_model.py by extracting the common retry loop, error handling,
token tracking, and request logic into a single base class.
"""

import logging
import time
from typing import Any, Dict, Optional

import requests

from models.base_model import BaseLLM

logger = logging.getLogger(__name__)


class OpenAICompatibleLLM(BaseLLM):
    """
    Base class for LLM providers using the OpenAI-compatible API format.

    Subclasses only need to set:
      - api_url: The chat completions endpoint URL
      - api_key: The authentication key
      - model_id: The model identifier string

    Everything else (retry logic, JSON extraction, token tracking) is handled here.
    """

    def __init__(
        self,
        model_name: str,
        api_url: str,
        api_key: str = "",
        model_id: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name, **kwargs)

        from config import MAX_RETRIES, RETRY_BASE_DELAY, REQUEST_TIMEOUT, TEMPERATURE

        self.api_url = api_url
        self.api_key = api_key
        self.model_id = model_id
        self.max_retries = MAX_RETRIES
        self.retry_base_delay = RETRY_BASE_DELAY
        self.request_timeout = REQUEST_TIMEOUT
        self.temperature = TEMPERATURE

        if not self.api_key:
            logger.warning("%s API key not configured. Calls will fail.", self.__class__.__name__)

        self._headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> Dict[str, Any]:
        """
        Generate a structured JSON response using the OpenAI-compatible API.

        Includes retry logic with exponential backoff, rate limit handling,
        and automatic JSON extraction from response text.
        """
        if not self.api_key:
            raise ValueError(f"{self.__class__.__name__} API key is not configured.")

        from preprocessing.cache_manager import CacheManager
        
        # Check LLM Cache first
        cached_response = CacheManager.get_llm_cache(self.model_id, system_prompt, user_prompt)
        if cached_response is not None:
            logger.info("Loaded response from LLM cache for model %s", self.model_id)
            return cached_response

        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
        }

        last_error = None

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug(
                    "Attempt %d/%d for %s model %s",
                    attempt,
                    self.max_retries,
                    self.__class__.__name__,
                    self.model_id,
                )

                response = requests.post(
                    self.api_url,
                    json=payload,
                    headers=self._headers,
                    timeout=self.request_timeout,
                )

                if response.status_code == 429:
                    last_error = RuntimeError(
                        f"{self.__class__.__name__} rate limit exceeded (HTTP 429): {response.text}"
                    )
                    logger.warning(
                        "%s rate limit hit on attempt %d. Response: %s",
                        self.__class__.__name__, attempt, response.text,
                    )
                elif response.status_code != 200:
                    last_error = RuntimeError(
                        f"{self.__class__.__name__} API error (HTTP {response.status_code}): {response.text}"
                    )
                    logger.error(
                        "%s API error on attempt %d: %s",
                        self.__class__.__name__, attempt, response.text,
                    )
                else:
                    result = response.json()

                    # Extract text content
                    try:
                        content_text = result["choices"][0]["message"]["content"]
                    except (KeyError, IndexError) as e:
                        raise ValueError(
                            f"Unexpected response structure from {self.__class__.__name__}: {result}"
                        ) from e

                    # Log token usage if available
                    usage = result.get("usage", {})
                    if usage:
                        logger.info(
                            "%s token usage: prompt=%d, completion=%d, total=%d",
                            self.__class__.__name__,
                            usage.get("prompt_tokens", 0),
                            usage.get("completion_tokens", 0),
                            usage.get("total_tokens", 0),
                        )

                    parsed = self._extract_json(content_text)
                    if parsed is not None:
                        # Save successful parse to cache
                        CacheManager.set_llm_cache(self.model_id, system_prompt, user_prompt, parsed)
                        return parsed

                    last_error = ValueError(
                        f"Could not extract valid JSON from {self.__class__.__name__} response: {content_text[:200]}"
                    )

            except requests.ConnectionError as e:
                last_error = ConnectionError(f"{self.__class__.__name__} connection failed: {e}")
                logger.error("Connection error on attempt %d: %s", attempt, e)

            except requests.Timeout as e:
                last_error = TimeoutError(f"{self.__class__.__name__} request timed out: {e}")
                logger.error("Timeout on attempt %d: %s", attempt, e)

            except Exception as e:
                last_error = e
                logger.error("Unexpected error on attempt %d: %s", attempt, e)

            # Exponential backoff retry
            if attempt < self.max_retries:
                delay = self.retry_base_delay * (2 ** (attempt - 1))
                logger.info("Retrying in %ds...", delay)
                time.sleep(delay)

        raise last_error or ValueError("All retries exhausted with no response")

    def generate_raw(self, prompt: str) -> str:
        """
        Generate a raw text response (non-JSON mode).

        Useful for evaluation prompts or free-form generation.
        """
        if not self.api_key:
            raise ValueError(f"{self.__class__.__name__} API key is not configured.")

        from config import TEMPERATURE_RAW

        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "temperature": TEMPERATURE_RAW,
        }

        response = requests.post(
            self.api_url,
            json=payload,
            headers=self._headers,
            timeout=self.request_timeout,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"{self.__class__.__name__} API error (HTTP {response.status_code}): {response.text}"
            )

        result = response.json()
        try:
            return result["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise ValueError(
                f"Unexpected response structure from {self.__class__.__name__}: {result}"
            ) from e
