"""
Gemini LLM Implementation

Communicates with Google's Generative AI REST API.
"""

import logging
import time
from typing import Any, Dict

import requests

from models.base_model import BaseLLM

logger = logging.getLogger(__name__)


class GeminiLLM(BaseLLM):
    """
    Google Gemini LLM implementation using direct REST API requests.
    """

    def __init__(
        self,
        model_name: str,
        model_id: str = "gemini-2.0-flash",
        api_key: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name, **kwargs)

        from config import GEMINI_API_KEY, MAX_RETRIES, RETRY_BASE_DELAY, REQUEST_TIMEOUT, TEMPERATURE

        self.model_id = model_id
        self.api_key = api_key or GEMINI_API_KEY
        self.max_retries = MAX_RETRIES
        self.retry_base_delay = RETRY_BASE_DELAY
        self.request_timeout = REQUEST_TIMEOUT
        self.temperature = TEMPERATURE

        if not self.api_key:
            logger.warning("Gemini API key not configured. Calls will fail.")

        # API endpoint URL
        self._url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_id}:generateContent?key={self.api_key}"

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> Dict[str, Any]:
        """
        Generate a structured JSON response from the Gemini API.
        """
        if not self.api_key:
            raise ValueError("Gemini API key is not configured in environment or config.")

        from preprocessing.cache_manager import CacheManager
        
        # Check LLM Cache first
        cached_response = CacheManager.get_llm_cache(self.model_id, system_prompt, user_prompt)
        if cached_response is not None:
            logger.info("Loaded response from LLM cache for model %s", self.model_id)
            return cached_response

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": user_prompt}]
                }
            ],
            "systemInstruction": {
                "parts": [{"text": system_prompt}]
            },
            "generationConfig": {
                "temperature": self.temperature,
                "responseMimeType": "application/json"
            }
        }

        last_error = None

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug(
                    "Attempt %d/%d for Gemini model %s",
                    attempt,
                    self.max_retries,
                    self.model_id,
                )

                response = requests.post(
                    self._url,
                    json=payload,
                    timeout=self.request_timeout,
                )

                if response.status_code == 429:
                    last_error = RuntimeError(f"Gemini API rate limit exceeded (HTTP 429): {response.text}")
                    logger.warning("Gemini API rate limit hit on attempt %d. Response: %s", attempt, response.text)
                elif response.status_code != 200:
                    last_error = RuntimeError(f"Gemini API error (HTTP {response.status_code}): {response.text}")
                    logger.error("Gemini API error on attempt %d: %s", attempt, response.text)
                else:
                    result = response.json()
                    # Extract text content
                    try:
                        content_text = result["candidates"][0]["content"]["parts"][0]["text"]
                    except (KeyError, IndexError) as e:
                        raise ValueError(f"Unexpected response structure from Gemini API: {result}") from e

                    parsed = self._extract_json(content_text)
                    if parsed is not None:
                        # Save successful parse to cache
                        CacheManager.set_llm_cache(self.model_id, system_prompt, user_prompt, parsed)
                        # Log token usage if available
                        usage = result.get("usageMetadata", {})
                        if usage:
                            logger.info(
                                "Gemini token usage: prompt=%d, completion=%d, total=%d",
                                usage.get("promptTokenCount", 0),
                                usage.get("candidatesTokenCount", 0),
                                usage.get("totalTokenCount", 0),
                            )
                        return parsed

                    last_error = ValueError(f"Could not extract valid JSON from Gemini response: {content_text}")

            except requests.ConnectionError as e:
                last_error = ConnectionError(f"Gemini API connection failed: {e}")
                logger.error("Connection error on attempt %d: %s", attempt, e)

            except requests.Timeout as e:
                last_error = TimeoutError(f"Gemini API request timed out: {e}")
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
        """
        if not self.api_key:
            raise ValueError("Gemini API key is not configured in environment or config.")

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {
                "temperature": self.temperature
            }
        }

        response = requests.post(
            self._url,
            json=payload,
            timeout=self.request_timeout,
        )

        if response.status_code != 200:
            raise RuntimeError(f"Gemini API error (HTTP {response.status_code}): {response.text}")

        result = response.json()
        try:
            return result["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as e:
            raise ValueError(f"Unexpected response structure from Gemini API: {result}") from e
