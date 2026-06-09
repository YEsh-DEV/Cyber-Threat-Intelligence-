"""
Mistral LLM Implementation

Communicates with Mistral AI's REST API.
"""

import logging
import time
from typing import Any, Dict

import requests

from models.base_model import BaseLLM

logger = logging.getLogger(__name__)


class MistralLLM(BaseLLM):
    """
    Mistral AI LLM implementation using direct REST API requests.
    """

    def __init__(
        self,
        model_name: str,
        model_id: str = "open-mistral-7b",
        api_key: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name, **kwargs)

        from config import MISTRAL_API_KEY, MAX_RETRIES, RETRY_BASE_DELAY, REQUEST_TIMEOUT

        self.model_id = model_id
        self.api_key = api_key or MISTRAL_API_KEY
        self.max_retries = MAX_RETRIES
        self.retry_base_delay = RETRY_BASE_DELAY
        self.request_timeout = REQUEST_TIMEOUT

        if not self.api_key:
            logger.warning("Mistral API key not configured. Calls will fail.")

        # API endpoint and headers
        self._url = "https://api.mistral.ai/v1/chat/completions"
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
        Generate a structured JSON response from the Mistral API.
        """
        if not self.api_key:
            raise ValueError("Mistral API key is not configured in environment or config.")

        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }

        last_error = None

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug(
                    "Attempt %d/%d for Mistral model %s",
                    attempt,
                    self.max_retries,
                    self.model_id,
                )

                response = requests.post(
                    self._url,
                    json=payload,
                    headers=self._headers,
                    timeout=self.request_timeout,
                )

                if response.status_code == 429:
                    last_error = RuntimeError(f"Mistral API rate limit exceeded (HTTP 429): {response.text}")
                    logger.warning("Mistral API rate limit hit on attempt %d. Response: %s", attempt, response.text)
                elif response.status_code != 200:
                    last_error = RuntimeError(f"Mistral API error (HTTP {response.status_code}): {response.text}")
                    logger.error("Mistral API error on attempt %d: %s", attempt, response.text)
                else:
                    result = response.json()
                    # Extract text content
                    try:
                        content_text = result["choices"][0]["message"]["content"]
                    except (KeyError, IndexError) as e:
                        raise ValueError(f"Unexpected response structure from Mistral API: {result}") from e

                    parsed = self._extract_json(content_text)
                    if parsed is not None:
                        return parsed

                    last_error = ValueError(f"Could not extract valid JSON from Mistral response: {content_text}")

            except requests.ConnectionError as e:
                last_error = ConnectionError(f"Mistral API connection failed: {e}")
                logger.error("Connection error on attempt %d: %s", attempt, e)

            except requests.Timeout as e:
                last_error = TimeoutError(f"Mistral API request timed out: {e}")
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
            raise ValueError("Mistral API key is not configured in environment or config.")

        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
        }

        response = requests.post(
            self._url,
            json=payload,
            headers=self._headers,
            timeout=self.request_timeout,
        )

        if response.status_code != 200:
            raise RuntimeError(f"Mistral API error (HTTP {response.status_code}): {response.text}")

        result = response.json()
        try:
            return result["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise ValueError(f"Unexpected response structure from Mistral API: {result}") from e
