"""
Base LLM Abstract Class

All LLM implementations must inherit from this class and implement
the `generate_json` method. This ensures the pipeline never contains
model-specific logic.
"""

import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class BaseLLM(ABC):
    """Abstract base class for all LLM implementations."""

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        self.model_name = model_name
        self._kwargs = kwargs
        logger.info("Initialized LLM: %s", model_name)

    @abstractmethod
    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> Dict[str, Any]:
        """
        Generate a structured JSON response from the LLM.

        Args:
            system_prompt: System-level instruction for the model.
            user_prompt: The user query / event narrative.

        Returns:
            Parsed JSON dictionary from the model's response.

        Raises:
            ValueError: If the model response cannot be parsed as JSON.
            ConnectionError: If the model API is unreachable.
        """
        ...

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extract and parse JSON from the model's raw text response.

        Attempts multiple strategies:
        1. Direct JSON parse of the full response
        2. Extract JSON from markdown code blocks
        3. Find JSON object boundaries using brace matching
        4. Attempt repair of common JSON issues

        Args:
            text: Raw text response from the model.

        Returns:
            Parsed dictionary, or None if extraction fails.
        """
        text = text.strip()

        # Strategy 1: Direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Strategy 2: Extract from markdown code blocks
        code_block_patterns = [
            r"```json\s*\n?(.*?)\n?\s*```",
            r"```\s*\n?(.*?)\n?\s*```",
        ]
        for pattern in code_block_patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1).strip())
                except json.JSONDecodeError:
                    continue

        # Strategy 3: Find JSON object by brace matching
        json_str = self._find_json_object(text)
        if json_str:
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass

        # Strategy 4: Attempt repair of common issues
        repaired = self._repair_json(text)
        if repaired:
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                pass

        return None

    def _find_json_object(self, text: str) -> Optional[str]:
        """
        Find a JSON object in text by matching curly brace boundaries.

        Args:
            text: Text potentially containing a JSON object.

        Returns:
            The extracted JSON string, or None.
        """
        start = text.find("{")
        if start == -1:
            return None

        depth = 0
        in_string = False
        escape_next = False

        for i in range(start, len(text)):
            char = text[i]

            if escape_next:
                escape_next = False
                continue

            if char == "\\":
                escape_next = True
                continue

            if char == '"':
                in_string = not in_string
                continue

            if in_string:
                continue

            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]

        return None

    def _repair_json(self, text: str) -> Optional[str]:
        """
        Attempt to repair common JSON formatting issues from LLM output.

        Fixes:
        - Trailing commas before closing braces/brackets
        - Single quotes instead of double quotes

        Args:
            text: Potentially malformed JSON string.

        Returns:
            Repaired JSON string, or None if repair not possible.
        """
        # Extract the JSON portion first
        json_str = self._find_json_object(text)
        if not json_str:
            return None

        # Fix trailing commas
        json_str = re.sub(r",\s*([\]}])", r"\1", json_str)

        # Fix single quotes (simple cases only)
        if "'" in json_str and '"' not in json_str:
            json_str = json_str.replace("'", '"')

        return json_str

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_name={self.model_name!r})"

