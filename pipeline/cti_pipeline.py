"""
CTI Extraction Pipeline

Orchestrates the end-to-end extraction flow:
  1. Load prompt template
  2. Parse XML events
  3. For each event:
     a. Fetch context via retriever
     b. Build prompt with context + narrative
     c. Query LLM model
     d. Validate JSON output against Pydantic schema
     e. Attempt repair on validation failures
  4. Save results as versioned JSON
  5. Create checkpoints for crash recovery
  6. Log all progress
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from data_parsers.xml_parser import CTIXMLParser
from models.base_model import BaseLLM
from models.llm_factory import LLMFactory
from retrievers.base_retriever import BaseRetriever
from retrievers.retriever_factory import RetrieverFactory
from schemas.extraction_schema import ExtractionResult
from schemas.experiment_schema import EventResult, ExperimentMetadata, ExperimentOutput

logger = logging.getLogger(__name__)


class CTIPipeline:
    """
    Main orchestration class for CTI knowledge extraction.

    Manages the complete workflow from XML parsing through LLM extraction
    to validated JSON output, with checkpoint recovery support.

    Usage:
        pipeline = CTIPipeline(model_name="ollama_gemma", retriever_name="llm_only")
        pipeline.run()
    """

    def __init__(
        self,
        model_name: str,
        retriever_name: str,
        dev_mode: bool = True,
    ) -> None:
        from config import (
            DATASET_DIR,
            OUTPUT_DIR,
            CHECKPOINT_DIR,
            CHECKPOINT_INTERVAL,
            MAX_EVENTS_DEV,
            EXTRACTION_PROMPT_FILE,
        )

        self.model_name = model_name
        self.retriever_name = retriever_name
        self.dev_mode = dev_mode
        self.max_events = MAX_EVENTS_DEV if dev_mode else None
        self.checkpoint_interval = CHECKPOINT_INTERVAL
        self.output_dir = OUTPUT_DIR
        self.checkpoint_dir = CHECKPOINT_DIR
        self.dataset_dir = str(DATASET_DIR)
        self.prompt_file = EXTRACTION_PROMPT_FILE

        # Initialize components
        self.model: BaseLLM = LLMFactory.create(model_name)
        self.retriever: BaseRetriever = RetrieverFactory.create(retriever_name)
        self.parser = CTIXMLParser(self.dataset_dir)

        # Load prompt template
        self.prompt_template = self._load_prompt()

        # Run metadata
        self.run_timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.run_dir = self.output_dir / f"run_{self.run_timestamp}"

        logger.info(
            "CTIPipeline initialized: model=%s, retriever=%s, dev=%s",
            model_name,
            retriever_name,
            dev_mode,
        )

    def run(self) -> str:
        """
        Execute the full extraction pipeline.

        Returns:
            Path to the output JSON file.
        """
        logger.info("Pipeline starting: %s + %s", self.model_name, self.retriever_name)
        start_time = time.time()

        # Parse events
        all_events = self.parser.parse_all()
        logger.info("Total events available: %d", len(all_events))

        # Apply dev mode limit
        if self.max_events and len(all_events) > self.max_events:
            all_events = all_events[: self.max_events]
            logger.info("Dev mode: limited to %d events", self.max_events)

        # Check for checkpoint resume
        resume_index = self._resume_from_checkpoint()
        completed_results = []

        if resume_index is not None and resume_index > 0:
            # Load checkpoint results
            completed_results = self._load_checkpoint_results(resume_index)
            logger.info("Resuming from checkpoint at event %d", resume_index)
        else:
            resume_index = 0

        # Process each event
        for i, event in enumerate(all_events[resume_index:], start=resume_index):
            event_start = time.time()
            event_id = event["event_id"]
            file_source = event["file_source"]

            logger.info(
                "Processing event %d/%d (ID=%s from %s)",
                i + 1,
                len(all_events),
                event_id,
                file_source,
            )

            try:
                # Step 1: Get retrieval context
                context = self.retriever.get_context(event["narrative"])

                # Step 2: Build the full prompt
                full_prompt = self._build_prompt(event["narrative"], context)

                # Step 3: Query the LLM
                system_prompt = self._get_system_prompt()
                raw_result = self.model.generate_json(system_prompt, full_prompt)

                # Step 4: Validate with Pydantic
                validated = self._validate_extraction(raw_result)
                extraction_dict = validated.model_dump() if validated else raw_result

                processing_time = time.time() - event_start

                result = EventResult(
                    event_id=event_id,
                    file_source=file_source,
                    extraction=extraction_dict,
                    processing_time_seconds=round(processing_time, 2),
                    status="success" if validated else "partial",
                )

            except Exception as e:
                processing_time = time.time() - event_start
                logger.error("Error processing event %s: %s", event_id, e)

                result = EventResult(
                    event_id=event_id,
                    file_source=file_source,
                    extraction={},
                    processing_time_seconds=round(processing_time, 2),
                    status="error",
                    error_message=str(e),
                )

            completed_results.append(result)

            # Checkpoint
            if (i + 1) % self.checkpoint_interval == 0:
                self._save_checkpoint(completed_results, i + 1)

            # Progress log
            if (i + 1) % 10 == 0 or (i + 1) == len(all_events):
                elapsed = time.time() - start_time
                logger.info(
                    "Progress: %d/%d events (%.1fs elapsed)",
                    i + 1,
                    len(all_events),
                    elapsed,
                )

        # Save final output
        output_path = self._save_results(completed_results, all_events)
        total_time = time.time() - start_time

        # Summary
        success = sum(1 for r in completed_results if r.status == "success")
        partial = sum(1 for r in completed_results if r.status == "partial")
        errors = sum(1 for r in completed_results if r.status == "error")

        logger.info(
            "Pipeline complete: %d success, %d partial, %d errors in %.1fs",
            success,
            partial,
            errors,
            total_time,
        )
        logger.info("Output saved to: %s", output_path)

        return str(output_path)

    def _load_prompt(self) -> str:
        """Load the extraction prompt template from file."""
        if not self.prompt_file.exists():
            logger.warning("Prompt file not found: %s", self.prompt_file)
            return ""

        text = self.prompt_file.read_text(encoding="utf-8")
        logger.debug("Loaded prompt template (%d chars)", len(text))
        return text

    def _get_system_prompt(self) -> str:
        """
        Extract the system-level instructions from the prompt template.

        Returns everything before the '{event_narrative}' placeholder.
        """
        # The system prompt is the template without the variable parts
        system = self.prompt_template
        # Remove the placeholders — they'll be in the user message
        system = system.replace("{context_block}", "").replace("{event_narrative}", "")
        return system.strip()

    def _build_prompt(self, narrative: str, context: List[str]) -> str:
        """
        Build the user prompt with event narrative and retrieval context.

        Args:
            narrative: The event narrative text.
            context: Retrieved context passages.

        Returns:
            Formatted user prompt string.
        """
        # Build context block
        if context:
            context_text = "## Retrieved Context\n"
            for i, passage in enumerate(context, 1):
                context_text += f"\n### Context {i}\n{passage}\n"
        else:
            context_text = ""

        user_prompt = f"{context_text}\n## Event Narrative\n{narrative}"
        return user_prompt

    def _validate_extraction(self, raw_result: Dict[str, Any]) -> Optional[ExtractionResult]:
        """
        Validate raw LLM output against the Pydantic schema.

        Attempts repair if validation fails.

        Args:
            raw_result: The parsed JSON dictionary from the LLM.

        Returns:
            Validated ExtractionResult, or None if validation fails.
        """
        try:
            return ExtractionResult(**raw_result)
        except Exception as e:
            logger.warning("Schema validation failed: %s", e)

            # Attempt repair: ensure entities/relations are lists
            repaired = {
                "entities": raw_result.get("entities", []),
                "relations": raw_result.get("relations", []),
            }

            # Filter out invalid entities
            valid_entities = []
            for entity in repaired.get("entities", []):
                if isinstance(entity, dict):
                    # Ensure required fields with defaults
                    entity.setdefault("text", "")
                    entity.setdefault("type", "unknown")
                    entity.setdefault("canonical_name", entity.get("text", ""))
                    entity.setdefault("confidence", 0.5)
                    # Clamp confidence
                    entity["confidence"] = max(0.0, min(1.0, float(entity.get("confidence", 0.5))))
                    if entity["text"]:  # Only keep non-empty entities
                        valid_entities.append(entity)

            # Filter out invalid relations
            valid_relations = []
            for relation in repaired.get("relations", []):
                if isinstance(relation, dict):
                    relation.setdefault("head", "")
                    relation.setdefault("relation", "associated_with")
                    relation.setdefault("tail", "")
                    relation.setdefault("confidence", 0.5)
                    relation["confidence"] = max(0.0, min(1.0, float(relation.get("confidence", 0.5))))
                    if relation["head"] and relation["tail"]:
                        valid_relations.append(relation)

            repaired["entities"] = valid_entities
            repaired["relations"] = valid_relations

            try:
                return ExtractionResult(**repaired)
            except Exception as e2:
                logger.error("Schema repair also failed: %s", e2)
                return None

    def _save_results(self, results: List[EventResult], events: list) -> Path:
        """
        Save validated results to a versioned JSON output file.

        Args:
            results: List of EventResult objects.
            events: Original event list (for metadata).

        Returns:
            Path to the saved output file.
        """
        # Create run directory
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Build experiment output
        metadata = ExperimentMetadata(
            method=self.retriever_name,
            model=self.model_name,
            timestamp=datetime.now().isoformat(),
            dataset_size=len(results),
            dev_mode=self.dev_mode,
        )

        output = ExperimentOutput(
            experiment_metadata=metadata,
            results=results,
        )

        # Filename format: Extracted_data_{method}_{model}.json
        filename = f"Extracted_data_{self.retriever_name}_{self.model_name}.json"
        output_path = self.run_dir / filename

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output.model_dump(), f, indent=2, ensure_ascii=False)

        logger.info("Results saved: %s", output_path)
        return output_path

    def _save_checkpoint(self, results: List[EventResult], index: int) -> None:
        """
        Save a checkpoint for crash recovery.

        Args:
            results: Results processed so far.
            index: Number of events processed.
        """
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_name": self.model_name,
            "retriever_name": self.retriever_name,
            "events_processed": index,
            "timestamp": datetime.now().isoformat(),
            "results": [r.model_dump() for r in results],
        }

        checkpoint_path = (
            self.checkpoint_dir
            / f"checkpoint_{self.model_name}_{self.retriever_name}_{index}.json"
        )

        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, indent=2, ensure_ascii=False)

        logger.info("Checkpoint saved: %s (%d events)", checkpoint_path, index)

    def _resume_from_checkpoint(self) -> Optional[int]:
        """
        Look for the most recent checkpoint for this model+retriever combo.

        Returns:
            The event index to resume from, or None if no checkpoint.
        """
        pattern = f"checkpoint_{self.model_name}_{self.retriever_name}_*.json"
        checkpoints = sorted(self.checkpoint_dir.glob(pattern))

        if not checkpoints:
            return None

        latest = checkpoints[-1]
        logger.info("Found checkpoint: %s", latest.name)

        try:
            with open(latest, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("events_processed", 0)
        except Exception as e:
            logger.warning("Failed to read checkpoint: %s", e)
            return None

    def _load_checkpoint_results(self, index: int) -> List[EventResult]:
        """
        Load results from a checkpoint file.

        Args:
            index: The checkpoint event index.

        Returns:
            List of EventResult objects from the checkpoint.
        """
        pattern = f"checkpoint_{self.model_name}_{self.retriever_name}_{index}.json"
        checkpoint_path = self.checkpoint_dir / pattern

        if not checkpoint_path.exists():
            return []

        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return [EventResult(**r) for r in data.get("results", [])]
        except Exception as e:
            logger.warning("Failed to load checkpoint results: %s", e)
            return []
