"""
CTI Extraction Pipeline

Orchestrates the end-to-end flow:
  1. Load prompt template
  2. Fetch context via retriever
  3. Query LLM model
  4. Validate JSON output against schema
  5. Save results
  6. Create checkpoints
  7. Log progress

Implementation: Phase 5
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class CTIPipeline:
    """Main orchestration class for CTI knowledge extraction."""

    def __init__(
        self,
        model_name: str,
        retriever_name: str,
        dev_mode: bool = True,
    ) -> None:
        self.model_name = model_name
        self.retriever_name = retriever_name
        self.dev_mode = dev_mode
        logger.info(
            "CTIPipeline initialized: model=%s, retriever=%s, dev=%s",
            model_name,
            retriever_name,
            dev_mode,
        )

    def run(self) -> None:
        """Execute the full extraction pipeline."""
        raise NotImplementedError("Pipeline execution — Phase 5")

    def _load_prompt(self) -> str:
        """Load the extraction prompt template."""
        raise NotImplementedError("Prompt loading — Phase 5")

    def _save_results(self, results: list, output_path: str) -> None:
        """Save validated results to JSON."""
        raise NotImplementedError("Result saving — Phase 5")

    def _save_checkpoint(self, checkpoint_data: dict, index: int) -> None:
        """Save a checkpoint for crash recovery."""
        raise NotImplementedError("Checkpoint saving — Phase 5")

    def _resume_from_checkpoint(self) -> Optional[int]:
        """Attempt to resume from the last checkpoint."""
        raise NotImplementedError("Checkpoint resumption — Phase 5")
