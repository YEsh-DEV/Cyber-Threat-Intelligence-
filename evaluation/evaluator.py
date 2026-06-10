"""
CTI Extraction Evaluator

Computes evaluation metrics for extraction quality:
  - Faithfulness
  - Relevance
  - Evidence Coverage
  - Hallucination Rate

Uses a LLM as an evaluation judge to score extractions against source narrative.

Improvements over original:
  - Uses cached narrative lookup instead of re-parsing XML
  - Saves partial evaluation results as checkpoints during batch runs
  - Reports standard deviation, min/max alongside averages
"""

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from models.llm_factory import LLMFactory

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluates CTI extraction quality using multiple LLM-judged metrics."""

    METRIC_KEYS = ["faithfulness", "relevance", "evidence_coverage", "hallucination_rate"]

    def __init__(self, evaluator_model_name: str = "llama_groq") -> None:
        from config import EVALUATION_PROMPT_FILE

        self.evaluator_model_name = evaluator_model_name
        self.prompt_file = EVALUATION_PROMPT_FILE

        # Initialize evaluation model
        self.evaluator_llm = LLMFactory.create(evaluator_model_name)

        # Load prompt template
        if self.prompt_file.exists():
            self.prompt_template = self.prompt_file.read_text(encoding="utf-8")
        else:
            logger.warning("Evaluation prompt file not found: %s", self.prompt_file)
            self.prompt_template = ""

        logger.info("Evaluator initialized with model=%s", evaluator_model_name)

    def evaluate(
        self,
        source_narrative: str,
        extraction_result: dict,
        retrieval_context: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate extraction quality for a single event.
        """
        if not self.prompt_template:
            raise ValueError("Prompt template not loaded.")

        # Format retrieval context block
        if retrieval_context:
            context_text = "\n".join(f"- Context: {c}" for c in retrieval_context)
        else:
            context_text = "No retrieval context was provided."

        # Format full prompt
        system_prompt = "You are an expert CTI evaluation judge. Analyze the source text and extraction and return scores in valid JSON format."
        user_prompt = (
            self.prompt_template
            .replace("{source_narrative}", source_narrative)
            .replace("{extraction_json}", json.dumps(extraction_result, indent=2, ensure_ascii=False))
            .replace("{retrieval_context}", context_text)
        )

        try:
            # Query the evaluation LLM
            logger.debug("Calling evaluator LLM...")
            scores = self.evaluator_llm.generate_json(system_prompt, user_prompt)

            # Ensure correct numeric types
            for key in self.METRIC_KEYS:
                if key in scores:
                    scores[key] = float(scores[key])

            return scores
        except Exception as e:
            logger.error("LLM evaluation call failed: %s", e)
            # Safe default scores in case of error
            return {
                "faithfulness": 0.0,
                "relevance": 0.0,
                "evidence_coverage": 0.0,
                "hallucination_rate": 1.0,
                "reasoning": f"Evaluation error occurred: {e}",
            }

    def evaluate_batch(
        self,
        experiment_output_path: str,
        retriever_for_context: Optional[Any] = None,
        checkpoint_interval: int = 10,
    ) -> Dict[str, Any]:
        """
        Evaluate all results in an experiment output file and save evaluation report.

        Now uses cached narrative lookup and saves partial results during processing.

        Args:
            experiment_output_path: Path to the experiment output JSON file.
            retriever_for_context: Optional retriever to get context for each event.
            checkpoint_interval: Save partial results every N events.
        """
        output_path = Path(experiment_output_path)
        if not output_path.exists():
            raise FileNotFoundError(f"Experiment output file not found: {output_path}")

        logger.info("Starting batch evaluation for %s...", output_path.name)

        # Load experiment results
        with open(output_path, "r", encoding="utf-8") as f:
            experiment_data = json.load(f)

        results = experiment_data.get("results", [])
        metadata = experiment_data.get("experiment_metadata", {})

        # Use cached narrative lookup instead of re-parsing XML
        from preprocessing.preprocess import get_narrative_lookup
        narrative_lookup = get_narrative_lookup()

        evaluated_results = []
        metrics_accum: Dict[str, List[float]] = {key: [] for key in self.METRIC_KEYS}

        # Prepare partial report path
        report_dir = output_path.parent
        report_filename = f"Evaluation_{output_path.name}"
        report_path = report_dir / report_filename

        for i, res in enumerate(results, 1):
            global_id = res.get("global_id", res.get("event_id")) # Fallback for old runs
            status = res.get("status")
            extraction = res.get("extraction", {})

            if status == "error" or not extraction:
                logger.info("Skipping failed event %s in evaluation.", global_id)
                continue

            narrative = narrative_lookup.get(global_id, "")
            if not narrative:
                logger.warning("Original narrative for event %s not found. Skipping.", global_id)
                continue

            # Fetch context if retriever is supplied
            context = None
            if retriever_for_context:
                context = retriever_for_context.get_context(narrative, global_id=global_id)

            logger.info("Evaluating event %d/%d (Global ID=%s)...", i, len(results), global_id)
            scores = self.evaluate(narrative, extraction, context)

            evaluated_results.append({
                "global_id": global_id,
                "scores": scores,
            })

            # Accumulate individual scores for statistics
            for key in self.METRIC_KEYS:
                if key in scores:
                    metrics_accum[key].append(scores[key])

            # Checkpoint: save partial results periodically
            if len(evaluated_results) % checkpoint_interval == 0:
                self._save_partial_report(
                    report_path, metadata, evaluated_results, metrics_accum,
                    is_partial=True,
                )

        # Compute final statistics
        statistics = self._compute_statistics(metrics_accum)

        # Build full report
        report = {
            "evaluation_metadata": {
                "evaluator_model": self.evaluator_model_name,
                "experiment_model": metadata.get("model"),
                "experiment_method": metadata.get("method"),
                "evaluated_events": len(evaluated_results),
            },
            "statistics": statistics,
            "detailed_scores": evaluated_results,
        }

        # Save final report
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info("Batch evaluation complete! Statistics: %s", statistics.get("averages", {}))
        logger.info("Evaluation report saved to: %s", report_path)

        return report

    def _compute_statistics(self, metrics_accum: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        Compute average, standard deviation, min, and max for each metric.

        Returns a structured statistics dictionary suitable for academic reporting.
        """
        valid_count = max(len(v) for v in metrics_accum.values()) if metrics_accum else 0

        averages = {}
        std_devs = {}
        mins = {}
        maxs = {}

        for key, values in metrics_accum.items():
            if values:
                avg = sum(values) / len(values)
                averages[key] = round(avg, 3)
                mins[key] = round(min(values), 3)
                maxs[key] = round(max(values), 3)

                # Standard deviation
                if len(values) > 1:
                    variance = sum((x - avg) ** 2 for x in values) / (len(values) - 1)
                    std_devs[key] = round(math.sqrt(variance), 3)
                else:
                    std_devs[key] = 0.0
            else:
                averages[key] = 0.0
                std_devs[key] = 0.0
                mins[key] = 0.0
                maxs[key] = 0.0

        return {
            "evaluated_count": valid_count,
            "averages": averages,
            "std_devs": std_devs,
            "mins": mins,
            "maxs": maxs,
        }

    def _save_partial_report(
        self,
        report_path: Path,
        metadata: dict,
        evaluated_results: List[Dict],
        metrics_accum: Dict[str, List[float]],
        is_partial: bool = True,
    ) -> None:
        """Save a partial evaluation report as a checkpoint."""
        statistics = self._compute_statistics(metrics_accum)

        partial_report = {
            "evaluation_metadata": {
                "evaluator_model": self.evaluator_model_name,
                "experiment_model": metadata.get("model"),
                "experiment_method": metadata.get("method"),
                "evaluated_events": len(evaluated_results),
                "is_partial": is_partial,
            },
            "statistics": statistics,
            "detailed_scores": evaluated_results,
        }

        partial_path = report_path.with_suffix(".partial.json")
        with open(partial_path, "w", encoding="utf-8") as f:
            json.dump(partial_report, f, indent=2, ensure_ascii=False)

        logger.debug("Partial evaluation checkpoint saved (%d events)", len(evaluated_results))
