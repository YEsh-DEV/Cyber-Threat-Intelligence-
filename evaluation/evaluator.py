"""
CTI Extraction Evaluator

Computes evaluation metrics for extraction quality:
  - Faithfulness
  - Relevance
  - Evidence Coverage
  - Hallucination Rate

Uses a LLM as an evaluation judge to score extractions against source narrative.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from data_parsers.xml_parser import CTIXMLParser
from models.llm_factory import LLMFactory

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluates CTI extraction quality using multiple LLM-judged metrics."""

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
            for key in ["faithfulness", "relevance", "evidence_coverage", "hallucination_rate"]:
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
    ) -> Dict[str, Any]:
        """
        Evaluate all results in an experiment output file and save evaluation report.
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

        # Use CTIXMLParser to map narratives by event_id
        from config import DATASET_DIR
        parser = CTIXMLParser(str(DATASET_DIR))
        all_xml_events = parser.parse_all()
        narrative_lookup = {e["event_id"]: e["narrative"] for e in all_xml_events}

        evaluated_results = []
        metrics_accum = {
            "faithfulness": 0.0,
            "relevance": 0.0,
            "evidence_coverage": 0.0,
            "hallucination_rate": 0.0,
        }
        valid_eval_count = 0

        for i, res in enumerate(results, 1):
            event_id = res.get("event_id")
            status = res.get("status")
            extraction = res.get("extraction", {})

            if status == "error" or not extraction:
                logger.info("Skipping failed event %s in evaluation.", event_id)
                continue

            narrative = narrative_lookup.get(event_id, "")
            if not narrative:
                logger.warning("Original narrative for event %s not found. Skipping.", event_id)
                continue

            # Fetch context if retriever is supplied
            context = None
            if retriever_for_context:
                context = retriever_for_context.get_context(narrative)

            logger.info("Evaluating event %d/%d (ID=%s)...", i, len(results), event_id)
            scores = self.evaluate(narrative, extraction, context)
            
            evaluated_results.append({
                "event_id": event_id,
                "scores": scores,
            })

            # Accumulate scores for average computation
            for key in metrics_accum:
                if key in scores:
                    metrics_accum[key] += scores[key]
            valid_eval_count += 1

        # Compute averages
        averages = {}
        if valid_eval_count > 0:
            for key, total in metrics_accum.items():
                averages[key] = round(total / valid_eval_count, 3)
        else:
            for key in metrics_accum:
                averages[key] = 0.0

        # Build full report
        report = {
            "evaluation_metadata": {
                "evaluator_model": self.evaluator_model_name,
                "experiment_model": metadata.get("model"),
                "experiment_method": metadata.get("method"),
                "evaluated_events": valid_eval_count,
            },
            "average_scores": averages,
            "detailed_scores": evaluated_results,
        }

        # Save report
        report_dir = output_path.parent
        report_filename = f"Evaluation_{output_path.name}"
        report_path = report_dir / report_filename

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info("Batch evaluation complete! Averages: %s", averages)
        logger.info("Evaluation report saved to: %s", report_path)

        return report
