"""
Phase 10 Verification: Evaluation Integration Tests

Tests:
  1. Evaluator initialization with 'llama_groq'
  2. Single narrative evaluation (LLM-judge scoring)
  3. Batch evaluation on an experiment output file
  4. Average score aggregation and report serialization
"""

import sys
import os
import json
import time
from pathlib import Path

# Force UTF-8 output on Windows
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_evaluator_init():
    print("=" * 60)
    print("  TEST 1: Evaluator Judge Model Initialization")
    print("=" * 60)

    from evaluation.evaluator import Evaluator

    print("  Initializing Evaluator with 'llama_groq' judge...")
    evaluator = Evaluator(evaluator_model_name="llama_groq")
    print(f"  [OK] Evaluator class created: {evaluator}")
    print(f"  [OK] Loaded prompt template size: {len(evaluator.prompt_template)} chars")
    assert len(evaluator.prompt_template) > 500, "Should successfully load evaluation prompt"
    
    return evaluator


def test_single_evaluation(evaluator):
    print("\n" + "=" * 60)
    print("  TEST 2: Single Event LLM-Judge Evaluation")
    print("=" * 60)

    # Re-use APT28 narrative and extraction
    source_narrative = (
        "APT28 used X-Agent malware to target the Democratic National Committee in 2016. "
        "They dumped lsass memory credentials from key servers."
    )
    
    # Clean extraction output
    extraction = {
        "entities": [
            {"text": "APT28", "type": "Threat Actor", "canonical_name": "APT28", "confidence": 0.95},
            {"text": "X-Agent", "type": "Malware", "canonical_name": "X-Agent", "confidence": 0.90},
            {"text": "Democratic National Committee", "type": "Organization", "canonical_name": "DNC", "confidence": 0.85},
        ],
        "relations": [
            {"head": "APT28", "relation": "used", "tail": "X-Agent", "time": "2016", "evidence": "APT28 used X-Agent malware", "confidence": 0.95},
            {"head": "APT28", "relation": "targeted", "tail": "DNC", "time": "2016", "evidence": "Targeted the Democratic National Committee", "confidence": 0.90},
        ]
    }

    print("  Submitting extraction to judge model for scoring...")
    start = time.time()
    scores = evaluator.evaluate(source_narrative, extraction)
    elapsed = time.time() - start

    print(f"  [OK] Evaluation completed in {elapsed:.1f}s")
    print(f"  Scores:")
    print(f"    - Faithfulness: {scores.get('faithfulness')}")
    print(f"    - Relevance: {scores.get('relevance')}")
    print(f"    - Evidence Coverage: {scores.get('evidence_coverage')}")
    print(f"    - Hallucination Rate: {scores.get('hallucination_rate')}")
    print(f"    - Reasoning: {scores.get('reasoning')}")

    # Basic validations
    assert "faithfulness" in scores, "Should contain faithfulness score"
    assert "relevance" in scores, "Should contain relevance score"
    assert "evidence_coverage" in scores, "Should contain evidence_coverage score"
    assert "hallucination_rate" in scores, "Should contain hallucination_rate score"
    
    return True


def test_batch_evaluation(evaluator):
    print("\n" + "=" * 60)
    print("  TEST 3: Batch Evaluation & Report Generation")
    print("=" * 60)

    from config import OUTPUT_DIR

    # 1. Create a dummy experiment output JSON file
    test_run_dir = OUTPUT_DIR / "test_eval_run"
    test_run_dir.mkdir(parents=True, exist_ok=True)
    experiment_file = test_run_dir / "Extracted_data_llm_only_test.json"
    
    # We will put event_id = 2, which corresponds to Fritz_HOW-CHINA-WILL-USE-CYBER-WARFARE in the dataset
    dummy_experiment = {
        "experiment_metadata": {
            "method": "llm_only",
            "model": "test",
            "timestamp": "2026-06-09T20:00:00",
            "dataset_size": 1,
            "dev_mode": True
        },
        "results": [
            {
                "global_id": "CTIDataset_2008_ReportEvent_2",
                "event_id": "2",
                "file_source": "CTIDataset_2008_ReportEvent.xml",
                "extraction": {
                    "entities": [
                        {"text": "China", "type": "Threat Actor Group", "canonical_name": "China", "confidence": 0.90},
                        {"text": "Fritz", "type": "Author", "canonical_name": "Fritz", "confidence": 0.85}
                    ],
                    "relations": [
                        {"head": "Fritz", "relation": "wrote", "tail": "HOW-CHINA-WILL-USE-CYBER-WARFARE", "time": "2008", "evidence": "Fritz_HOW-CHINA-WILL-USE-CYBER-WARFARE", "confidence": 0.88}
                    ]
                },
                "processing_time_seconds": 1.5,
                "status": "success"
            }
        ]
    }

    with open(experiment_file, "w", encoding="utf-8") as f:
        json.dump(dummy_experiment, f, indent=2)
    print(f"  [OK] Created dummy experiment results at: {experiment_file}")

    # 2. Run batch evaluation
    print("  Executing batch evaluation...")
    report = evaluator.evaluate_batch(str(experiment_file))
    
    # 3. Verify evaluation report output
    report_file = test_run_dir / f"Evaluation_{experiment_file.name}"
    assert report_file.exists(), f"Evaluation report file does not exist: {report_file}"
    print(f"  [OK] Evaluation report successfully saved at: {report_file}")
    
    # Read saved report
    with open(report_file, "r", encoding="utf-8") as f:
        saved_report = json.load(f)
        
    print(f"  [OK] Evaluated Events Count: {saved_report['evaluation_metadata']['evaluated_events']}")
    print(f"  [OK] Batch Average Scores:")
    for metric, avg in saved_report["statistics"]["averages"].items():
        print(f"       - {metric}: {avg}")
        
    assert saved_report['evaluation_metadata']['evaluated_events'] == 1, "Should evaluate exactly 1 event"
    assert "statistics" in saved_report, "Report should include statistics"
    assert len(saved_report["detailed_scores"]) == 1, "Report should include detailed scores for the event"

    return True


def main():
    print("\n" + "=" * 60)
    print("  CTI Framework -- Phase 10 Verification: Evaluation Integration")
    print("=" * 60 + "\n")

    evaluator = test_evaluator_init()
    
    success = test_single_evaluation(evaluator)
    
    success &= test_batch_evaluation(evaluator)

    print("\n" + "=" * 60)
    if success:
        print("  ALL EVALUATION TESTS PASSED -- Phase 10 Verified!")
    else:
        print("  SOME TESTS FAILED -- Phase 10 Verification Failed!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
