"""
Phase 5 Verification: CTI Pipeline Integration Tests

Tests:
  1. Pipeline initialization
  2. End-to-end dev run with 5 events
  3. JSON output generation and location
  4. Pydantic schema validation of output file
"""

import sys
import os
import json
from pathlib import Path

# Force UTF-8 output on Windows
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_pipeline_execution():
    print("=" * 60)
    print("  TEST: CTI Pipeline End-to-End Dev Execution")
    print("=" * 60)

    from pipeline.cti_pipeline import CTIPipeline
    from schemas.experiment_schema import ExperimentOutput

    # Initialize in dev mode (processes 5 events)
    print("  Initializing pipeline with 'ollama_gemma' + 'llm_only' in dev mode...")
    pipeline = CTIPipeline(
        model_name="ollama_gemma",
        retriever_name="llm_only",
        dev_mode=True
    )

    print("  Running pipeline (limit to 5 events)...")
    output_path_str = pipeline.run()
    output_path = Path(output_path_str)

    print(f"  [OK] Pipeline completed. Output saved at: {output_path}")

    # Verify file exists
    assert output_path.exists(), f"Output file does not exist: {output_path}"
    print("  [OK] Output file exists.")

    # Read and parse JSON
    print("  Reading output JSON...")
    with open(output_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Validate output schema using Pydantic
    print("  Validating output JSON against Pydantic schema...")
    try:
        validated = ExperimentOutput(**data)
        print(f"  [OK] Validation passed!")
        print(f"  [OK] Method: {validated.experiment_metadata.method}")
        print(f"  [OK] Model: {validated.experiment_metadata.model}")
        print(f"  [OK] Dataset Size: {validated.experiment_metadata.dataset_size}")
        print(f"  [OK] Dev Mode: {validated.experiment_metadata.dev_mode}")
        
        # Verify event details
        assert len(validated.results) == 5, f"Expected 5 results, got {len(validated.results)}"
        print("  [OK] Found exactly 5 results in the output file.")

        success_count = sum(1 for r in validated.results if r.status == "success")
        partial_count = sum(1 for r in validated.results if r.status == "partial")
        error_count = sum(1 for r in validated.results if r.status == "error")

        print(f"  [OK] Event status distribution: "
              f"{success_count} success, {partial_count} partial, {error_count} error")

        return True
    except Exception as e:
        print(f"  [FAIL] Schema validation failed: {e}")
        return False


def main():
    print("\n" + "=" * 60)
    print("  CTI Framework -- Phase 5 Verification: Pipeline Integration")
    print("=" * 60 + "\n")

    # Check Ollama connectivity first
    from models.ollama_model import OllamaLLM
    llm = OllamaLLM(model_name="ollama_gemma")
    if not llm.is_available():
        print("  [FATAL] Ollama not available. Make sure 'ollama serve' is running.")
        sys.exit(1)

    success = test_pipeline_execution()

    print("\n" + "=" * 60)
    if success:
        print("  ALL TESTS PASSED -- Phase 5 Verified!")
    else:
        print("  TESTS FAILED -- Phase 5 Verification Failed!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
