"""
Checkpoint Isolation Tests

Verifies:
  1. A new pipeline run has a unique run_id.
  2. Checkpoint files saved in one run are named after that run's run_id.
  3. A second run with the same model and retriever combination generates a new run_id
     and does not resume from the first run's checkpoints (isolation).
  4. Manually resuming by specifying a run_id works.
"""

import sys
import os
import json
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


def test_checkpoint_isolation():
    logger.info("=" * 60)
    logger.info("TEST: Checkpoint Isolation")
    logger.info("=" * 60)

    from pipeline.cti_pipeline import CTIPipeline
    from schemas.experiment_schema import EventResult

    # 1. Initialize First Pipeline Run
    logger.info("Initializing run 1 (gemma + llm_only)...")
    pipeline_1 = CTIPipeline(
        model_name="ollama_gemma",
        retriever_name="llm_only",
        dev_mode=True
    )
    run_id_1 = pipeline_1.run_id
    logger.info(f"Run 1 ID: {run_id_1}")

    # 2. Save checkpoint in first run
    dummy_results = [
        EventResult(
            global_id="dummy_1",
            event_id="1",
            file_source="test.xml",
            extraction={},
            processing_time_seconds=0.1,
            status="success"
        )
    ]
    
    # Save checkpoint at index 1
    logger.info("Saving checkpoint for Run 1 at index 1...")
    pipeline_1._save_checkpoint(dummy_results, 1)

    checkpoint_file_1 = pipeline_1.checkpoint_dir / f"checkpoint_{run_id_1}_1.json"
    assert checkpoint_file_1.exists(), f"Checkpoint 1 file not found: {checkpoint_file_1}"
    logger.info(f"Verified Checkpoint 1 exists at: {checkpoint_file_1.name}")

    # Verify run 1 can detect its own checkpoint
    logger.info("Checking if Run 1 detects its own checkpoint...")
    resume_idx_1 = pipeline_1._resume_from_checkpoint()
    assert resume_idx_1 == 1, f"Run 1 failed to detect its checkpoint (got {resume_idx_1}, expected 1)"
    logger.info("Run 1 successfully detected its own checkpoint.")

    # 3. Initialize Second Pipeline Run (same model and retriever)
    logger.info("\nInitializing run 2 (same model/retriever)...")
    pipeline_2 = CTIPipeline(
        model_name="ollama_gemma",
        retriever_name="llm_only",
        dev_mode=True
    )
    run_id_2 = pipeline_2.run_id
    logger.info(f"Run 2 ID: {run_id_2}")
    
    assert run_id_1 != run_id_2, "Run IDs must be unique!"
    logger.info("Verified run_id_1 and run_id_2 are unique.")

    # Verify run 2 DOES NOT detect run 1's checkpoint
    logger.info("Checking if Run 2 resumes from Run 1's checkpoint...")
    resume_idx_2 = pipeline_2._resume_from_checkpoint()
    assert resume_idx_2 is None, f"Run 2 incorrectly resumed from Run 1's checkpoint! (got {resume_idx_2}, expected None)"
    logger.info("PASS: Run 2 successfully isolated (did not resume from Run 1).")

    # 4. Verify explicit resume using run_id parameter
    logger.info("\nInitializing run 3 with explicit run_id_1 (resume scenario)...")
    pipeline_3 = CTIPipeline(
        model_name="ollama_gemma",
        retriever_name="llm_only",
        dev_mode=True,
        run_id=run_id_1
    )
    logger.info(f"Run 3 ID: {pipeline_3.run_id}")
    assert pipeline_3.run_id == run_id_1, "Run 3 failed to reuse run_id_1"
    
    resume_idx_3 = pipeline_3._resume_from_checkpoint()
    assert resume_idx_3 == 1, f"Run 3 failed to resume from checkpoint (got {resume_idx_3}, expected 1)"
    logger.info("PASS: Run 3 successfully resumed from Run 1's checkpoint by specifying run_id.")

    # Clean up checkpoint files
    logger.info("\nCleaning up test checkpoints...")
    try:
        checkpoint_file_1.unlink()
        logger.info("Cleaned up checkpoint file.")
    except Exception as e:
        logger.warning(f"Failed to clean up: {e}")

    logger.info("PASS: Checkpoint isolation verified successfully!")
    return True


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  CTI Framework — Checkpoint Isolation Test")
    print("=" * 60)

    try:
        test_checkpoint_isolation()
        print("\n" + "=" * 60)
        print("  ALL ISOLATION TESTS PASSED")
        print("=" * 60)
    except Exception as e:
        logger.error("FAIL: test_checkpoint_isolation() — %s", e)
        import traceback
        traceback.print_exc()
        print("\n" + "=" * 60)
        print("  TESTS FAILED")
        print("=" * 60)
        sys.exit(1)
