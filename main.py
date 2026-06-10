"""
CTI Knowledge Graph Extraction & Benchmarking Framework
Main Entry Point

Supports multiple modes:
  1. Interactive Mode -- User selects retrieval strategy and model
  2. Batch Benchmark Mode -- Runs full 3x4 experiment matrix
  3. Preprocess Mode -- Run preprocessing independently
  4. Evaluate Mode -- Run evaluation on existing outputs
"""

import argparse
import sys
import logging

from config import LOG_LEVEL, LOG_FORMAT, LOG_DATE_FORMAT, LOG_DIR


def setup_logging() -> None:
    """Configure project-wide logging."""
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_DIR / "cti_framework.log", encoding="utf-8"),
        ],
    )


def preprocess_mode(rebuild: bool = False) -> None:
    """Run preprocessing to cache datasets."""
    from preprocessing.preprocess import preprocess_all

    print("\n" + "=" * 60)
    print("  CTI Knowledge Graph -- Preprocessing")
    print("=" * 60)

    summary = preprocess_all(force=rebuild)

    print(f"\n-> Preprocessing complete:")
    print(f"   XML events cached: {summary['xml_events']}")
    print(f"   XML parse time: {summary['xml_time_seconds']}s")
    print(f"   STIX setup time: {summary['stix_time_seconds']}s")
    print(f"   Total time: {summary['total_time_seconds']}s")


def interactive_mode(dev_mode: bool = True) -> None:
    """Run a single experiment with user-selected parameters."""
    from config import RETRIEVER_REGISTRY, MODEL_REGISTRY
    from pipeline.cti_pipeline import CTIPipeline

    print("\n" + "=" * 60)
    print("  CTI Knowledge Graph -- Interactive Mode")
    print("=" * 60)

    # Select retrieval strategy
    print("\nAvailable Retrieval Strategies:")
    retriever_names = list(RETRIEVER_REGISTRY.keys())
    for i, name in enumerate(retriever_names, 1):
        print(f"  {i}. {name}")

    while True:
        try:
            choice = int(input("\nSelect strategy (number): "))
            if 1 <= choice <= len(retriever_names):
                selected_retriever = retriever_names[choice - 1]
                break
        except (ValueError, IndexError):
            pass
        print("Invalid selection. Try again.")

    # Select model
    print("\nAvailable Models:")
    model_names = list(MODEL_REGISTRY.keys())
    for i, name in enumerate(model_names, 1):
        print(f"  {i}. {name}")

    while True:
        try:
            choice = int(input("\nSelect model (number): "))
            if 1 <= choice <= len(model_names):
                selected_model = model_names[choice - 1]
                break
        except (ValueError, IndexError):
            pass
        print("Invalid selection. Try again.")

    print(f"\n-> Running: {selected_retriever} + {selected_model}")
    print(f"   Dev mode: {dev_mode}")
    print()

    # Run pipeline
    pipeline = CTIPipeline(
        model_name=selected_model,
        retriever_name=selected_retriever,
        dev_mode=dev_mode,
    )
    output_path = pipeline.run()
    print(f"\n-> Output saved: {output_path}")

    # Ask about Neo4j loading
    load_neo4j = input("\nLoad results into Neo4j? (y/n): ").strip().lower()
    if load_neo4j == "y":
        from graph.neo4j_loader import Neo4jLoader

        append = input("Append to existing graph? (y/n): ").strip().lower() == "y"

        with Neo4jLoader() as loader:
            stats = loader.load_json(output_path, append=append)
            print(f"\n  Loaded: {stats['events_created']} events, "
                  f"{stats['entities_created']} entities, "
                  f"{stats['relations_created']} relations")


def evaluate_mode(output_path: str, eval_model: str = "llama_groq") -> None:
    """Run evaluation on an existing experiment output file."""
    from evaluation.evaluator import Evaluator

    print("\n" + "=" * 60)
    print("  CTI Knowledge Graph -- Evaluation Mode")
    print("=" * 60)

    print(f"\n-> Evaluating: {output_path}")
    print(f"   Judge model: {eval_model}")
    print()

    evaluator = Evaluator(evaluator_model_name=eval_model)
    report = evaluator.evaluate_batch(output_path)

    stats = report.get("statistics", {})
    averages = stats.get("averages", {})
    std_devs = stats.get("std_devs", {})

    print(f"\n-> Evaluation complete ({stats.get('evaluated_count', 0)} events):")
    for metric in ["faithfulness", "relevance", "evidence_coverage", "hallucination_rate"]:
        avg = averages.get(metric, 0.0)
        std = std_devs.get(metric, 0.0)
        print(f"   {metric}: {avg:.3f} ± {std:.3f}")


def batch_mode(dev_mode: bool = True) -> None:
    """Run the full 3x4 benchmark matrix."""
    from config import BENCHMARK_MODELS, BENCHMARK_RETRIEVERS
    from pipeline.cti_pipeline import CTIPipeline

    print("\n" + "=" * 60)
    print("  CTI Knowledge Graph -- Batch Benchmark Mode")
    print("=" * 60)

    total = len(BENCHMARK_RETRIEVERS) * len(BENCHMARK_MODELS)
    print(f"\nWill run {total} experiments:")

    for retriever in BENCHMARK_RETRIEVERS:
        for model in BENCHMARK_MODELS:
            print(f"  - {retriever} + {model}")

    confirm = input(f"\nProceed with {total} experiments? (y/n): ").strip().lower()
    if confirm != "y":
        print("Cancelled.")
        return

    results = []
    for i, retriever in enumerate(BENCHMARK_RETRIEVERS):
        for j, model in enumerate(BENCHMARK_MODELS):
            run_num = i * len(BENCHMARK_MODELS) + j + 1
            print(f"\n{'='*60}")
            print(f"  Experiment {run_num}/{total}: {retriever} + {model}")
            print(f"{'='*60}")

            try:
                pipeline = CTIPipeline(
                    model_name=model,
                    retriever_name=retriever,
                    dev_mode=dev_mode,
                )
                output_path = pipeline.run()
                results.append({
                    "retriever": retriever,
                    "model": model,
                    "output": output_path,
                    "status": "success",
                })
                print(f"  -> Output: {output_path}")
            except Exception as e:
                print(f"  -> Error: {e}")
                results.append({
                    "retriever": retriever,
                    "model": model,
                    "output": None,
                    "status": f"error: {e}",
                })

    # Summary
    print(f"\n{'='*60}")
    print(f"  Batch Benchmark Complete")
    print(f"{'='*60}")
    success = sum(1 for r in results if r["status"] == "success")
    print(f"  {success}/{total} experiments completed successfully")

    for r in results:
        status_icon = "[OK]" if r["status"] == "success" else "[FAIL]"
        print(f"  {status_icon} {r['retriever']} + {r['model']}: {r['status']}")


def main() -> None:
    """Parse CLI arguments and dispatch to the appropriate mode."""
    parser = argparse.ArgumentParser(
        description="CTI Knowledge Graph Extraction & Benchmarking Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["interactive", "batch", "preprocess", "evaluate"],
        default="interactive",
        help="Run mode: 'interactive' for single experiment, 'batch' for full benchmark, "
             "'preprocess' to build caches, 'evaluate' to score outputs (default: interactive)",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        default=True,
        help="Enable development mode (limits events to MAX_EVENTS_DEV)",
    )
    parser.add_argument(
        "--prod",
        action="store_true",
        help="Enable production mode (process all events)",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Force rebuild all preprocessed caches",
    )
    parser.add_argument(
        "--evaluate",
        dest="evaluate_path",
        type=str,
        default=None,
        help="Path to experiment output JSON file for evaluation",
    )
    parser.add_argument(
        "--eval-model",
        type=str,
        default="llama_groq",
        help="LLM model to use as evaluation judge (default: llama_groq)",
    )

    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)

    dev_mode = not args.prod
    logger.info("CTI Framework starting in %s mode (dev=%s)", args.mode, dev_mode)

    # Handle --evaluate shortcut (overrides --mode)
    if args.evaluate_path:
        evaluate_mode(args.evaluate_path, args.eval_model)
        return

    if args.mode == "preprocess":
        preprocess_mode(rebuild=args.rebuild_cache)
    elif args.mode == "interactive":
        interactive_mode(dev_mode=dev_mode)
    elif args.mode == "batch":
        batch_mode(dev_mode=dev_mode)
    elif args.mode == "evaluate":
        # If --mode evaluate but no --evaluate path, prompt
        path = input("Enter path to experiment output JSON: ").strip()
        evaluate_mode(path, args.eval_model)


if __name__ == "__main__":
    main()
