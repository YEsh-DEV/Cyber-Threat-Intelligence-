"""
CTI Knowledge Graph Extraction & Benchmarking Framework
Main Entry Point

Supports two modes:
  1. Interactive Mode — User selects retrieval strategy and model
  2. Batch Benchmark Mode — Runs full 3×4 experiment matrix
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


def interactive_mode() -> None:
    """Run a single experiment with user-selected parameters."""
    from config import RETRIEVER_REGISTRY, MODEL_REGISTRY

    print("\n" + "=" * 60)
    print("  CTI Knowledge Graph — Interactive Mode")
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

    print(f"\n→ Running: {selected_retriever} + {selected_model}")
    print("  (Pipeline not yet implemented — Phase 5)")
    # TODO: Phase 5 — Instantiate and run CTIPipeline


def batch_mode() -> None:
    """Run the full 3×4 benchmark matrix."""
    from config import BENCHMARK_MODELS, BENCHMARK_RETRIEVERS

    print("\n" + "=" * 60)
    print("  CTI Knowledge Graph — Batch Benchmark Mode")
    print("=" * 60)

    total = len(BENCHMARK_RETRIEVERS) * len(BENCHMARK_MODELS)
    print(f"\nWill run {total} experiments:")

    for retriever in BENCHMARK_RETRIEVERS:
        for model in BENCHMARK_MODELS:
            print(f"  • {retriever} + {model}")

    print("\n  (Batch execution not yet implemented — Phase 11)")
    # TODO: Phase 11 — Iterate and run all experiments


def main() -> None:
    """Parse CLI arguments and dispatch to the appropriate mode."""
    parser = argparse.ArgumentParser(
        description="CTI Knowledge Graph Extraction & Benchmarking Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["interactive", "batch"],
        default="interactive",
        help="Run mode: 'interactive' for single experiment, 'batch' for full benchmark (default: interactive)",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Enable development mode (limits events to MAX_EVENTS_DEV)",
    )

    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("CTI Framework starting in %s mode", args.mode)

    if args.dev:
        import config
        config.DEV_MODE = True
        logger.info("Development mode enabled (max %d events)", config.MAX_EVENTS_DEV)

    if args.mode == "interactive":
        interactive_mode()
    elif args.mode == "batch":
        batch_mode()


if __name__ == "__main__":
    main()
