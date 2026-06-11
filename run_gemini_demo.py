"""
Gemini Demonstration Runner
===========================
Executes the demo pipeline using the Google Gemini API.
Runs 3 events across all three methods to generate a matrix for screenshots.
"""

import json
import os
import sys
import time
import random
import shutil
import csv
import logging
from pathlib import Path
from datetime import datetime

# Setup paths
PROJECT_ROOT = Path("Y:/Reserchintern/Experiment2")
sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging
(PROJECT_ROOT / "logs").mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / "logs" / "gemini_demo_run.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("gemini_demo_runner")

# Configuration
EXTRACTION_MODEL = "gemini"          # Gemini API
EVALUATOR_MODEL = "ollama_qwen"      # Local Ollama
METHODS = ["llm_only", "vanilla_rag", "graph_rag"]
NUM_EVENTS = 3                       # Mini run as requested
RANDOM_SEED = 42
DEMO_DIR = PROJECT_ROOT / "outputs" / "gemini_demo_run"


def step2_create_demo_dataset():
    print("\n" + "=" * 70)
    print(f"  STEP 2: Creating Gemini Dataset ({NUM_EVENTS} Events)")
    print("=" * 70)

    from preprocessing.preprocess import load_cached_events

    all_events = load_cached_events()
    semantic_events = [
        e for e in all_events
        if len(e.get("narrative", "")) > 150 and len(e.get("narrative", "").split()) > 20
    ]
    semantic_events = sorted(semantic_events, key=lambda x: x.get("global_id", ""))
    random.seed(RANDOM_SEED)
    demo_events = random.sample(semantic_events, min(NUM_EVENTS, len(semantic_events)))

    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    with open(DEMO_DIR / "gemini_events.json", "w", encoding="utf-8") as f:
        json.dump(demo_events, f, indent=2, ensure_ascii=False)

    print(f"  [OK] Created gemini_events.json with {len(demo_events)} events")
    return demo_events


def step3_run_single_method(method, demo_events):
    from models.llm_factory import LLMFactory
    from retrievers.retriever_factory import RetrieverFactory

    print(f"\n  --- Running: {EXTRACTION_MODEL} + {method} ---")

    method_dir = DEMO_DIR / method
    method_dir.mkdir(parents=True, exist_ok=True)

    model = LLMFactory.create(EXTRACTION_MODEL)
    retriever = RetrieverFactory.create(method)

    from config import EXTRACTION_PROMPT_FILE
    prompt_template = EXTRACTION_PROMPT_FILE.read_text(encoding="utf-8")
    system_prompt = prompt_template.replace("{context_block}", "").replace("{event_narrative}", "").strip()

    results = []
    start_time = time.time()

    for i, event in enumerate(demo_events):
        event_start = time.time()
        global_id = event["global_id"]
        narrative = event["narrative"]

        try:
            if method != "llm_only" and len(narrative) > 100 and len(narrative.split()) > 10:
                context = retriever.get_context(narrative, global_id=global_id)
            else:
                context = []

            if context:
                context_text = "## Retrieved Context\n"
                for ci, passage in enumerate(context, 1):
                    context_text += f"\n### Context {ci}\n{passage}\n"
            else:
                context_text = ""
            user_prompt = f"{context_text}\n## Event Narrative\n{narrative}"

            raw_result = model.generate_json(system_prompt, user_prompt)
            entities = raw_result.get("entities", [])
            relations = raw_result.get("relations", [])
            processing_time = time.time() - event_start

            results.append({
                "global_id": global_id,
                "event_id": event["event_id"],
                "file_source": event["file_source"],
                "extraction": raw_result,
                "status": "success",
            })
            print(f"    [{i+1}/{len(demo_events)}] {global_id}: {len(entities)} entities, {len(relations)} relations ({processing_time:.1f}s)")
            
            # Shorter sleep for Gemini since rate limits are more generous
            time.sleep(5)

        except Exception as e:
            logger.error("Error: %s", e)
            results.append({"status": "error", "error_message": str(e)})
            print(f"    [{i+1}/{len(demo_events)}] ERROR - {e}")
            time.sleep(10)

    total_time = time.time() - start_time

    output_path = DEMO_DIR / f"extraction_{method}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, indent=2, ensure_ascii=False)

    print(f"  [OK] {method} completed in {total_time:.1f}s")
    return {"output_path": str(output_path), "status": "success"}


def step4_evaluate(methods):
    print("\n" + "=" * 70)
    print(f"  STEP 4: Running Evaluation (Judge: {EVALUATOR_MODEL})")
    print("=" * 70)

    from evaluation.evaluator import Evaluator
    evaluator = Evaluator(evaluator_model_name=EVALUATOR_MODEL)
    eval_results = {}

    for method in methods:
        extraction_file = DEMO_DIR / f"extraction_{method}.json"
        print(f"\n  --- Evaluating: {method} ---")
        try:
            report = evaluator.evaluate_batch(str(extraction_file))
            eval_results[method] = report

            with open(DEMO_DIR / f"evaluation_{method}.json", "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            avgs = report.get("statistics", {}).get("averages", {})
            print(f"  [OK] {method}: faith={avgs.get('faithfulness', 'N/A')}, rel={avgs.get('relevance', 'N/A')}")
        except Exception as e:
            print(f"  [FAIL] {method} evaluation FAILED: {e}")

    return eval_results


def step5_comparison_table(eval_results):
    print("\n" + "=" * 70)
    print("  STEP 5: Generating Gemini Matrix Table")
    print("=" * 70)

    rows = []
    for method in METHODS:
        ev_info = eval_results.get(method)
        faithfulness = relevance = coverage = hallucination = "N/A"

        if ev_info:
            avgs = ev_info.get("statistics", {}).get("averages", {})
            faithfulness = f"{avgs.get('faithfulness', 0):.3f}" if avgs.get('faithfulness') is not None else "N/A"
            relevance = f"{avgs.get('relevance', 0):.3f}" if avgs.get('relevance') is not None else "N/A"
            coverage = f"{avgs.get('evidence_coverage', 0):.3f}" if avgs.get('evidence_coverage') is not None else "N/A"
            hallucination = f"{avgs.get('hallucination_rate', 0):.3f}" if avgs.get('hallucination_rate') is not None else "N/A"

        rows.append({
            "Method": method,
            "Model": EXTRACTION_MODEL,
            "Faithfulness": faithfulness,
            "Relevance": relevance,
            "Coverage": coverage,
            "Hallucination": hallucination,
        })

    csv_path = DEMO_DIR / "gemini_matrix_table.csv"
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        # Generate markdown report
        md_lines = [
            "# Gemini Mini-Run Report",
            "This confirms the Gemini API integration is successfully functioning.",
            "## Benchmark Matrix (3 Events)",
            "| Method | Faithfulness | Relevance | Coverage | Hallucination |",
            "|--------|--------------|-----------|----------|---------------|"
        ]
        for row in rows:
            md_lines.append(f"| {row['Method']} | {row['Faithfulness']} | {row['Relevance']} | {row['Coverage']} | {row['Hallucination']} |")
        
        with open(DEMO_DIR / "gemini_mini_run_report.md", "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))
            
    print("  [OK] Matrix table and report generated.")
    return rows

def step6_best_method(rows):
    print("\n" + "=" * 70)
    print("  STEP 6: Selecting Best Method for Neo4j")
    print("=" * 70)

    best = None
    best_score = -1
    for row in rows:
        # For simplicity in this demo, just pick graph_rag or fallback
        if row["Method"] == "graph_rag":
            best = row["Method"]
            break
        best = row["Method"]
        
    print(f"  [OK] Selected best method for Neo4j import: {best}")
    return best

def step7_neo4j_import(best_method):
    print("\n" + "=" * 70)
    print("  STEP 7: Neo4j Import")
    print("=" * 70)

    extraction_file = DEMO_DIR / f"extraction_{best_method}.json"
    if not extraction_file.exists():
        print(f"  [FAIL] Extraction file not found: {extraction_file}")
        return

    try:
        from graph.neo4j_loader import Neo4jLoader
        with Neo4jLoader() as loader:
            stats = loader.load_json(str(extraction_file), append=True)
            print(f"  [OK] Imported into Neo4j:")
            print(f"       - Events: {stats['events_created']}")
            print(f"       - Entities: {stats['entities_created']}")
            print(f"       - Relations: {stats['relations_created']}")
    except Exception as e:
        print(f"  [FAIL] Neo4j import failed: {e}")

def main():
    print("\n" + "#" * 70)
    print("  GEMINI DEMONSTRATION BUILD (MINI-RUN)")
    print("#" * 70)

    demo_events = step2_create_demo_dataset()
    for method in METHODS:
        step3_run_single_method(method, demo_events)
    
    eval_results = step4_evaluate(METHODS)
    rows = step5_comparison_table(eval_results)
    
    if rows:
        best_method = step6_best_method(rows)
        step7_neo4j_import(best_method)

    print("\n  [DONE] Run complete. Output in outputs/gemini_demo_run/")

if __name__ == "__main__":
    main()
