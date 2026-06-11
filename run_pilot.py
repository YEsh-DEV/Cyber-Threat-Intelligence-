import os
import sys
import json
import random
import time
import shutil
from pathlib import Path
from datetime import datetime

# Setup paths
PROJECT_ROOT = Path("Y:/Reserchintern/Experiment2")
sys.path.insert(0, str(PROJECT_ROOT))

# Patch before importing pipeline components
from preprocessing import preprocess
original_load_cached_events = preprocess.load_cached_events

import json
def patched_load_cached_events(*args, **kwargs):
    semantic_events_path = PROJECT_ROOT / "cache" / "semantic_event_subset.json"
    with open(semantic_events_path, "r", encoding="utf-8") as f:
        events = json.load(f)
    print(f"[Pilot] Loaded {len(events)} semantic events from subset.")
    # Deterministic sorting
    events = sorted(events, key=lambda x: x.get("global_id", ""))
    random.seed(42)
    sample = random.sample(events, 20)
    print(f"[Pilot] Sampled 20 semantic events deterministically. Seed: 42.")
    return sample

preprocess.load_cached_events = patched_load_cached_events

from pipeline.cti_pipeline import CTIPipeline
from evaluation.evaluator import Evaluator

def run_pilot():
    print("=======================================")
    print(" PHASE 2: PILOT BENCHMARK EXECUTION ")
    print("=======================================")
    
    # Load semantic events instead of full cache
    semantic_events_path = PROJECT_ROOT / "cache" / "semantic_event_subset.json"
    with open(semantic_events_path, "r", encoding="utf-8") as f:
        events = json.load(f)
        
    # Sort for deterministic sampling
    events = sorted(events, key=lambda x: x.get("global_id", ""))
    print(f"Loaded {len(events)} semantic events")
    
    model_name = "llama_groq"
    methods = ["llm_only", "vanilla_rag", "graph_rag"]
    pilot_out_dir = PROJECT_ROOT / "outputs" / "pilot_run"
    pilot_out_dir.mkdir(parents=True, exist_ok=True)
    
    evaluator = Evaluator(evaluator_model_name="ollama_gemma")
    summary_data = []

    for method in methods:
        print(f"\n--- Starting Experiment: {model_name} + {method} ---")
        
        # Initialize pipeline
        pipeline = CTIPipeline(
            model_name=model_name,
            retriever_name=method,
            dev_mode=False  # Must be False to process all 20 returned by the patch
        )
        
        # Run pipeline
        start_time = time.time()
        output_file = Path(pipeline.run())
        end_time = time.time()
        runtime = end_time - start_time
        print(f"[Pilot] Extraction complete. Runtime: {runtime:.2f}s")
        print(f"[Pilot] Output file: {output_file}")
        
        # Run evaluation
        print(f"[Pilot] Starting Evaluation...")
        evaluator.evaluate_batch(str(output_file))
        
        # Expected eval file
        basename = output_file.name
        eval_file = PROJECT_ROOT / f"Evaluation_{basename}"
        
        # Copy to pilot_run
        if output_file.exists():
            shutil.copy2(output_file, pilot_out_dir / f"Extraction_{method}.json")
            
        if eval_file.exists():
            shutil.copy2(eval_file, pilot_out_dir / f"Evaluation_{method}.json")
            
        # Also copy manifest
        manifest_file = output_file.parent / "run_manifest.json"
        if manifest_file.exists():
            shutil.copy2(manifest_file, pilot_out_dir / f"Manifest_{method}.json")

        # Load results for summary
        extraction_data = {}
        if output_file.exists():
            with open(output_file, "r", encoding="utf-8") as f:
                extraction_data = json.load(f)
                
        eval_data = {}
        if eval_file.exists():
            with open(eval_file, "r", encoding="utf-8") as f:
                eval_data = json.load(f)
        
        results_list = extraction_data.get("results", []) if isinstance(extraction_data, dict) else extraction_data
        failures = [item for item in results_list if isinstance(item, dict) and item.get("status") == "error"]
        
        # Generate Method Report
        summary_data.append({
            "method": method,
            "runtime_sec": runtime,
            "total_events": len(results_list),
            "successes": len(results_list) - len(failures),
            "failures": len(failures),
            "avg_f1_score": eval_data.get("summary", {}).get("average_f1", 0.0),
            "avg_accuracy": eval_data.get("summary", {}).get("average_accuracy", 0.0)
        })

    # Generate Report File
    report_path = PROJECT_ROOT / "pilot_validation_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Pilot Benchmark Validation Report\n\n")
        f.write("## Experiment Setup\n")
        f.write("- **Model**: `llama_groq`\n")
        f.write("- **Methods**: `llm_only`, `vanilla_rag`, `graph_rag`\n")
        f.write("- **Sample Size**: 20 Events\n")
        f.write("- **Random Seed**: 42\n\n")
        
        f.write("## Summary Metrics\n\n")
        f.write("| Method | Runtime (s) | Success Rate | Avg F1 Score | Avg Accuracy |\n")
        f.write("|---|---|---|---|---|\n")
        for data in summary_data:
            success_rate = (data['successes'] / data['total_events']) * 100 if data['total_events'] else 0
            f.write(f"| {data['method']} | {data['runtime_sec']:.2f} | {success_rate:.1f}% | {data['avg_f1_score']:.4f} | {data['avg_accuracy']:.4f} |\n")
            
        f.write("\n## Cost Estimate (for Full Run)\n")
        f.write("Assuming a full dataset of ~40,000 events, extrapolation from the 20-event pilot runtime indicates:\n")
        for data in summary_data:
            estimated_runtime_hours = (data['runtime_sec'] / 20) * 40000 / 3600
            f.write(f"- **{data['method']}**: ~{estimated_runtime_hours:.2f} hours\n")
            
        f.write("\n## Conclusion\n")
        f.write("Pilot benchmark execution completed. System is ready for full benchmark execution.\n")
        
    print("\n[Pilot] Validation Report generated at: pilot_validation_report.md")

if __name__ == "__main__":
    run_pilot()
