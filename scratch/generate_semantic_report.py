import json
import random
from pathlib import Path

PROJECT_ROOT = Path("y:/Reserchintern/Experiment2")

def compute_metrics(ground_truth_entities, extracted_entities, ground_truth_relations, extracted_relations):
    gt_ents = set([e.get("value", "").lower() for e in ground_truth_entities if e.get("value")])
    ex_ents = set([e.get("text", "").lower() for e in extracted_entities if e.get("text")])
    
    tp_ent = len(gt_ents.intersection(ex_ents))
    fp_ent = len(ex_ents - gt_ents)
    fn_ent = len(gt_ents - ex_ents)
    
    ent_prec = tp_ent / (tp_ent + fp_ent) if (tp_ent + fp_ent) > 0 else 0
    ent_rec = tp_ent / (tp_ent + fn_ent) if (tp_ent + fn_ent) > 0 else 0
    ent_f1 = 2 * (ent_prec * ent_rec) / (ent_prec + ent_rec) if (ent_prec + ent_rec) > 0 else 0
    
    def rel_to_str(rel, is_gt=False):
        if is_gt:
            return f"{rel.get('source_value','').lower()}|{rel.get('relation_type','').lower()}|{rel.get('target_value','').lower()}"
        else:
            return f"{rel.get('head','').lower()}|{rel.get('relation','').lower()}|{rel.get('tail','').lower()}"
            
    gt_rels = set([rel_to_str(r, True) for r in ground_truth_relations])
    ex_rels = set([rel_to_str(r, False) for r in extracted_relations])
    
    tp_rel = len(gt_rels.intersection(ex_rels))
    fp_rel = len(ex_rels - gt_rels)
    fn_rel = len(gt_rels - ex_rels)
    
    rel_prec = tp_rel / (tp_rel + fp_rel) if (tp_rel + fp_rel) > 0 else 0
    rel_rec = tp_rel / (tp_rel + fn_rel) if (tp_rel + fn_rel) > 0 else 0
    rel_f1 = 2 * (rel_prec * rel_rec) / (rel_prec + rel_rec) if (rel_prec + rel_rec) > 0 else 0
    
    return {"ent_f1": ent_f1, "rel_f1": rel_f1}

def generate():
    pilot_dir = PROJECT_ROOT / "outputs" / "pilot_run"
    
    semantic_events_path = PROJECT_ROOT / "cache" / "semantic_event_subset.json"
    with open(semantic_events_path, "r", encoding="utf-8") as f:
        events = json.load(f)
    events = sorted(events, key=lambda x: x.get("global_id", ""))
    random.seed(42)
    sampled_events = random.sample(events, 20)
    
    methods = ["llm_only", "vanilla_rag", "graph_rag"]
    
    report = ["# Semantic Pilot Benchmark Report\n"]
    report.append("## Setup\n")
    report.append("- **Dataset:** 20 True Semantic Events (Length > 150, Words > 20)")
    report.append("- **Extractor Model:** llama_groq")
    report.append("- **Evaluator Model:** ollama_llama3 (Local, bypassing API rate limits)\n")
    
    report.append("## Evaluation Metrics\n")
    report.append("| Method | Faithfulness | Relevance | Hallucination Rate | Evidence Coverage | Entity F1 | Relation F1 |")
    report.append("|---|---|---|---|---|---|---|")
    
    for method in methods:
        fpath = pilot_dir / f"Extracted_data_{method}_llama_groq.json"
        
        avg_faith = 0
        avg_rel = 0
        avg_halluc = 0
        avg_cov = 0
        avg_ent_f1 = 0
        avg_rel_f1 = 0
        count = 0
        
        if fpath.exists():
            with open(fpath, "r", encoding="utf-8") as f:
                d = json.load(f)
                results = d.get("results", [])
                
                for r in results:
                    if r.get("status") == "success":
                        eval_data = r.get("evaluation", {})
                        if eval_data:
                            avg_faith += eval_data.get("faithfulness", 0)
                            avg_rel += eval_data.get("relevance", 0)
                            avg_halluc += eval_data.get("hallucination_rate", 0)
                            avg_cov += eval_data.get("evidence_coverage", 0)
                            
                        # Calculate exact match F1
                        evt = next((e for e in sampled_events if e["global_id"] == r["global_id"]), None)
                        if evt and r.get("extraction"):
                            m = compute_metrics(
                                evt.get("entities", []), r["extraction"].get("entities", []), 
                                evt.get("relations", []), r["extraction"].get("relations", [])
                            )
                            avg_ent_f1 += m["ent_f1"]
                            avg_rel_f1 += m["rel_f1"]
                            
                        count += 1
        
        if count > 0:
            report.append(f"| {method} | {avg_faith/count:.2f} | {avg_rel/count:.2f} | {avg_halluc/count:.2f} | {avg_cov/count:.2f} | {avg_ent_f1/count:.3f} | {avg_rel_f1/count:.3f} |")
        else:
            report.append(f"| {method} | N/A | N/A | N/A | N/A | N/A | N/A |")
            
    with open(PROJECT_ROOT / "semantic_pilot_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report))
        
    print("Semantic pilot report generated.")

if __name__ == "__main__":
    generate()
