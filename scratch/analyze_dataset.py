import json
import sys
from pathlib import Path

PROJECT_ROOT = Path("y:/Reserchintern/Experiment2")
sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing import preprocess

def analyze():
    events = preprocess.load_cached_events()
    total_events = len(events)
    
    ioc_only = []
    narrative_events = []
    mixed_events = []
    
    for event in events:
        narrative = event.get("narrative", "")
            
        length = len(narrative)
        words = len(narrative.split())
        
        if length > 150 and words > 20:
            narrative_events.append(event)
        elif length > 0:
            ioc_only.append(event)
            
    # Calculate percentages
    ioc_pct = (len(ioc_only) / total_events) * 100 if total_events else 0
    narr_pct = (len(narrative_events) / total_events) * 100 if total_events else 0
    
    # Generate report
    report = [
        "# Dataset Composition Report\n",
        f"**Total Events Analyzed:** {total_events}\n",
        "## Categorization\n",
        f"- **IOC-only Events (Length <= 150 or Words <= 20):** {len(ioc_only)} ({ioc_pct:.2f}%)",
        f"- **Narrative Events (Length > 150 and Words > 20):** {len(narrative_events)} ({narr_pct:.2f}%)\n",
        "*(Note: Mixed events were grouped into the above definitions based on strict thresholds.)*\n"
    ]
    
    with open(PROJECT_ROOT / "dataset_composition_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report))
        
    # Save semantic events
    cache_dir = PROJECT_ROOT / "cache"
    cache_dir.mkdir(exist_ok=True)
    with open(cache_dir / "semantic_event_subset.json", "w", encoding="utf-8") as f:
        json.dump(narrative_events, f, indent=2)
        
    print(f"Analysis complete. {len(narrative_events)} semantic events saved.")

if __name__ == "__main__":
    analyze()
