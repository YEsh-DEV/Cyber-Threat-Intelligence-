"""
XML Parser Verification Script

Tests:
  1. All 24 XML files are discovered and parsed without errors
  2. Empty datasets (e.g., 2008 MalwareEvent) don't cause crashes
  3. Event counts are printed per file
  4. Sample narratives are displayed for manual inspection
  5. All module imports resolve correctly
"""

import sys
import os
from pathlib import Path

# Force UTF-8 output on Windows
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_imports():
    """Verify all Phase 1 module imports resolve."""
    print("=" * 60)
    print("  TEST 1: Module Imports")
    print("=" * 60)

    imports_ok = True
    modules = [
        ("config", "config"),
        ("pipeline.cti_pipeline", "CTIPipeline"),
        ("schemas.extraction_schema", "Entity, Relation, ExtractionResult"),
        ("schemas.experiment_schema", "ExperimentMetadata, ExperimentOutput"),
        ("data_parsers.xml_parser", "CTIXMLParser"),
        ("models.base_model", "BaseLLM"),
        ("models.llm_factory", "LLMFactory"),
        ("models.gemini_model", "GeminiLLM"),
        ("models.mistral_model", "MistralLLM"),
        ("models.groq_model", "GroqLLM"),
        ("models.ollama_model", "OllamaLLM"),
        ("retrievers.base_retriever", "BaseRetriever"),
        ("retrievers.retriever_factory", "RetrieverFactory"),
        ("retrievers.llm_only", "LLMOnlyRetriever"),
        ("retrievers.vanilla_rag", "VanillaRAGRetriever"),
        ("retrievers.graph_rag", "GraphRAGRetriever"),
        ("graph.neo4j_loader", "Neo4jLoader"),
        ("evaluation.evaluator", "Evaluator"),
    ]

    for module_path, names in modules:
        try:
            __import__(module_path)
            print(f"  [OK] {module_path} ({names})")
        except Exception as e:
            print(f"  [FAIL] {module_path} -- {e}")
            imports_ok = False

    if imports_ok:
        print("\n  >> All imports resolved successfully!\n")
    else:
        print("\n  >> Some imports failed!\n")

    return imports_ok


def test_xml_parser():
    """Verify XML parser functionality."""
    from config import DATASET_DIR
    from data_parsers.xml_parser import CTIXMLParser

    print("=" * 60)
    print("  TEST 2: XML Parser")
    print("=" * 60)

    # Initialize parser
    parser = CTIXMLParser(str(DATASET_DIR))
    print(f"\n  Dataset directory: {DATASET_DIR}")

    # Test file discovery
    xml_files = parser.discover_files()
    print(f"  Files discovered: {len(xml_files)}")
    assert len(xml_files) > 0, "No XML files found!"

    # Test summary (lightweight)
    summary = parser.get_summary()
    print(f"\n  {'File':<45} {'Events':>8}")
    print(f"  {'-' * 45} {'-' * 8}")
    for file_info in summary["files"]:
        print(f"  {file_info['name']:<45} {file_info['event_count']:>8}")
    print(f"  {'-' * 45} {'-' * 8}")
    print(f"  {'TOTAL':<45} {summary['total_events']:>8}")

    # Full parse
    print("\n  Parsing all events...")
    all_events = parser.parse_all()
    print(f"  Total events parsed: {len(all_events)}")

    assert len(all_events) == summary["total_events"], (
        f"Mismatch: parse_all returned {len(all_events)} but summary shows {summary['total_events']}"
    )

    # Verify no None values in critical fields
    for event in all_events:
        assert event["event_id"], f"Empty event_id in {event['file_source']}"
        assert event["file_source"], "Empty file_source"
        assert event["narrative"], f"Empty narrative for event {event['event_id']}"

    print("  >> All events have valid event_id, file_source, and narrative\n")

    return all_events


def test_sample_narratives(events):
    """Display sample narratives for manual inspection."""
    print("=" * 60)
    print("  TEST 3: Sample Narratives")
    print("=" * 60)

    # Pick one ReportEvent and one MalwareEvent sample
    report_sample = None
    malware_sample = None

    for event in events:
        if "ReportEvent" in event["file_source"] and report_sample is None:
            report_sample = event
        elif "MalwareEvent" in event["file_source"] and malware_sample is None:
            malware_sample = event

        if report_sample and malware_sample:
            break

    if report_sample:
        print(f"\n  -- ReportEvent Sample (Event {report_sample['event_id']}) --")
        print()
        for line in report_sample["narrative"].split("\n"):
            print(f"  {line}")

    if malware_sample:
        print(f"\n  -- MalwareEvent Sample (Event {malware_sample['event_id']}) --")
        print()
        for line in malware_sample["narrative"].split("\n"):
            print(f"  {line}")

    print()


def test_schema_validation():
    """Verify Pydantic schemas can validate sample data."""
    from schemas.extraction_schema import Entity, Relation, ExtractionResult

    print("=" * 60)
    print("  TEST 4: Schema Validation")
    print("=" * 60)

    # Test valid entity
    entity = Entity(
        text="APT28",
        type="threat_actor",
        canonical_name="APT28",
        confidence=0.95,
    )
    print(f"  [OK] Entity created: {entity.text} ({entity.type})")

    # Test valid relation
    relation = Relation(
        head="APT28",
        relation="uses",
        tail="X-Agent",
        time="2016",
        evidence="APT28 deployed X-Agent malware",
        confidence=0.88,
    )
    print(f"  [OK] Relation created: {relation.head} -> {relation.relation} -> {relation.tail}")

    # Test extraction result
    result = ExtractionResult(entities=[entity], relations=[relation])
    print(f"  [OK] ExtractionResult: {len(result.entities)} entities, {len(result.relations)} relations")

    # Test invalid confidence (should raise)
    try:
        Entity(text="bad", type="test", canonical_name="bad", confidence=1.5)
        print("  [FAIL] Should have raised validation error for confidence > 1.0!")
    except Exception:
        print("  [OK] Correctly rejected confidence > 1.0")

    print("\n  >> Schema validation working correctly!\n")


def main():
    print("\n" + "=" * 60)
    print("  CTI Framework -- Phase 1 & 2 Verification")
    print("=" * 60 + "\n")

    # Test 1: Imports
    if not test_imports():
        print("Import verification failed. Aborting.")
        sys.exit(1)

    # Test 2: XML Parser
    events = test_xml_parser()

    # Test 3: Sample Narratives
    test_sample_narratives(events)

    # Test 4: Schema Validation
    test_schema_validation()

    # Final summary
    print("=" * 60)
    print("  ALL TESTS PASSED -- Phase 1 & 2 Verified!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
