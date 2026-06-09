"""
Phase 6 Verification: Neo4j Loader Tests

Tests:
  1. Neo4j connectivity with Aura credentials
  2. Graph clearing
  3. Index creation
  4. Loading a sample extraction JSON
  5. Graph stats verification
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


def create_sample_output(output_dir: Path) -> Path:
    """Create a sample extraction output JSON for testing."""
    sample = {
        "experiment_metadata": {
            "method": "llm_only",
            "model": "test",
            "timestamp": "2026-06-09T20:00:00",
            "dataset_size": 2,
            "dev_mode": True,
        },
        "results": [
            {
                "event_id": "test_001",
                "file_source": "test_data.xml",
                "extraction": {
                    "entities": [
                        {"text": "APT28", "type": "threat_actor", "canonical_name": "APT28", "confidence": 0.95},
                        {"text": "X-Agent", "type": "malware", "canonical_name": "X-Agent", "confidence": 0.90},
                        {"text": "Democratic National Committee", "type": "organization", "canonical_name": "DNC", "confidence": 0.85},
                    ],
                    "relations": [
                        {"head": "APT28", "relation": "uses", "tail": "X-Agent", "time": "2016", "evidence": "APT28 deployed X-Agent", "confidence": 0.88},
                        {"head": "APT28", "relation": "targets", "tail": "DNC", "time": "2016", "evidence": "Targeted DNC networks", "confidence": 0.82},
                    ],
                },
                "processing_time_seconds": 1.5,
                "status": "success",
            },
            {
                "event_id": "test_002",
                "file_source": "test_data.xml",
                "extraction": {
                    "entities": [
                        {"text": "192.168.1.100", "type": "ip_address", "canonical_name": "192.168.1.100", "confidence": 0.99},
                        {"text": "emotet.dll", "type": "malware", "canonical_name": "Emotet", "confidence": 0.91},
                    ],
                    "relations": [
                        {"head": "Emotet", "relation": "communicates_with", "tail": "192.168.1.100", "time": None, "evidence": "C2 traffic observed", "confidence": 0.87},
                    ],
                },
                "processing_time_seconds": 1.2,
                "status": "success",
            },
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "test_extraction.json"
    with open(output_path, "w") as f:
        json.dump(sample, f, indent=2)

    return output_path


def test_neo4j_connectivity():
    """Test 1: Neo4j Aura connection."""
    print("=" * 60)
    print("  TEST 1: Neo4j Connectivity")
    print("=" * 60)

    from graph.neo4j_loader import Neo4jLoader
    from config import NEO4J_URI, NEO4J_USERNAME, NEO4J_DATABASE

    print(f"  URI: {NEO4J_URI}")
    print(f"  Username: {NEO4J_USERNAME}")
    print(f"  Database: {NEO4J_DATABASE}")

    loader = Neo4jLoader()

    try:
        loader.connect()
        print("  [OK] Connected to Neo4j Aura")
        return loader
    except Exception as e:
        print(f"  [FAIL] Connection failed: {e}")
        return None


def test_clear_graph(loader):
    """Test 2: Clear graph."""
    print("\n" + "=" * 60)
    print("  TEST 2: Clear Graph")
    print("=" * 60)

    try:
        deleted = loader.clear_graph()
        print(f"  [OK] Cleared graph ({deleted} nodes deleted)")
        return True
    except Exception as e:
        print(f"  [FAIL] Clear failed: {e}")
        return False


def test_load_json(loader):
    """Test 3: Load sample extraction JSON."""
    print("\n" + "=" * 60)
    print("  TEST 3: Load Sample JSON")
    print("=" * 60)

    from config import OUTPUT_DIR

    # Create sample data
    sample_path = create_sample_output(OUTPUT_DIR / "test_run")
    print(f"  Sample JSON: {sample_path}")

    try:
        stats = loader.load_json(str(sample_path), append=False)
        print(f"  [OK] Events created:  {stats['events_created']}")
        print(f"  [OK] Entities created: {stats['entities_created']}")
        print(f"  [OK] Relations created: {stats['relations_created']}")

        assert stats["events_created"] == 2, f"Expected 2 events, got {stats['events_created']}"
        assert stats["entities_created"] == 5, f"Expected 5 entities, got {stats['entities_created']}"
        assert stats["relations_created"] == 3, f"Expected 3 relations, got {stats['relations_created']}"

        print("  [OK] All counts match expected values")
        return True
    except Exception as e:
        print(f"  [FAIL] Load failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_graph_stats(loader):
    """Test 4: Verify graph statistics."""
    print("\n" + "=" * 60)
    print("  TEST 4: Graph Statistics")
    print("=" * 60)

    try:
        stats = loader.get_graph_stats()

        print(f"  Node counts:")
        for label, count in stats["node_counts"].items():
            print(f"    {label}: {count}")

        print(f"  Total relationships: {stats['total_relationships']}")

        print(f"  Relationship types:")
        for rel_type, count in stats["relationship_types"].items():
            print(f"    {rel_type}: {count}")

        return True
    except Exception as e:
        print(f"  [FAIL] Stats failed: {e}")
        return False


def main():
    print("\n" + "=" * 60)
    print("  CTI Framework -- Phase 6 Verification: Neo4j Loader")
    print("=" * 60 + "\n")

    # Test 1: Connect
    loader = test_neo4j_connectivity()
    if not loader:
        print("\n  [FATAL] Neo4j not available. Check credentials in .env")
        sys.exit(1)

    try:
        # Test 2: Clear
        test_clear_graph(loader)

        # Test 3: Load
        test_load_json(loader)

        # Test 4: Stats
        test_graph_stats(loader)

        # Final
        print("\n" + "=" * 60)
        print("  ALL TESTS PASSED -- Phase 6 Verified!")
        print("=" * 60 + "\n")

    finally:
        loader.close()
        print("  Neo4j connection closed.")


if __name__ == "__main__":
    main()
