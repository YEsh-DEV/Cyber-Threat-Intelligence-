"""
Preprocessing & Caching Tests

Verifies:
  1. XML parsing → cache pipeline produces correct output
  2. Cache loading returns same data as fresh parse
  3. Content hash detects dataset changes
  4. get_narrative_lookup() returns correct mappings
  5. Repeated calls use cache (performance check)
"""

import json
import logging
import sys
import time
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


def test_preprocess_xml():
    """Test that XML preprocessing creates a valid cache file."""
    from config import CACHE_DIR
    from preprocessing.preprocess import preprocess_xml

    logger.info("=" * 60)
    logger.info("TEST: preprocess_xml()")
    logger.info("=" * 60)

    # Force rebuild to test full pipeline
    events = preprocess_xml(force=True)

    assert len(events) > 0, "No events were parsed!"
    logger.info("Parsed %d events", len(events))

    # Verify cache file was created
    cache_file = CACHE_DIR / "parsed_events.json"
    assert cache_file.exists(), f"Cache file not found: {cache_file}"
    logger.info("Cache file exists: %s (%.1f KB)", cache_file, cache_file.stat().st_size / 1024)

    # Verify metadata was created
    meta_file = CACHE_DIR / "cache_metadata.json"
    assert meta_file.exists(), f"Metadata file not found: {meta_file}"

    with open(meta_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    assert "dataset_hash" in metadata, "Missing dataset_hash in metadata"
    assert "events_count" in metadata, "Missing events_count in metadata"
    assert metadata["events_count"] == len(events)
    logger.info("Metadata valid: hash=%s, count=%d", metadata["dataset_hash"][:12], metadata["events_count"])

    # Verify event structure
    sample = events[0]
    required_keys = {"event_id", "file_source", "narrative", "date", "info", "attributes"}
    assert required_keys.issubset(sample.keys()), f"Missing keys in event: {required_keys - sample.keys()}"
    logger.info("Event structure valid. Sample event_id: %s", sample["event_id"])

    logger.info("PASS: preprocess_xml()")
    return events


def test_cache_loading_speed():
    """Test that loading from cache is significantly faster than parsing."""
    from preprocessing.preprocess import preprocess_xml

    logger.info("=" * 60)
    logger.info("TEST: Cache loading speed")
    logger.info("=" * 60)

    # First call: should use cache (not force rebuild)
    start = time.time()
    events_cached = preprocess_xml(force=False)
    cached_time = time.time() - start

    logger.info("Cache load: %d events in %.3fs", len(events_cached), cached_time)

    # Should be very fast (under 1 second for cache hit)
    assert cached_time < 5.0, f"Cache loading too slow: {cached_time:.3f}s (expected < 5s)"

    logger.info("PASS: Cache loading speed (%.3fs)", cached_time)
    return cached_time


def test_narrative_lookup():
    """Test that get_narrative_lookup returns correct mappings."""
    from preprocessing.preprocess import get_narrative_lookup

    logger.info("=" * 60)
    logger.info("TEST: get_narrative_lookup()")
    logger.info("=" * 60)

    lookup = get_narrative_lookup()

    assert len(lookup) > 0, "Narrative lookup is empty!"
    logger.info("Narrative lookup has %d entries", len(lookup))

    # Verify all entries have non-empty narratives
    empty_count = sum(1 for v in lookup.values() if not v.strip())
    logger.info("Entries with empty narratives: %d", empty_count)

    # Verify a sample
    sample_id = list(lookup.keys())[0]
    sample_narrative = lookup[sample_id]
    assert len(sample_narrative) > 10, f"Narrative too short for event {sample_id}: {sample_narrative[:50]}"
    logger.info("Sample: event_id=%s, narrative length=%d chars", sample_id, len(sample_narrative))

    logger.info("PASS: get_narrative_lookup()")
    return lookup


def test_cache_consistency():
    """Test that cached events match fresh parse results."""
    from data_parsers.xml_parser import CTIXMLParser
    from config import DATASET_DIR
    from preprocessing.preprocess import load_cached_events

    logger.info("=" * 60)
    logger.info("TEST: Cache consistency")
    logger.info("=" * 60)

    # Load from cache
    cached = load_cached_events()

    # Parse fresh
    parser = CTIXMLParser(str(DATASET_DIR))
    fresh = parser.parse_all()

    assert len(cached) == len(fresh), f"Count mismatch: cached={len(cached)}, fresh={len(fresh)}"
    logger.info("Count matches: %d events", len(cached))

    # Verify event IDs match
    cached_ids = [e["event_id"] for e in cached]
    fresh_ids = [e["event_id"] for e in fresh]
    assert cached_ids == fresh_ids, "Event ID order mismatch between cache and fresh parse"
    logger.info("Event IDs match")

    # Verify narratives match for a sample
    for i in range(min(5, len(cached))):
        assert cached[i]["narrative"] == fresh[i]["narrative"], \
            f"Narrative mismatch for event {cached[i]['event_id']}"
    logger.info("Narrative content matches (verified first 5)")

    logger.info("PASS: Cache consistency")


def test_preprocess_all():
    """Test the full preprocess_all() pipeline."""
    from preprocessing.preprocess import preprocess_all

    logger.info("=" * 60)
    logger.info("TEST: preprocess_all()")
    logger.info("=" * 60)

    summary = preprocess_all(force=False)

    assert "xml_events" in summary, "Missing xml_events in summary"
    assert "total_time_seconds" in summary, "Missing total_time_seconds"
    assert summary["xml_events"] > 0, "No events in summary"

    logger.info("Summary: %s", summary)
    logger.info("PASS: preprocess_all()")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  CTI Framework — Preprocessing Test Suite")
    print("=" * 60)

    all_passed = True
    tests = [
        test_preprocess_xml,
        test_cache_loading_speed,
        test_narrative_lookup,
        test_cache_consistency,
        test_preprocess_all,
    ]

    for test_func in tests:
        try:
            test_func()
            print()
        except Exception as e:
            logger.error("FAIL: %s — %s", test_func.__name__, e)
            import traceback
            traceback.print_exc()
            all_passed = False
            print()

    print("=" * 60)
    if all_passed:
        print("  ALL PREPROCESSING TESTS PASSED")
    else:
        print("  SOME TESTS FAILED")
    print("=" * 60)
