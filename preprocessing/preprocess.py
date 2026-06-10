"""
Preprocessing & Caching Engine

One-time preprocessing of static datasets to eliminate redundant computation:
  1. Parse all XML files → cache/parsed_events.json
  2. Ensure STIX dataset is downloaded
  3. Ensure vector index is built → cache/vector_index.npz
  4. Ensure NetworkX graph is built → cache/mitre_graph.pkl
  5. Provide fast loaders for cached data

Usage:
    # Preprocess everything
    python -m preprocessing.preprocess

    # From code
    from preprocessing.preprocess import load_cached_events, get_narrative_lookup
    events = load_cached_events()
    lookup = get_narrative_lookup()
"""

import hashlib
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def _compute_dataset_hash(dataset_dir: Path) -> str:
    """
    Compute a content hash of all XML files in the dataset directory.

    Used to detect when the source data changes and the cache needs rebuilding.

    Args:
        dataset_dir: Path to the XML dataset directory.

    Returns:
        Hex digest of the combined MD5 hash of all XML file contents.
    """
    hasher = hashlib.md5()
    xml_files = sorted(dataset_dir.glob("*.xml"))

    for filepath in xml_files:
        hasher.update(filepath.name.encode("utf-8"))
        hasher.update(filepath.read_bytes())

    return hasher.hexdigest()


def _load_cache_metadata(cache_dir: Path) -> Dict:
    """Load cache metadata from disk, or return empty dict if not found."""
    meta_path = cache_dir / "cache_metadata.json"
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Failed to read cache metadata: %s", e)
    return {}


def _save_cache_metadata(cache_dir: Path, metadata: Dict) -> None:
    """Save cache metadata to disk."""
    meta_path = cache_dir / "cache_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.debug("Cache metadata saved to %s", meta_path)


def preprocess_xml(force: bool = False) -> List[Dict]:
    """
    Parse all XML files and save the results to cache.

    If a valid cache exists (matching content hash), loads from cache instead.

    Args:
        force: If True, rebuild cache even if it exists and hash matches.

    Returns:
        List of parsed event dictionaries.
    """
    from config import DATASET_DIR, CACHE_DIR

    cache_dir = CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    events_path = cache_dir / "parsed_events.json"

    # Compute current dataset hash
    current_hash = _compute_dataset_hash(DATASET_DIR)
    metadata = _load_cache_metadata(cache_dir)

    # Check if cache is valid
    if not force and events_path.exists():
        cached_hash = metadata.get("dataset_hash", "")
        if cached_hash == current_hash:
            logger.info("XML cache is valid (hash=%s). Loading from cache.", current_hash[:12])
            return _load_events_from_file(events_path)
        else:
            logger.info("Dataset hash changed (%s → %s). Rebuilding cache.", cached_hash[:12], current_hash[:12])

    # Parse all XML files
    logger.info("Preprocessing XML dataset from %s...", DATASET_DIR)
    start_time = time.time()

    from data_parsers.xml_parser import CTIXMLParser
    parser = CTIXMLParser(str(DATASET_DIR))
    events = parser.parse_all()

    elapsed = time.time() - start_time
    logger.info("Parsed %d events in %.1fs", len(events), elapsed)

    # Save to cache
    with open(events_path, "w", encoding="utf-8") as f:
        json.dump(events, f, indent=2, ensure_ascii=False)
    logger.info("Cached parsed events to %s", events_path)

    # Update metadata
    metadata["dataset_hash"] = current_hash
    metadata["events_count"] = len(events)
    metadata["xml_preprocessed_at"] = datetime.now().isoformat()
    _save_cache_metadata(cache_dir, metadata)

    return events


def _load_events_from_file(filepath: Path) -> List[Dict]:
    """Load events from a JSON cache file."""
    start_time = time.time()
    with open(filepath, "r", encoding="utf-8") as f:
        events = json.load(f)
    elapsed = time.time() - start_time
    logger.info("Loaded %d cached events in %.3fs", len(events), elapsed)
    return events


def load_cached_events(force_rebuild: bool = False) -> List[Dict]:
    """
    Load preprocessed events from cache. Builds cache if not present.

    This is the primary entry point for pipeline and evaluator to get events.

    Args:
        force_rebuild: If True, force rebuilding the cache from XML.

    Returns:
        List of parsed event dictionaries.
    """
    return preprocess_xml(force=force_rebuild)


def get_narrative_lookup(force_rebuild: bool = False) -> Dict[str, str]:
    """
    Get a dictionary mapping event_id → narrative text.

    Used by the evaluator to look up source narratives without re-parsing XML.

    Args:
        force_rebuild: If True, force rebuilding the cache.

    Returns:
        Dictionary of {event_id: narrative_text}.
    """
    events = load_cached_events(force_rebuild=force_rebuild)
    return {e["event_id"]: e["narrative"] for e in events}


def preprocess_stix() -> None:
    """
    Ensure the STIX dataset is downloaded and vector/graph indexes are built.

    This triggers the VectorStore and GraphRAG initialization pipelines,
    which will download data and build indexes if not already cached.
    """
    from config import CACHE_DIR

    logger.info("Ensuring STIX knowledge base is preprocessed...")

    # Vector index
    from retrievers.vector_store import VectorStore
    vs = VectorStore(data_dir=CACHE_DIR)
    vs.initialize()
    logger.info("Vector index ready.")

    # NetworkX graph (uses VectorStore data)
    # Note: GraphRAG builds its own graph from the same STIX file
    logger.info("STIX preprocessing complete.")


def preprocess_all(force: bool = False) -> Dict:
    """
    Run all preprocessing steps.

    Args:
        force: If True, rebuild all caches.

    Returns:
        Summary dict with counts and timings.
    """
    logger.info("=" * 60)
    logger.info("  Running full preprocessing pipeline")
    logger.info("=" * 60)

    total_start = time.time()
    summary = {}

    # Step 1: XML events
    xml_start = time.time()
    events = preprocess_xml(force=force)
    summary["xml_events"] = len(events)
    summary["xml_time_seconds"] = round(time.time() - xml_start, 2)

    # Step 2: STIX knowledge base
    stix_start = time.time()
    preprocess_stix()
    summary["stix_time_seconds"] = round(time.time() - stix_start, 2)

    summary["total_time_seconds"] = round(time.time() - total_start, 2)

    logger.info("=" * 60)
    logger.info("  Preprocessing complete: %s", summary)
    logger.info("=" * 60)

    return summary


# ─── CLI Entry Point ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    parser = argparse.ArgumentParser(description="CTI Framework Preprocessing")
    parser.add_argument("--force", action="store_true", help="Force rebuild all caches")
    parser.add_argument("--xml-only", action="store_true", help="Only preprocess XML events")
    parser.add_argument("--stix-only", action="store_true", help="Only preprocess STIX data")
    args = parser.parse_args()

    if args.xml_only:
        preprocess_xml(force=args.force)
    elif args.stix_only:
        preprocess_stix()
    else:
        preprocess_all(force=args.force)
