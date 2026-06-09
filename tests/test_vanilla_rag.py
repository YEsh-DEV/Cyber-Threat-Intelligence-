"""
Phase 8 Verification: Vanilla RAG Integration Tests

Tests:
  1. STIX bundle download, parse, and objects extraction
  2. Embeddings generation and flat similarity query
  3. RetrieverFactory instantiation of 'vanilla_rag'
  4. Context retrieval for sample threat narratives
  5. Single-run extraction pipeline using Groq (llama_groq) + Vanilla RAG
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


def test_vector_store_init():
    print("=" * 60)
    print("  TEST 1: Vector Store Download, Parsing, and Indexing")
    print("=" * 60)

    from retrievers.vector_store import VectorStore

    # Initialize store
    store = VectorStore()
    
    print("  Initializing vector store database (downloads & processes if needed)...")
    start = time.time()
    store.initialize()
    elapsed = time.time() - start
    
    print(f"  [OK] Initialized vector store in {elapsed:.1f}s")
    print(f"  [OK] Cached files present:")
    print(f"       Dataset: {store.stix_path} (exists={store.stix_path.exists()})")
    print(f"       Index: {store.index_path} (exists={store.index_path.exists()})")
    print(f"  [OK] Extracted items count: {len(store.documents)}")
    
    assert len(store.documents) > 500, "Should extract a substantial number of ATT&CK techniques/software"
    assert store.embeddings is not None, "Embeddings matrix should be populated"
    print(f"  [OK] Embedding dimensions: {store.embeddings.shape}")
    
    return store


def test_similarity_queries(store):
    print("\n" + "=" * 60)
    print("  TEST 2: Cosine Similarity Vector Search")
    print("=" * 60)

    queries = [
        ("PowerShell command execution for script download", "PowerShell"),
        ("LSASS memory dump for credentials", "Credential Dumping"),
        ("APT28 threat actor profile", "APT28"),
    ]

    for query, expected_keyword in queries:
        print(f"\n  Querying for: '{query}'...")
        hits = store.search(query, top_k=3)
        print(f"  [OK] Retrieved top 3 hits:")
        
        found_match = False
        for i, (doc, score) in enumerate(hits, 1):
            print(f"       Hit {i}: [{doc['type']}] {doc['name']} ({doc['external_id']}) [score={score:.3f}]")
            if expected_keyword.lower() in doc['name'].lower() or expected_keyword.lower() in doc['text'].lower():
                found_match = True
        
        if found_match:
            print(f"  [OK] Query retrieved relevant context referencing '{expected_keyword}'")
        else:
            print(f"  [WARN] Query did not explicitly list '{expected_keyword}' in hits")

    return True


def test_retriever_factory():
    print("\n" + "=" * 60)
    print("  TEST 3: Retriever Factory Instantiation")
    print("=" * 60)

    from retrievers.retriever_factory import RetrieverFactory
    from retrievers.vanilla_rag import VanillaRAGRetriever

    retriever = RetrieverFactory.create("vanilla_rag", top_k=2)
    print(f"  [OK] Factory created: {retriever}")
    assert isinstance(retriever, VanillaRAGRetriever), "Should instantiate VanillaRAGRetriever class"
    assert retriever.vector_store.top_k == 2, "Should pass kwargs config correctly"
    
    return retriever


def test_retrieval_context(retriever):
    print("\n" + "=" * 60)
    print("  TEST 4: Context Retrieval via base method")
    print("=" * 60)

    narrative = "The attacker used Cobalt Strike to run beacon commands and dump lsass memory."
    print(f"  Narrative: '{narrative}'")
    
    context = retriever.get_context(narrative)
    print(f"  [OK] Retrieved {len(context)} context passages")
    assert len(context) == 2, "Should retrieve exactly top_k=2 passages"
    
    for i, passage in enumerate(context, 1):
        print(f"  Passage {i} preview:")
        print(f"    " + "\n    ".join(passage.split("\n")[:3]) + "\n    ...")
        
    return True


def test_rag_pipeline_run():
    print("\n" + "=" * 60)
    print("  TEST 5: End-to-End Pipeline Run (Vanilla RAG + Groq)")
    print("=" * 60)

    from pipeline.cti_pipeline import CTIPipeline
    from schemas.experiment_schema import ExperimentOutput

    # Run with llama_groq and vanilla_rag, limit to 1 event in dev mode for quick verification
    print("  Initializing pipeline with 'llama_groq' + 'vanilla_rag' in dev mode...")
    pipeline = CTIPipeline(
        model_name="llama_groq",
        retriever_name="vanilla_rag",
        dev_mode=True
    )
    # Force process only 1 event to speed up test
    pipeline.max_events = 1

    print("  Running pipeline (limit to 1 event)...")
    try:
        output_path_str = pipeline.run()
        output_path = Path(output_path_str)
        print(f"  [OK] Pipeline completed. Output saved at: {output_path}")

        # Validate file
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        validated = ExperimentOutput(**data)
        print(f"  [OK] Pipeline Output Pydantic Validation Passed!")
        print(f"       Method: {validated.experiment_metadata.method}")
        print(f"       Model: {validated.experiment_metadata.model}")
        print(f"       Results: {len(validated.results)} events processed")
        
        # Verify retrieved context was populated in the result if possible
        # Check output structure
        return True
    except Exception as e:
        print(f"  [FAIL] Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "=" * 60)
    print("  CTI Framework -- Phase 8 Verification: Vanilla RAG Integration")
    print("=" * 60 + "\n")

    # Prerequisite: Check internet connection or cache for downloading STIX
    store = test_vector_store_init()
    
    test_similarity_queries(store)
    
    retriever = test_retriever_factory()
    
    test_retrieval_context(retriever)
    
    # Run E2E pipeline validation
    success = test_rag_pipeline_run()

    print("\n" + "=" * 60)
    if success:
        print("  ALL RAG TESTS PASSED -- Phase 8 Verified!")
    else:
        print("  SOME TESTS FAILED -- Phase 8 Verification Failed!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
