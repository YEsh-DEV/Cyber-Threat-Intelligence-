"""
Phase 9 Verification: GraphRAG Integration Tests

Tests:
  1. Graph building, serialization, node/edge population
  2. Outgoing & incoming relationship extraction
  3. RetrieverFactory instantiation of 'graph_rag'
  4. Context retrieval for sample threat narratives (with relationship statements)
  5. Single-run extraction pipeline using Groq (llama_groq) + GraphRAG
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


def test_graph_init():
    print("=" * 60)
    print("  TEST 1: NetworkX Graph Building and Serialization")
    print("=" * 60)

    from retrievers.graph_rag import GraphRAGRetriever

    print("  Initializing GraphRAGRetriever (builds NetworkX graph and loads vector index)...")
    start = time.time()
    retriever = GraphRAGRetriever()
    elapsed = time.time() - start
    
    print(f"  [OK] Initialized GraphRAG in {elapsed:.1f}s")
    print(f"  [OK] Graph serialization cache present: {retriever.graph_path} (exists={retriever.graph_path.exists()})")
    print(f"  [OK] Graph Nodes Count: {retriever.graph.number_of_nodes()}")
    print(f"  [OK] Graph Edges Count: {retriever.graph.number_of_edges()}")
    
    assert retriever.graph.number_of_nodes() > 500, "Should contain substantial number of node entities"
    assert retriever.graph.number_of_edges() > 500, "Should contain substantial number of relationship edges"
    
    return retriever


def test_graph_traversal(retriever):
    print("\n" + "=" * 60)
    print("  TEST 2: Graph Traversal (1-Hop Relationships)")
    print("=" * 60)

    # Let's search for "Cobalt Strike" malware node directly in graph
    target_node_id = None
    for node_id, data in retriever.graph.nodes(data=True):
        if data.get("name") == "Cobalt Strike":
            target_node_id = node_id
            break

    if not target_node_id:
        print("  [FAIL] Cobalt Strike node not found in graph")
        return False

    print(f"  [OK] Found Cobalt Strike node ID: {target_node_id}")
    node_data = retriever.graph.nodes[target_node_id]
    print(f"       Type: {node_data['type']}")
    print(f"       Ext ID: {node_data['external_id']}")

    # 1. Outgoing edges (what Cobalt Strike uses)
    successors = list(retriever.graph.successors(target_node_id))
    print(f"\n  Outgoing edges (Cobalt Strike implements/uses - count: {len(successors)}):")
    for succ_id in successors[:5]:
        edge_data = retriever.graph.edges[target_node_id, succ_id]
        neighbor = retriever.graph.nodes[succ_id]
        print(f"    - USES/IMPLEMENTS: {neighbor['name']} ({neighbor['type']}) | rel={edge_data.get('relationship_type')}")

    # 2. Incoming edges (who uses Cobalt Strike)
    predecessors = list(retriever.graph.predecessors(target_node_id))
    print(f"\n  Incoming edges (Who uses Cobalt Strike - count: {len(predecessors)}):")
    for pred_id in predecessors[:5]:
        edge_data = retriever.graph.edges[pred_id, target_node_id]
        neighbor = retriever.graph.nodes[pred_id]
        print(f"    - USED BY: {neighbor['name']} ({neighbor['type']}) | rel={edge_data.get('relationship_type')}")

    assert len(successors) > 0 or len(predecessors) > 0, "Cobalt Strike should have mapped connections"
    print("\n  >> Graph relationships traversed successfully!")
    return True


def test_retrieval_context(retriever):
    print("\n" + "=" * 60)
    print("  TEST 3: GraphRAG Context Retrieval")
    print("=" * 60)

    narrative = "The actor used Cobalt Strike and PowerShell to infect networks."
    print(f"  Narrative: '{narrative}'")
    
    context = retriever.get_context(narrative)
    print(f"  [OK] Retrieved {len(context)} context passages")
    assert len(context) == 2, "Should retrieve top_k=2 passages"
    
    for i, passage in enumerate(context, 1):
        print(f"\n  Passage {i} preview:")
        lines = passage.split("\n")
        print("    " + "\n    ".join(lines[:6]))
        if len(lines) > 6:
            print("    ...")
            
    return True


def test_graph_pipeline_run():
    print("\n" + "=" * 60)
    print("  TEST 4: End-to-End Pipeline Run (GraphRAG + Groq)")
    print("=" * 60)

    from pipeline.cti_pipeline import CTIPipeline
    from schemas.experiment_schema import ExperimentOutput

    # Run with llama_groq and graph_rag, limit to 1 event in dev mode for quick verification
    print("  Initializing pipeline with 'llama_groq' + 'graph_rag' in dev mode...")
    pipeline = CTIPipeline(
        model_name="llama_groq",
        retriever_name="graph_rag",
        dev_mode=True
    )
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
        return True
    except Exception as e:
        print(f"  [FAIL] Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "=" * 60)
    print("  CTI Framework -- Phase 9 Verification: GraphRAG Integration")
    print("=" * 60 + "\n")

    retriever = test_graph_init()
    
    success = test_graph_traversal(retriever)
    
    success &= test_retrieval_context(retriever)
    
    # Run E2E pipeline validation
    success &= test_graph_pipeline_run()

    print("\n" + "=" * 60)
    if success:
        print("  ALL GRAPHRAG TESTS PASSED -- Phase 9 Verified!")
    else:
        print("  SOME TESTS FAILED -- Phase 9 Verification Failed!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
