"""
GraphRAG Retriever

Constructs a directed NetworkX knowledge graph of MITRE ATT&CK tactics, techniques,
software, and groups. Uses vector search to identify "seed" entry points, traverses
the graph to collect 1-hop relationships, and formats them into contextual passages.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx

from retrievers.base_retriever import BaseRetriever
from retrievers.vector_store import VectorStore

logger = logging.getLogger(__name__)


class GraphRAGRetriever(BaseRetriever):
    """
    Graph-based RAG retriever utilizing a local NetworkX representation of MITRE ATT&CK.
    """

    def __init__(self, name: str = "graph_rag", **kwargs: Any) -> None:
        super().__init__(name, **kwargs)

        from config import PROJECT_ROOT

        self.data_dir = PROJECT_ROOT / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.stix_path = self.data_dir / "enterprise-attack.json"
        self.graph_path = self.data_dir / "mitre_graph.pkl"

        # Graph initialization
        self.graph: nx.DiGraph = nx.DiGraph()

        # Initialize vector store for similarity entry points
        self.vector_store = VectorStore(top_k=2)
        self.vector_store.initialize()

        # Build or load the NetworkX graph
        self._initialize_graph()

    def _initialize_graph(self) -> None:
        """Load the NetworkX graph from cache, or build it from the STIX dataset."""
        if self.graph_path.exists():
            logger.info("Loading NetworkX graph from cache: %s", self.graph_path)
            try:
                with open(self.graph_path, "rb") as f:
                    self.graph = pickle.load(f)
                logger.info("Loaded graph containing %d nodes and %d edges.", 
                            self.graph.number_of_nodes(), self.graph.number_of_edges())
                return
            except Exception as e:
                logger.warning("Failed to load cached graph: %s. Rebuilding...", e)

        # Build graph
        self._build_graph()

    def _build_graph(self) -> None:
        """Parse STIX bundle to construct nodes and edges, then serialize."""
        logger.info("Building NetworkX graph from STIX data: %s", self.stix_path)

        if not self.stix_path.exists():
            raise FileNotFoundError(f"STIX dataset not found at {self.stix_path}. "
                                    f"Run VanillaRAGRetriever first to download it.")

        with open(self.stix_path, "r", encoding="utf-8") as f:
            bundle = json.load(f)

        objects = bundle.get("objects", [])
        
        # Temp lookup for STIX ID mapping
        nodes_dict: Dict[str, Dict[str, Any]] = {}
        relationships = []

        # Step 1: Extract nodes (Techniques, Malware, Tools, Groups)
        for obj in objects:
            if obj.get("revoked", False) or obj.get("x_mitre_deprecated", False):
                continue

            obj_type = obj.get("type")
            if obj_type not in ("attack-pattern", "malware", "tool", "intrusion-set"):
                continue

            stix_id = obj.get("id")
            name = obj.get("name", "")
            description = obj.get("description", "")
            
            # Extract external ID
            external_id = "unknown"
            for ref in obj.get("external_references", []):
                if ref.get("source_name") in ("mitre-attack", "mitre-enterprise-attack"):
                    external_id = ref.get("external_id", "unknown")
                    break

            readable_type = {
                "attack-pattern": "Technique",
                "malware": "Malware",
                "tool": "Tool",
                "intrusion-set": "Threat Actor Group",
            }.get(obj_type, obj_type)

            nodes_dict[stix_id] = {
                "name": name,
                "type": readable_type,
                "external_id": external_id,
                "description": description.strip(),
            }

        # Step 2: Extract relationship edges
        for obj in objects:
            if obj.get("type") != "relationship":
                continue

            rel_type = obj.get("relationship_type")
            source_ref = obj.get("source_ref")
            target_ref = obj.get("target_ref")

            # Only retain connections between nodes in our node list
            if source_ref in nodes_dict and target_ref in nodes_dict:
                relationships.append((source_ref, target_ref, rel_type))

        # Step 3: Populate NetworkX graph
        self.graph.clear()
        
        # Add nodes
        for node_id, attrs in nodes_dict.items():
            self.graph.add_node(node_id, **attrs)

        # Add edges
        for source, target, rel_type in relationships:
            self.graph.add_edge(source, target, relationship_type=rel_type)

        logger.info("Graph built: %d nodes, %d edges.", 
                    self.graph.number_of_nodes(), self.graph.number_of_edges())

        # Step 4: Serialize graph to cache file
        try:
            with open(self.graph_path, "wb") as f:
                pickle.dump(self.graph, f)
            logger.info("Saved graph cache to %s", self.graph_path)
        except Exception as e:
            logger.error("Failed to save graph cache: %s", e)

    def get_context(self, query_text: str) -> List[str]:
        """
        Retrieve context utilizing hybrid vector search + graph traversal.
        """
        logger.info("Retrieving GraphRAG context for narrative...")
        
        try:
            # 1. Enter the graph via semantic similarity (get top-2 seeds)
            hits = self.vector_store.search(query_text, top_k=2)
            if not hits:
                return []

            context_passages = []

            for doc, score in hits:
                stix_id = doc["id"]
                
                # Check if seed node exists in graph
                if not self.graph.has_node(stix_id):
                    # Fallback to direct description
                    context_passages.append(doc["text"])
                    continue

                node_attrs = self.graph.nodes[stix_id]
                node_name = node_attrs["name"]
                node_type = node_attrs["type"]
                node_ext_id = node_attrs["external_id"]
                node_desc = node_attrs["description"]

                # Gather 1-hop relationships
                relations_str_list = []

                # A. Outgoing relationships (seed -> neighbor)
                for neighbor_id in self.graph.successors(stix_id):
                    edge_data = self.graph.edges[stix_id, neighbor_id]
                    rel_type = edge_data.get("relationship_type", "associated_with")
                    neighbor_attrs = self.graph.nodes[neighbor_id]
                    
                    # Format: "Cobalt Strike (Malware) USES PowerShell (Technique)"
                    rel_statement = (
                        f"- {node_name} ({node_type}) {rel_type.upper().replace('-', '_')} "
                        f"{neighbor_attrs['name']} ({neighbor_attrs['type']})"
                    )
                    relations_str_list.append(rel_statement)

                # B. Incoming relationships (neighbor -> seed)
                for neighbor_id in self.graph.predecessors(stix_id):
                    edge_data = self.graph.edges[neighbor_id, stix_id]
                    rel_type = edge_data.get("relationship_type", "associated_with")
                    neighbor_attrs = self.graph.nodes[neighbor_id]
                    
                    # Format: "APT29 (Threat Group) USES Cobalt Strike (Malware)"
                    rel_statement = (
                        f"- {neighbor_attrs['name']} ({neighbor_attrs['type']}) {rel_type.upper().replace('-', '_')} "
                        f"{node_name} ({node_type})"
                    )
                    relations_str_list.append(rel_statement)

                # Format clean passage
                relations_block = "\n".join(relations_str_list) if relations_str_list else "- No direct relations mapped."

                passage = (
                    f"MITRE ATT&CK {node_type}: {node_name} ({node_ext_id})\n"
                    f"Description: {node_desc[:600]}...\n\n"
                    f"Mapped Relationships:\n{relations_block}"
                )

                context_passages.append(passage)

            return context_passages

        except Exception as e:
            logger.error("GraphRAG retrieval failed: %s", e)
            import traceback
            logger.debug(traceback.format_exc())
            return []
