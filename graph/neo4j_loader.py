"""
Neo4j Knowledge Graph Loader

Reads validated JSON extraction outputs and creates nodes
and relationships in Neo4j. Fully independent from the
extraction pipeline -- operates on saved JSON files only.

Supports:
  - Clear Graph Mode: Wipe and rebuild
  - Append Mode: Add to existing graph
  - Batch loading for efficiency
  - Connection pooling via the official neo4j driver

Node types created:
  - Event (source event from XML)
  - Entity (extracted CTI entity)

Relationship types created:
  - HAS_ENTITY (Event -> Entity)
  - Dynamic relation types from extraction (Entity -> Entity)
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class Neo4jLoader:
    """Loads CTI extraction results into a Neo4j graph database."""

    def __init__(
        self,
        uri: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
    ) -> None:
        from config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE

        self.uri = uri or NEO4J_URI
        self.username = username or NEO4J_USERNAME
        self.password = password or NEO4J_PASSWORD
        self.database = database or NEO4J_DATABASE
        self._driver = None

        logger.info(
            "Neo4jLoader initialized (uri=%s, database=%s)",
            self.uri,
            self.database,
        )

    def connect(self) -> None:
        """Establish connection to Neo4j."""
        try:
            from neo4j import GraphDatabase

            self._driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password),
            )
            # Verify connectivity
            self._driver.verify_connectivity()
            logger.info("Connected to Neo4j at %s", self.uri)
        except Exception as e:
            logger.error("Failed to connect to Neo4j: %s", e)
            raise ConnectionError(f"Neo4j connection failed: {e}") from e

    def close(self) -> None:
        """Close Neo4j connection."""
        if self._driver:
            self._driver.close()
            self._driver = None
            logger.info("Neo4j connection closed")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def clear_graph(self) -> int:
        """
        Remove all nodes and relationships from the graph.

        Returns:
            Number of nodes deleted.
        """
        self._ensure_connected()

        with self._driver.session(database=self.database) as session:
            # Count before
            count_result = session.run("MATCH (n) RETURN count(n) as count")
            node_count = count_result.single()["count"]

            # Delete all
            session.run("MATCH (n) DETACH DELETE n")
            logger.info("Cleared graph: %d nodes deleted", node_count)

            return node_count

    def create_indexes(self) -> None:
        """Create indexes for efficient querying."""
        self._ensure_connected()

        indexes = [
            "CREATE INDEX event_id_idx IF NOT EXISTS FOR (e:Event) ON (e.event_id)",
            "CREATE INDEX entity_name_idx IF NOT EXISTS FOR (e:Entity) ON (e.canonical_name)",
            "CREATE INDEX entity_type_idx IF NOT EXISTS FOR (e:Entity) ON (e.type)",
        ]

        with self._driver.session(database=self.database) as session:
            for query in indexes:
                try:
                    session.run(query)
                except Exception as e:
                    logger.debug("Index creation note: %s", e)

        logger.info("Indexes created/verified")

    def load_json(self, json_path: str, append: bool = True) -> Dict[str, int]:
        """
        Load extraction results from a JSON file into Neo4j.

        Args:
            json_path: Path to the experiment output JSON file.
            append: If True, add to existing graph. If False, clear first.

        Returns:
            Statistics dict with nodes_created, relationships_created.
        """
        self._ensure_connected()
        json_path = Path(json_path)

        if not json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_path}")

        # Load JSON
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not append:
            self.clear_graph()

        # Create indexes
        self.create_indexes()

        metadata = data.get("experiment_metadata", {})
        results = data.get("results", [])

        logger.info(
            "Loading %d event results from %s (method=%s, model=%s)",
            len(results),
            json_path.name,
            metadata.get("method", "unknown"),
            metadata.get("model", "unknown"),
        )

        stats = {"events_created": 0, "entities_created": 0, "relations_created": 0}

        for result in results:
            if result.get("status") == "error":
                continue

            event_stats = self._load_event_result(result, metadata)
            stats["events_created"] += event_stats["events"]
            stats["entities_created"] += event_stats["entities"]
            stats["relations_created"] += event_stats["relations"]

        logger.info(
            "Load complete: %d events, %d entities, %d relations",
            stats["events_created"],
            stats["entities_created"],
            stats["relations_created"],
        )

        return stats

    def _load_event_result(self, result: Dict, metadata: Dict) -> Dict[str, int]:
        """
        Load a single event's extraction results into Neo4j.

        Creates:
        - An Event node
        - Entity nodes for each extracted entity
        - HAS_ENTITY relationships (Event -> Entity)
        - Extracted relationships (Entity -> Entity)
        """
        stats = {"events": 0, "entities": 0, "relations": 0}

        event_id = result.get("event_id", "")
        file_source = result.get("file_source", "")
        extraction = result.get("extraction", {})
        entities = extraction.get("entities", [])
        relations = extraction.get("relations", [])

        with self._driver.session(database=self.database) as session:
            # Create Event node
            session.run(
                """
                MERGE (e:Event {event_id: $event_id})
                SET e.file_source = $file_source,
                    e.method = $method,
                    e.model = $model,
                    e.processing_time = $processing_time,
                    e.status = $status
                """,
                event_id=event_id,
                file_source=file_source,
                method=metadata.get("method", ""),
                model=metadata.get("model", ""),
                processing_time=result.get("processing_time_seconds", 0),
                status=result.get("status", ""),
            )
            stats["events"] = 1

            # Create Entity nodes and HAS_ENTITY relationships
            for entity in entities:
                if not isinstance(entity, dict):
                    continue

                entity_text = entity.get("text", "")
                entity_type = entity.get("type", "unknown")
                canonical = entity.get("canonical_name", entity_text)
                confidence = entity.get("confidence", 0.0)

                if not entity_text:
                    continue

                session.run(
                    """
                    MERGE (ent:Entity {canonical_name: $canonical_name, type: $type})
                    SET ent.text = $text,
                        ent.confidence = $confidence

                    WITH ent
                    MATCH (ev:Event {event_id: $event_id})
                    MERGE (ev)-[:HAS_ENTITY]->(ent)
                    """,
                    canonical_name=canonical,
                    type=entity_type,
                    text=entity_text,
                    confidence=confidence,
                    event_id=event_id,
                )
                stats["entities"] += 1

            # Create extracted relationships
            for relation in relations:
                if not isinstance(relation, dict):
                    continue

                head = relation.get("head", "")
                tail = relation.get("tail", "")
                rel_type = relation.get("relation", "RELATED_TO")
                confidence = relation.get("confidence", 0.0)
                evidence = relation.get("evidence", "")
                time_context = relation.get("time", "")

                if not head or not tail:
                    continue

                # Sanitize relationship type for Neo4j (only alphanumeric + underscore)
                safe_rel_type = "".join(
                    c if c.isalnum() or c == "_" else "_" for c in rel_type.upper()
                )
                if not safe_rel_type:
                    safe_rel_type = "RELATED_TO"

                # Use APOC-free approach: create with generic type, store specific type as property
                session.run(
                    f"""
                    MERGE (h:Entity {{canonical_name: $head}})
                    MERGE (t:Entity {{canonical_name: $tail}})
                    MERGE (h)-[r:{safe_rel_type}]->(t)
                    SET r.confidence = $confidence,
                        r.evidence = $evidence,
                        r.time = $time,
                        r.event_id = $event_id
                    """,
                    head=head,
                    tail=tail,
                    confidence=confidence,
                    evidence=evidence,
                    time=time_context,
                    event_id=event_id,
                )
                stats["relations"] += 1

        return stats

    def get_graph_stats(self) -> Dict[str, Any]:
        """
        Get current graph statistics.

        Returns:
            Dictionary with node counts, relationship counts, etc.
        """
        self._ensure_connected()

        with self._driver.session(database=self.database) as session:
            # Node counts by label
            node_result = session.run(
                """
                MATCH (n)
                RETURN labels(n) as labels, count(n) as count
                ORDER BY count DESC
                """
            )
            node_counts = {
                str(record["labels"]): record["count"]
                for record in node_result
            }

            # Total relationship count
            rel_result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
            rel_count = rel_result.single()["count"]

            # Relationship types
            rel_types_result = session.run(
                "MATCH ()-[r]->() RETURN type(r) as type, count(r) as count ORDER BY count DESC"
            )
            rel_types = {
                record["type"]: record["count"]
                for record in rel_types_result
            }

        return {
            "node_counts": node_counts,
            "total_relationships": rel_count,
            "relationship_types": rel_types,
        }

    def _ensure_connected(self) -> None:
        """Raise if not connected."""
        if self._driver is None:
            raise ConnectionError(
                "Not connected to Neo4j. Call connect() first."
            )
