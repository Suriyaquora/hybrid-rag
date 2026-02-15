"""
Neo4j Knowledge Graph manager.
Handles: graph construction, entity/relation storage, Cypher-based retrieval,
and subgraph extraction for RAG context.
"""

import logging
from typing import Any
from neo4j import GraphDatabase, Driver

from core.config import config
from core.models import Entity, Relation, RetrievedContext, SourceType

logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """Neo4j-backed Knowledge Graph for the RAG pipeline."""

    def __init__(self, uri: str | None = None, user: str | None = None, password: str | None = None):
        self.uri = uri or config.neo4j.uri
        self.user = user or config.neo4j.username
        self.password = password or config.neo4j.password
        self._driver: Driver | None = None

    # ── Connection management ─────────────────────────────────────────────────

    @property
    def driver(self) -> Driver:
        if self._driver is None:
            self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        return self._driver

    def verify_connection(self) -> bool:
        """Test that Neo4j is reachable."""
        try:
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info("Neo4j connection verified")
            return True
        except Exception as e:
            logger.error(f"Neo4j connection failed: {e}")
            return False

    def close(self) -> None:
        if self._driver:
            self._driver.close()
            self._driver = None

    # ── Schema setup ──────────────────────────────────────────────────────────

    def setup_schema(self) -> None:
        """Create indexes and constraints for optimal performance."""
        queries = [
            "CREATE INDEX entity_name_idx IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX entity_label_idx IF NOT EXISTS FOR (e:Entity) ON (e.label)",
            "CREATE CONSTRAINT entity_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
            "CREATE INDEX chunk_id_idx IF NOT EXISTS FOR (c:Chunk) ON (c.chunk_id)",
        ]
        with self.driver.session() as session:
            for q in queries:
                try:
                    session.run(q)
                except Exception as e:
                    logger.debug(f"Schema query note: {e}")
        logger.info("Neo4j schema setup complete")

    # ── Data ingestion ────────────────────────────────────────────────────────

    def add_entities(self, entities: list[Entity]) -> int:
        """Upsert entities into the graph. Returns count of entities added."""
        count = 0
        with self.driver.session() as session:
            for entity in entities:
                try:
                    session.run(
                        """
                        MERGE (e:Entity {name: $name})
                        SET e.label = $label,
                            e.description = $description,
                            e.source_chunks = $source_chunks
                        """,
                        name=entity.name,
                        label=entity.label,
                        description=entity.properties.get("description", ""),
                        source_chunks=entity.source_chunks,
                    )
                    # Also add a label-specific node label (e.g., :PERSON, :ORGANIZATION)
                    session.run(
                        f"MATCH (e:Entity {{name: $name}}) SET e:{entity.label}",
                        name=entity.name,
                    )
                    count += 1
                except Exception as e:
                    logger.warning(f"Failed to add entity {entity.name}: {e}")

        logger.info(f"Added {count} entities to graph")
        return count

    def add_relations(self, relations: list[Relation]) -> int:
        """Add relations between entities. Returns count of relations added."""
        count = 0
        with self.driver.session() as session:
            for rel in relations:
                try:
                    # Use APOC-free approach: dynamic rel types via string formatting
                    # (safe here because relation_type is from our extraction, not user input)
                    rel_type = rel.relation_type.upper().replace(" ", "_")
                    session.run(
                        f"""
                        MATCH (s:Entity {{name: $source}})
                        MATCH (t:Entity {{name: $target}})
                        MERGE (s)-[r:{rel_type}]->(t)
                        SET r.description = $description,
                            r.confidence = $confidence,
                            r.source_chunks = $source_chunks
                        """,
                        source=rel.source,
                        target=rel.target,
                        description=rel.properties.get("description", ""),
                        confidence=rel.confidence,
                        source_chunks=rel.source_chunks,
                    )
                    count += 1
                except Exception as e:
                    logger.warning(f"Failed to add relation {rel.source}->{rel.target}: {e}")

        logger.info(f"Added {count} relations to graph")
        return count

    def add_chunk_nodes(self, chunks: list[dict]) -> int:
        """Add chunk nodes and link them to their entities."""
        count = 0
        with self.driver.session() as session:
            for chunk in chunks:
                try:
                    session.run(
                        """
                        MERGE (c:Chunk {chunk_id: $chunk_id})
                        SET c.text = $text,
                            c.source = $source,
                            c.doc_id = $doc_id
                        """,
                        chunk_id=chunk["chunk_id"],
                        text=chunk["text"][:1000],  # Store first 1000 chars
                        source=chunk.get("source", ""),
                        doc_id=chunk.get("doc_id", ""),
                    )
                    count += 1
                except Exception as e:
                    logger.warning(f"Failed to add chunk {chunk.get('chunk_id')}: {e}")
        return count

    def link_entities_to_chunks(self, entity_name: str, chunk_ids: list[str]) -> None:
        """Create MENTIONED_IN relationships between entities and chunks."""
        with self.driver.session() as session:
            for cid in chunk_ids:
                session.run(
                    """
                    MATCH (e:Entity {name: $name})
                    MATCH (c:Chunk {chunk_id: $chunk_id})
                    MERGE (e)-[:MENTIONED_IN]->(c)
                    """,
                    name=entity_name,
                    chunk_id=cid,
                )

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def get_entity_context(self, entity_name: str, depth: int = 2) -> list[RetrievedContext]:
        """
        Get context around an entity: its properties, neighbors, 
        and associated chunk texts.
        """
        results = []
        with self.driver.session() as session:
            # Get entity + immediate relations (up to `depth` hops)
            records = session.run(
                """
                MATCH (e:Entity)
                WHERE toLower(e.name) CONTAINS toLower($name)
                OPTIONAL MATCH path = (e)-[r*1..2]-(neighbor:Entity)
                RETURN e.name AS entity_name,
                       e.label AS entity_label,
                       e.description AS description,
                       [rel IN relationships(path) | 
                        {type: type(rel), 
                         source: startNode(rel).name,
                         target: endNode(rel).name,
                         desc: rel.description}
                       ] AS relations,
                       [n IN nodes(path) WHERE n:Entity AND n <> e | 
                        {name: n.name, label: n.label, desc: n.description}
                       ] AS neighbors
                LIMIT 20
                """,
                name=entity_name,
            ).data()

            if not records:
                return results

            # Build context text from graph structure
            seen_triples = set()
            entity_info = []
            relation_info = []
            neighbor_names = []

            for record in records:
                ename = record.get("entity_name", "")
                elabel = record.get("entity_label", "")
                edesc = record.get("description", "")

                if ename and f"entity:{ename}" not in seen_triples:
                    entity_info.append(f"{ename} ({elabel}): {edesc}")
                    seen_triples.add(f"entity:{ename}")

                for rel in (record.get("relations") or []):
                    if rel:
                        triple_key = f"{rel.get('source')}-{rel.get('type')}-{rel.get('target')}"
                        if triple_key not in seen_triples:
                            relation_info.append(
                                f"{rel.get('source')} --[{rel.get('type')}]--> {rel.get('target')}"
                                f" ({rel.get('desc', '')})"
                            )
                            seen_triples.add(triple_key)

                for n in (record.get("neighbors") or []):
                    if n and n.get("name"):
                        neighbor_names.append(n["name"])

            # Compose context text
            context_parts = []
            if entity_info:
                context_parts.append("Entities:\n" + "\n".join(entity_info))
            if relation_info:
                context_parts.append("Relations:\n" + "\n".join(relation_info))

            if context_parts:
                context_text = "\n\n".join(context_parts)
                results.append(
                    RetrievedContext(
                        text=context_text,
                        source=SourceType.GRAPH,
                        score=1.0,
                        metadata={"query_entity": entity_name},
                        entities=[entity_name] + neighbor_names[:10],
                        relations=[r for r in relation_info[:10]],
                    )
                )

            # Also get associated chunk texts
            chunk_records = session.run(
                """
                MATCH (e:Entity)-[:MENTIONED_IN]->(c:Chunk)
                WHERE toLower(e.name) CONTAINS toLower($name)
                RETURN c.text AS text, c.chunk_id AS chunk_id, c.source AS source
                LIMIT 5
                """,
                name=entity_name,
            ).data()

            for cr in chunk_records:
                if cr.get("text"):
                    results.append(
                        RetrievedContext(
                            text=cr["text"],
                            source=SourceType.GRAPH,
                            score=0.8,
                            metadata={
                                "chunk_id": cr.get("chunk_id", ""),
                                "source": cr.get("source", ""),
                                "retrieval_method": "entity_to_chunk",
                            },
                            entities=[entity_name],
                        )
                    )

        return results

    def fulltext_search(self, query: str, limit: int = 5) -> list[RetrievedContext]:
        """
        Broad keyword search across ALL entity names and descriptions.
        Used as a fallback when entity extraction finds nothing specific.
        This ensures the graph pipeline ALWAYS returns results if the KG
        has any relevant data.
        """
        results = []
        # Split query into individual keywords for broader matching
        keywords = [w.strip() for w in query.split() if len(w.strip()) > 2]

        with self.driver.session() as session:
            # Strategy 1: Search entity names and descriptions for any keyword match
            for keyword in keywords:
                records = session.run(
                    """
                    MATCH (e:Entity)
                    WHERE toLower(e.name) CONTAINS toLower($keyword)
                       OR toLower(e.description) CONTAINS toLower($keyword)
                    OPTIONAL MATCH (e)-[r]-(neighbor:Entity)
                    RETURN e.name AS entity_name,
                           e.label AS entity_label,
                           e.description AS description,
                           collect(DISTINCT {
                               name: neighbor.name,
                               label: neighbor.label,
                               rel_type: type(r)
                           })[..5] AS neighbors
                    LIMIT $limit
                    """,
                    keyword=keyword,
                    limit=limit,
                ).data()

                for record in records:
                    ename = record.get("entity_name", "")
                    elabel = record.get("entity_label", "")
                    edesc = record.get("description", "")
                    neighbors = record.get("neighbors", [])

                    # Build context text
                    parts = [f"{ename} ({elabel}): {edesc}"]
                    neighbor_names = []
                    for n in neighbors:
                        if n and n.get("name"):
                            parts.append(
                                f"  → {n.get('rel_type', 'RELATED_TO')} → {n['name']} ({n.get('label', '')})"
                            )
                            neighbor_names.append(n["name"])

                    results.append(
                        RetrievedContext(
                            text="\n".join(parts),
                            source=SourceType.GRAPH,
                            score=0.6,  # Lower score than direct entity match
                            metadata={
                                "retrieval_method": "fulltext_search",
                                "matched_keyword": keyword,
                            },
                            entities=[ename] + neighbor_names,
                        )
                    )

            # Strategy 2: Search chunk nodes stored in the graph
            for keyword in keywords[:3]:  # Limit to top 3 keywords for chunk search
                chunk_records = session.run(
                    """
                    MATCH (c:Chunk)
                    WHERE toLower(c.text) CONTAINS toLower($keyword)
                    OPTIONAL MATCH (e:Entity)-[:MENTIONED_IN]->(c)
                    RETURN c.text AS text, c.chunk_id AS chunk_id, c.source AS source,
                           collect(DISTINCT e.name)[..5] AS entities
                    LIMIT $limit
                    """,
                    keyword=keyword,
                    limit=limit,
                ).data()

                for cr in chunk_records:
                    if cr.get("text"):
                        results.append(
                            RetrievedContext(
                                text=cr["text"],
                                source=SourceType.GRAPH,
                                score=0.5,
                                metadata={
                                    "chunk_id": cr.get("chunk_id", ""),
                                    "source": cr.get("source", ""),
                                    "retrieval_method": "graph_chunk_search",
                                    "matched_keyword": keyword,
                                },
                                entities=cr.get("entities", []),
                            )
                        )

        # Deduplicate
        seen = set()
        unique = []
        for r in results:
            key = r.text[:200]
            if key not in seen:
                seen.add(key)
                unique.append(r)

        logger.info(f"Fulltext graph search returned {len(unique)} results for query: {query[:50]}")
        return unique[:limit]

    def get_all_relations_context(self, limit: int = 10) -> list[RetrievedContext]:
        """
        Get a broad overview of all relations in the graph.
        Useful as additional KG context when no specific entities match.
        """
        results = []
        with self.driver.session() as session:
            records = session.run(
                """
                MATCH (s:Entity)-[r]->(t:Entity)
                RETURN s.name AS source, type(r) AS rel_type, t.name AS target,
                       r.description AS description, r.confidence AS confidence
                ORDER BY r.confidence DESC
                LIMIT $limit
                """,
                limit=limit,
            ).data()

            if records:
                lines = []
                entity_names = set()
                for rec in records:
                    src = rec.get("source", "")
                    tgt = rec.get("target", "")
                    rel = rec.get("rel_type", "")
                    desc = rec.get("description", "")
                    lines.append(f"{src} --[{rel}]--> {tgt}" + (f" ({desc})" if desc else ""))
                    entity_names.add(src)
                    entity_names.add(tgt)

                results.append(
                    RetrievedContext(
                        text="Knowledge Graph Relations:\n" + "\n".join(lines),
                        source=SourceType.GRAPH,
                        score=0.4,
                        metadata={"retrieval_method": "all_relations_overview"},
                        entities=list(entity_names)[:20],
                    )
                )

        return results

    def query_by_relation(self, relation_type: str, limit: int = 10) -> list[dict]:
        """Query the graph for all triples of a given relation type."""
        with self.driver.session() as session:
            records = session.run(
                f"""
                MATCH (s:Entity)-[r:{relation_type}]->(t:Entity)
                RETURN s.name AS source, type(r) AS relation, t.name AS target,
                       r.description AS description
                LIMIT $limit
                """,
                limit=limit,
            ).data()
        return records

    def get_all_entities(self, limit: int = 100) -> list[dict]:
        """Get all entities in the graph."""
        with self.driver.session() as session:
            records = session.run(
                """
                MATCH (e:Entity)
                RETURN e.name AS name, e.label AS label, e.description AS description
                ORDER BY e.name
                LIMIT $limit
                """,
                limit=limit,
            ).data()
        return records

    def get_graph_stats(self) -> dict:
        """Get basic statistics about the knowledge graph."""
        with self.driver.session() as session:
            entity_count = session.run("MATCH (e:Entity) RETURN count(e) AS c").single()["c"]
            rel_count = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
            chunk_count = session.run("MATCH (c:Chunk) RETURN count(c) AS c").single()["c"]

            # Get relation types
            rel_types = session.run(
                "MATCH ()-[r]->() RETURN DISTINCT type(r) AS type, count(r) AS count ORDER BY count DESC"
            ).data()

            # Get label distribution
            label_dist = session.run(
                "MATCH (e:Entity) RETURN e.label AS label, count(e) AS count ORDER BY count DESC"
            ).data()

        return {
            "entity_count": entity_count,
            "relation_count": rel_count,
            "chunk_count": chunk_count,
            "relation_types": rel_types,
            "label_distribution": label_dist,
        }

    def clear_graph(self) -> None:
        """Delete all nodes and relationships. Use with caution."""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        logger.warning("Knowledge graph cleared")
