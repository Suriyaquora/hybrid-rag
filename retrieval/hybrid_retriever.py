"""
Hybrid Retriever: ALWAYS runs both Vector and Knowledge Graph pipelines.

Design philosophy:
  Every query hits BOTH pipelines. No conditional routing. No skipping.
  The graph pipeline uses a 3-layer fallback strategy to guarantee results:
    Layer 1: Direct entity match (highest quality)
    Layer 2: Fulltext keyword search across entities + chunk nodes
    Layer 3: Broad relation overview (lowest specificity, but always available)

  Results are fused via Reciprocal Rank Fusion (RRF).
"""

import logging
from core.config import config
from core.models import RetrievedContext, SourceType, HybridResult
from retrieval.vector_store import VectorStore
from retrieval.query_analyzer import analyze_query
from graph.kg_manager import KnowledgeGraph
from ingestion.embeddings import embed_query

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Dual-pipeline retriever. Every query ALWAYS goes through:
      1. FAISS vector search (semantic similarity)
      2. Neo4j knowledge graph (entity traversal + fulltext fallback)
    Results are merged via RRF and returned together.
    """

    def __init__(self, vector_store: VectorStore, knowledge_graph: KnowledgeGraph):
        self.vector_store = vector_store
        self.kg = knowledge_graph
        self.weight_vector = config.retrieval.hybrid_weight_vector
        self.weight_graph = config.retrieval.hybrid_weight_graph

    # ── Vector retrieval ──────────────────────────────────────────────────────

    def _vector_search(self, query: str, top_k: int | None = None) -> list[RetrievedContext]:
        """Embed query and search FAISS index. Always runs."""
        k = top_k or config.retrieval.vector_top_k
        try:
            query_emb = embed_query(query)
            results = self.vector_store.search(query_emb, top_k=k)
            logger.info(f"Vector pipeline: {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    # ── Graph retrieval (3-layer fallback) ────────────────────────────────────

    def _graph_search(
        self,
        query: str,
        entities: list[str],
        concepts: list[str],
        top_k: int | None = None,
    ) -> list[RetrievedContext]:
        """
        Knowledge Graph retrieval with 3-layer fallback strategy.
        ALWAYS returns results if the KG has any data.

        Layer 1: Direct entity match → subgraph traversal (highest quality)
        Layer 2: Fulltext keyword search across entities + chunks
        Layer 3: Broad relation overview (guaranteed if KG is populated)
        """
        k = top_k or config.retrieval.graph_top_k
        results: list[RetrievedContext] = []

        try:
            if not self.kg.verify_connection():
                logger.warning("Neo4j not connected — graph pipeline skipped")
                return results
        except Exception:
            logger.warning("Neo4j not reachable — graph pipeline skipped")
            return results

        # ── Layer 1: Direct entity match ──────────────────────────────────
        for entity in entities:
            try:
                entity_contexts = self.kg.get_entity_context(entity, depth=2)
                results.extend(entity_contexts)
            except Exception as e:
                logger.warning(f"Entity search failed for '{entity}': {e}")

        for concept in concepts:
            if concept.lower() not in [e.lower() for e in entities]:
                try:
                    concept_contexts = self.kg.get_entity_context(concept, depth=1)
                    results.extend(concept_contexts)
                except Exception as e:
                    logger.debug(f"No graph hit for concept '{concept}'")

        layer1_count = len(results)
        logger.info(f"  Graph Layer 1 (entity match): {layer1_count} results")

        # ── Layer 2: Fulltext keyword search (always runs) ────────────────
        try:
            fulltext_results = self.kg.fulltext_search(query, limit=k)
            results.extend(fulltext_results)
            logger.info(f"  Graph Layer 2 (fulltext): {len(fulltext_results)} results")
        except Exception as e:
            logger.warning(f"Fulltext graph search failed: {e}")

        # ── Layer 3: Broad relation overview (if still sparse) ────────────
        if len(results) < 3:
            try:
                overview = self.kg.get_all_relations_context(limit=k)
                results.extend(overview)
                logger.info(f"  Graph Layer 3 (overview): {len(overview)} results")
            except Exception as e:
                logger.debug(f"Relation overview failed: {e}")

        # Deduplicate by text content
        seen_texts = set()
        unique_results = []
        for r in results:
            text_key = r.text[:200]
            if text_key not in seen_texts:
                seen_texts.add(text_key)
                unique_results.append(r)

        logger.info(f"Graph pipeline total: {len(unique_results)} unique results")
        return unique_results[:k * 2]  # Allow more graph results since we fuse later

    # ── Reciprocal Rank Fusion ────────────────────────────────────────────────

    def _reciprocal_rank_fusion(
        self,
        vector_results: list[RetrievedContext],
        graph_results: list[RetrievedContext],
        k: int = 60,
    ) -> list[RetrievedContext]:
        """
        Merge results from BOTH pipelines using Reciprocal Rank Fusion.

        RRF score = Σ (weight_i / (k + rank_i))

        Why RRF over simple score averaging:
        - FAISS returns cosine similarity [0,1]
        - Neo4j returns traversal-depth-based scores [0,1] but on different scale
        - RRF is rank-based, so scale differences don't matter
        - Items appearing in BOTH lists get a natural boost
        """
        fused: dict[str, dict] = {}

        # Score vector results
        for rank, ctx in enumerate(vector_results):
            text_key = ctx.text[:200]
            rrf_score = self.weight_vector * (1.0 / (k + rank + 1))

            if text_key in fused:
                fused[text_key]["rrf_score"] += rrf_score
                fused[text_key]["sources"].add(SourceType.VECTOR)
            else:
                fused[text_key] = {
                    "context": ctx,
                    "rrf_score": rrf_score,
                    "sources": {SourceType.VECTOR},
                }

        # Score graph results
        for rank, ctx in enumerate(graph_results):
            text_key = ctx.text[:200]
            rrf_score = self.weight_graph * (1.0 / (k + rank + 1))

            if text_key in fused:
                fused[text_key]["rrf_score"] += rrf_score
                fused[text_key]["sources"].add(SourceType.GRAPH)
            else:
                fused[text_key] = {
                    "context": ctx,
                    "rrf_score": rrf_score,
                    "sources": {SourceType.GRAPH},
                }

        # Sort by RRF score
        sorted_items = sorted(fused.values(), key=lambda x: x["rrf_score"], reverse=True)

        results = []
        for item in sorted_items:
            ctx = item["context"]
            if len(item["sources"]) > 1:
                ctx.source = SourceType.HYBRID
            ctx.score = item["rrf_score"]
            ctx.metadata["fusion_sources"] = [s.value for s in item["sources"]]
            results.append(ctx)

        return results

    # ── Main retrieval entry point ────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        vector_top_k: int | None = None,
        graph_top_k: int | None = None,
    ) -> HybridResult:
        """
        Full hybrid retrieval — BOTH pipelines ALWAYS run.

        Pipeline:
        1. Analyze query → extract entities + concepts (for graph Layer 1)
        2. Vector search (ALWAYS)
        3. Graph search with 3-layer fallback (ALWAYS)
        4. Fuse via RRF
        5. Return HybridResult with separate + fused contexts
        """
        # Step 1: Query analysis (for entity extraction, NOT for routing)
        try:
            analysis = analyze_query(query)
            entities = analysis.get("entities", [])
            concepts = analysis.get("concepts", [])
        except Exception as e:
            logger.warning(f"Query analysis failed, proceeding with empty entities: {e}")
            entities = []
            concepts = [query]  # Fall back to using the whole query as a concept

        # Step 2: Vector search — ALWAYS runs
        vector_results = self._vector_search(query, vector_top_k)

        # Step 3: Graph search — ALWAYS runs (3-layer fallback ensures results)
        graph_results = self._graph_search(query, entities, concepts, graph_top_k)

        # Step 4: Fuse via RRF
        if vector_results and graph_results:
            fused = self._reciprocal_rank_fusion(vector_results, graph_results)
        elif vector_results:
            fused = vector_results
        elif graph_results:
            fused = graph_results
        else:
            fused = []

        logger.info(
            f"Hybrid retrieval complete: "
            f"{len(vector_results)} vector + {len(graph_results)} graph "
            f"→ {len(fused)} fused results"
        )

        return HybridResult(
            contexts=fused,
            vector_contexts=vector_results,
            graph_contexts=graph_results,
            query=query,
        )
