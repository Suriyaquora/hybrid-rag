"""
Hybrid RAG Pipeline Orchestrator.
Provides a single clean interface for:
  - Document ingestion (load → chunk → embed → extract entities → store)
  - Querying (analyze → retrieve → fuse → generate)
"""

import logging
import os
from pathlib import Path

from core.config import config
from core.models import DocumentChunk, Entity, Relation, HybridResult
from ingestion.loader import load_and_chunk, chunk_documents, load_single_document
from ingestion.embeddings import embed_chunks
from ingestion.entity_extractor import extract_from_chunks
from retrieval.vector_store import VectorStore
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.generator import generate_answer, generate_answer_streaming
from graph.kg_manager import KnowledgeGraph

logger = logging.getLogger(__name__)


class HybridRAGPipeline:
    """
    End-to-end Hybrid RAG pipeline combining:
    - FAISS vector search (semantic)
    - Neo4j knowledge graph (structural)
    - Reciprocal Rank Fusion (merging)
    - GPT-4o-mini (generation)
    """

    def __init__(self):
        self.vector_store = VectorStore()
        self.kg = KnowledgeGraph()
        self.retriever: HybridRetriever | None = None

        # State tracking
        self.chunks: list[DocumentChunk] = []
        self.entities: list[Entity] = []
        self.relations: list[Relation] = []
        self.is_initialized = False

    # ── Initialization ────────────────────────────────────────────────────────

    def initialize(self) -> dict:
        """
        Initialize connections and load existing data if available.
        Returns status dict.
        """
        status = {
            "vector_store": False,
            "knowledge_graph": False,
            "retriever": False,
        }

        # Try to load existing FAISS index
        if self.vector_store.load():
            status["vector_store"] = True
            logger.info(f"Loaded existing vector store: {self.vector_store.size} vectors")

        # Test Neo4j connection
        try:
            if self.kg.verify_connection():
                self.kg.setup_schema()
                status["knowledge_graph"] = True
                logger.info("Neo4j connected and schema ready")
        except Exception as e:
            logger.warning(f"Neo4j not available: {e}. KG features will be disabled.")

        # Create retriever
        self.retriever = HybridRetriever(self.vector_store, self.kg)
        status["retriever"] = True

        self.is_initialized = True
        return status

    # ── Ingestion pipeline ────────────────────────────────────────────────────

    def ingest_documents(
        self,
        source: str,
        progress_callback=None,
    ) -> dict:
        """
        Full ingestion pipeline:
        1. Load and chunk documents
        2. Generate embeddings
        3. Build FAISS index
        4. Extract entities and relations
        5. Populate Neo4j knowledge graph
        
        Args:
            source: File path or directory path
            progress_callback: fn(step_name, current, total) for UI updates
            
        Returns:
            Statistics dict
        """
        stats = {
            "documents_loaded": 0,
            "chunks_created": 0,
            "embeddings_generated": 0,
            "entities_extracted": 0,
            "relations_extracted": 0,
            "vector_store_size": 0,
            "kg_entities": 0,
            "kg_relations": 0,
        }

        def _progress(step, current, total):
            if progress_callback:
                progress_callback(step, current, total)

        # Step 1: Load and chunk
        _progress("Loading documents", 0, 1)
        self.chunks = load_and_chunk(source)
        stats["chunks_created"] = len(self.chunks)
        _progress("Loading documents", 1, 1)

        if not self.chunks:
            logger.warning("No chunks created from source")
            return stats

        # Step 2: Embed chunks
        _progress("Generating embeddings", 0, len(self.chunks))
        self.chunks = embed_chunks(self.chunks)
        stats["embeddings_generated"] = len(self.chunks)
        _progress("Generating embeddings", len(self.chunks), len(self.chunks))

        # Step 3: Build FAISS index
        _progress("Building vector index", 0, 1)
        self.vector_store.build_index(self.chunks)
        self.vector_store.save()
        stats["vector_store_size"] = self.vector_store.size
        _progress("Building vector index", 1, 1)

        # Step 4: Extract entities & relations
        _progress("Extracting entities", 0, len(self.chunks))

        def _entity_progress(current, total):
            _progress("Extracting entities", current, total)

        self.entities, self.relations = extract_from_chunks(
            self.chunks, progress_callback=_entity_progress
        )
        stats["entities_extracted"] = len(self.entities)
        stats["relations_extracted"] = len(self.relations)

        # Step 5: Populate Neo4j (if connected)
        try:
            if self.kg.verify_connection():
                _progress("Building knowledge graph", 0, 3)

                self.kg.add_entities(self.entities)
                stats["kg_entities"] = len(self.entities)
                _progress("Building knowledge graph", 1, 3)

                self.kg.add_relations(self.relations)
                stats["kg_relations"] = len(self.relations)
                _progress("Building knowledge graph", 2, 3)

                # Add chunk nodes and link entities
                chunk_dicts = [
                    {
                        "chunk_id": c.chunk_id,
                        "text": c.text,
                        "source": c.metadata.get("source", ""),
                        "doc_id": c.doc_id,
                    }
                    for c in self.chunks
                ]
                self.kg.add_chunk_nodes(chunk_dicts)

                for entity in self.entities:
                    if entity.source_chunks:
                        self.kg.link_entities_to_chunks(entity.name, entity.source_chunks)

                _progress("Building knowledge graph", 3, 3)
        except Exception as e:
            logger.warning(f"KG population skipped (Neo4j unavailable): {e}")

        # Recreate retriever with updated stores
        self.retriever = HybridRetriever(self.vector_store, self.kg)

        logger.info(f"Ingestion complete: {stats}")
        return stats

    def ingest_single_file(self, file_path: str, progress_callback=None) -> dict:
        """Convenience method for ingesting a single file."""
        return self.ingest_documents(file_path, progress_callback)

    # ── Query pipeline ────────────────────────────────────────────────────────

    def query(self, question: str) -> HybridResult:
        """
        Full query pipeline:
        1. Analyze query → entities + intent
        2. Hybrid retrieval (vector + graph + RRF)
        3. LLM answer generation
        
        Returns HybridResult with answer, contexts, and metadata.
        """
        if not self.retriever:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

        # Retrieve
        result = self.retriever.retrieve(question)

        # Generate answer
        result = generate_answer(result)

        return result

    def query_streaming(self, question: str):
        """
        Streaming version of query(). Yields (chunk_text, result) tuples.
        The final yield has the complete HybridResult.
        """
        if not self.retriever:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

        result = self.retriever.retrieve(question)

        for token in generate_answer_streaming(result):
            yield token, result

    # ── Utilities ─────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Get current system statistics."""
        stats = {
            "vector_store_size": self.vector_store.size,
            "chunks_in_memory": len(self.chunks),
            "entities_in_memory": len(self.entities),
            "relations_in_memory": len(self.relations),
        }

        try:
            if self.kg.verify_connection():
                kg_stats = self.kg.get_graph_stats()
                stats.update({f"kg_{k}": v for k, v in kg_stats.items()})
        except:
            stats["kg_status"] = "disconnected"

        return stats

    def clear_all(self) -> None:
        """Clear all stored data. Use with caution."""
        self.vector_store.clear()
        try:
            self.kg.clear_graph()
        except:
            pass
        self.chunks = []
        self.entities = []
        self.relations = []
        logger.warning("All data cleared")
