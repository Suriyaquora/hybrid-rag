"""
FAISS vector store wrapper.
Handles indexing, persistence, and similarity search.
"""

import os
import json
import pickle
import logging
import numpy as np
import faiss

from core.config import config
from core.models import DocumentChunk, RetrievedContext, SourceType

logger = logging.getLogger(__name__)


class VectorStore:
    """FAISS-based vector store with metadata persistence."""

    def __init__(self, index_path: str | None = None):
        self.index_path = index_path or config.faiss_index_path
        self.index: faiss.IndexFlatIP | None = None  # Inner product (cosine after normalization)
        self.chunks: list[DocumentChunk] = []
        self.dimension: int = config.embedding.dimensions

    @property
    def size(self) -> int:
        return self.index.ntotal if self.index else 0

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """L2-normalize vectors for cosine similarity via inner product."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1  # avoid division by zero
        return vectors / norms

    def build_index(self, chunks: list[DocumentChunk]) -> None:
        """Build FAISS index from embedded chunks."""
        if not chunks:
            raise ValueError("No chunks provided")

        embeddings = []
        valid_chunks = []
        for chunk in chunks:
            if chunk.embedding is not None:
                embeddings.append(chunk.embedding)
                valid_chunks.append(chunk)

        if not embeddings:
            raise ValueError("No chunks have embeddings. Run embed_chunks first.")

        vectors = np.array(embeddings, dtype=np.float32)
        vectors = self._normalize(vectors)

        self.dimension = vectors.shape[1]
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(vectors)
        self.chunks = valid_chunks

        logger.info(f"Built FAISS index: {self.index.ntotal} vectors, dim={self.dimension}")

    def search(
        self,
        query_embedding: list[float],
        top_k: int | None = None,
    ) -> list[RetrievedContext]:
        """Search for similar chunks. Returns scored RetrievedContext objects."""
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Vector store is empty")
            return []

        k = min(top_k or config.retrieval.vector_top_k, self.index.ntotal)
        query_vec = np.array([query_embedding], dtype=np.float32)
        query_vec = self._normalize(query_vec)

        scores, indices = self.index.search(query_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]
            results.append(
                RetrievedContext(
                    text=chunk.text,
                    source=SourceType.VECTOR,
                    score=float(score),
                    metadata=chunk.metadata,
                )
            )

        return results

    def save(self) -> None:
        """Persist index and metadata to disk."""
        os.makedirs(self.index_path, exist_ok=True)
        if self.index:
            faiss.write_index(self.index, os.path.join(self.index_path, "index.faiss"))

        # Save chunk metadata (without embeddings to save space)
        meta = []
        for c in self.chunks:
            meta.append({
                "text": c.text,
                "doc_id": c.doc_id,
                "chunk_id": c.chunk_id,
                "metadata": c.metadata,
            })
        with open(os.path.join(self.index_path, "chunks.json"), "w") as f:
            json.dump(meta, f)

        logger.info(f"Saved vector store to {self.index_path}")

    def load(self) -> bool:
        """Load index and metadata from disk. Returns True if successful."""
        index_file = os.path.join(self.index_path, "index.faiss")
        meta_file = os.path.join(self.index_path, "chunks.json")

        if not os.path.exists(index_file) or not os.path.exists(meta_file):
            logger.info("No existing vector store found")
            return False

        self.index = faiss.read_index(index_file)

        with open(meta_file, "r") as f:
            meta = json.load(f)

        self.chunks = [
            DocumentChunk(
                text=m["text"],
                doc_id=m["doc_id"],
                chunk_id=m["chunk_id"],
                metadata=m["metadata"],
            )
            for m in meta
        ]

        logger.info(f"Loaded vector store: {self.index.ntotal} vectors")
        return True

    def clear(self) -> None:
        """Clear the index."""
        self.index = None
        self.chunks = []
