"""
Embedding generation using OpenAI's text-embedding-3-small.
Handles batching and rate-limit retries.
"""

import logging
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from core.config import config
from core.models import DocumentChunk

logger = logging.getLogger(__name__)

# ── Client ────────────────────────────────────────────────────────────────────
_client: OpenAI | None = None


def get_openai_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=config.openai_api_key)
    return _client


# ── Embedding functions ───────────────────────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def embed_texts(texts: list[str], model: str | None = None) -> list[list[float]]:
    """Embed a batch of texts. Returns list of embedding vectors."""
    model = model or config.embedding.model
    client = get_openai_client()

    response = client.embeddings.create(
        input=texts,
        model=model,
    )
    embeddings = [item.embedding for item in response.data]
    logger.debug(f"Embedded {len(texts)} texts with {model}")
    return embeddings


def embed_query(query: str, model: str | None = None) -> list[float]:
    """Embed a single query string."""
    return embed_texts([query], model)[0]


def embed_chunks(
    chunks: list[DocumentChunk],
    batch_size: int = 100,
) -> list[DocumentChunk]:
    """
    Embed all chunks in batches. Mutates chunks in-place by setting .embedding.
    Returns the same list for chaining.
    """
    total = len(chunks)
    logger.info(f"Embedding {total} chunks in batches of {batch_size}...")

    for i in range(0, total, batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c.text for c in batch]
        embeddings = embed_texts(texts)

        for chunk, emb in zip(batch, embeddings):
            chunk.embedding = emb

        logger.info(f"  Embedded batch {i // batch_size + 1}/{(total + batch_size - 1) // batch_size}")

    return chunks
