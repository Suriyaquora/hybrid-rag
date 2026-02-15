"""
LLM answer generation layer.
Takes hybrid-retrieved context and synthesizes a grounded answer.
Tracks token usage and cost.
"""

import logging
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from core.config import config
from core.models import HybridResult, RetrievedContext, SourceType

logger = logging.getLogger(__name__)

# ── Cost tracking (approximate) ──────────────────────────────────────────────
COST_PER_1K = {
    "gpt-4o-mini": {"input": 0.000150, "output": 0.000600},
    "gpt-4o": {"input": 0.005, "output": 0.015},
}

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert research assistant powered by a Hybrid RAG system.
You have access to two types of retrieved context:

1. **Vector Context** (semantic similarity): Relevant text passages from documents.
2. **Graph Context** (knowledge graph): Structured entity-relation information extracted from documents.

Instructions:
- Answer the question using ONLY the provided context. Do not use prior knowledge.
- If the context is insufficient, say so clearly. Do NOT hallucinate.
- When citing information, indicate whether it came from document text or knowledge graph.
- Be precise, structured, and helpful.
- If entity relationships from the KG are relevant, mention them explicitly.
"""

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=config.openai_api_key)
    return _client


def _format_context(result: HybridResult, max_contexts: int = 8) -> str:
    """Format retrieved contexts into a structured prompt section."""
    parts = []

    # Separate by source for clarity
    vector_ctxs = [c for c in result.contexts if c.source in (SourceType.VECTOR, SourceType.HYBRID)]
    graph_ctxs = [c for c in result.contexts if c.source in (SourceType.GRAPH, SourceType.HYBRID)]

    if vector_ctxs:
        parts.append("=== DOCUMENT CONTEXT (Semantic Search) ===")
        for i, ctx in enumerate(vector_ctxs[:max_contexts]):
            source = ctx.metadata.get("file_name", ctx.metadata.get("source", "unknown"))
            parts.append(f"\n[Doc {i+1}] Source: {source} | Score: {ctx.score:.3f}")
            parts.append(ctx.text)

    if graph_ctxs:
        parts.append("\n=== KNOWLEDGE GRAPH CONTEXT (Entity Relations) ===")
        for i, ctx in enumerate(graph_ctxs[:max_contexts]):
            parts.append(f"\n[KG {i+1}] Score: {ctx.score:.3f}")
            parts.append(ctx.text)
            if ctx.entities:
                parts.append(f"  Entities: {', '.join(ctx.entities[:5])}")

    return "\n".join(parts)


def _estimate_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    rates = COST_PER_1K.get(model, COST_PER_1K["gpt-4o-mini"])
    return (input_tokens / 1000 * rates["input"]) + (output_tokens / 1000 * rates["output"])


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def generate_answer(result: HybridResult) -> HybridResult:
    """
    Generate an answer from the hybrid retrieval result.
    Mutates result.answer, result.total_tokens_used, result.cost_estimate.
    """
    client = _get_client()
    context = _format_context(result)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Context:\n{context}\n\n---\nQuestion: {result.query}\n\nAnswer:",
        },
    ]

    response = client.chat.completions.create(
        model=config.llm.model,
        temperature=config.llm.temperature,
        max_tokens=config.llm.max_tokens,
        messages=messages,
    )

    result.answer = response.choices[0].message.content.strip()

    # Token tracking
    usage = response.usage
    if usage:
        result.total_tokens_used = usage.total_tokens
        result.cost_estimate = _estimate_cost(
            usage.prompt_tokens, usage.completion_tokens, config.llm.model
        )

    logger.info(
        f"Generated answer: {len(result.answer)} chars, "
        f"{result.total_tokens_used} tokens, ${result.cost_estimate:.5f}"
    )
    return result


def generate_answer_streaming(result: HybridResult):
    """
    Stream the answer token by token. Yields text chunks.
    Use this for the Streamlit UI.
    """
    client = _get_client()
    context = _format_context(result)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Context:\n{context}\n\n---\nQuestion: {result.query}\n\nAnswer:",
        },
    ]

    stream = client.chat.completions.create(
        model=config.llm.model,
        temperature=config.llm.temperature,
        max_tokens=config.llm.max_tokens,
        messages=messages,
        stream=True,
    )

    full_answer = []
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            full_answer.append(delta.content)
            yield delta.content

    result.answer = "".join(full_answer)
