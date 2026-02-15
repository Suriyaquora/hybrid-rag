"""
Query analyzer: detects entities in user queries to route to the KG.
Uses a lightweight OpenAI call to extract entity mentions from the query,
then fuzzy-matches them against the KG.
"""

import json
import logging
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from core.config import config

logger = logging.getLogger(__name__)

QUERY_ANALYSIS_PROMPT = """Analyze this user query and extract:
1. Named entities mentioned (people, organizations, technologies, products, locations, concepts).
2. The intent of the query (factual, relational, comparative, exploratory).
3. Key concepts/topics that should be searched in a knowledge base.

Be thorough — extract ALL possible entity names and concept keywords,
including implied or partial references. More is better than fewer.

Respond ONLY with valid JSON:
{
    "entities": ["entity1", "entity2"],
    "intent": "factual|relational|comparative|exploratory",
    "concepts": ["concept1", "concept2"]
}
"""

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=config.openai_api_key)
    return _client


@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=5))
def analyze_query(query: str) -> dict:
    """
    Analyze a user query to extract entities and determine retrieval strategy.
    
    Returns:
        {
            "entities": [...],
            "intent": "...",
            "concepts": [...],
        }
    """
    client = _get_client()

    response = client.chat.completions.create(
        model=config.llm.model,
        temperature=0.0,
        max_tokens=300,
        messages=[
            {"role": "system", "content": QUERY_ANALYSIS_PROMPT},
            {"role": "user", "content": query},
        ],
    )

    text = response.choices[0].message.content.strip()

    # Parse JSON (handle markdown code blocks)
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()

    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse query analysis: {text}")
        result = {
            "entities": [],
            "intent": "exploratory",
            "concepts": [query],
        }

    # Ensure all keys exist
    result.setdefault("entities", [])
    result.setdefault("intent", "exploratory")
    result.setdefault("concepts", [])

    # Always add the raw query terms as extra concepts (ensures graph Layer 2 coverage)
    query_words = [w.strip() for w in query.split() if len(w.strip()) > 2]
    existing_concepts = [c.lower() for c in result["concepts"]]
    for w in query_words:
        if w.lower() not in existing_concepts:
            result["concepts"].append(w)

    logger.info(f"Query analysis: entities={result['entities']}, concepts={result['concepts'][:5]}")
    return result
