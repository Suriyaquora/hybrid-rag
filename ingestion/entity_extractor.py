"""
Entity and Relation extraction using OpenAI function calling.
This is the critical NLP layer that powers the Knowledge Graph pipeline.

Strategy:
  1. Use GPT-4o-mini with structured output (function calling) to extract
     entities and relations from each chunk.
  2. Normalize entity names for deduplication.
  3. Return typed Entity and Relation objects.
"""

import json
import logging
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from core.config import config
from core.models import Entity, Relation, DocumentChunk

logger = logging.getLogger(__name__)

# ── Extraction prompt ─────────────────────────────────────────────────────────

EXTRACTION_SYSTEM_PROMPT = """You are an expert knowledge graph builder. Given a text chunk, extract:

1. **Entities**: Named things (people, organizations, technologies, concepts, locations, events, products, metrics).
2. **Relations**: How entities relate to each other.

Rules:
- Entity names should be canonical (e.g., "OpenAI" not "openai" or "Open AI").
- Relation types should be UPPERCASE_SNAKE_CASE (e.g., WORKS_FOR, DEVELOPED_BY, LOCATED_IN, PART_OF, RELATED_TO, USES, COMPETED_WITH, FOUNDED_BY, HAS_FEATURE).
- Only extract entities and relations that are explicitly stated or strongly implied.
- Each relation must connect exactly two extracted entities.
- Assign a confidence score (0.0-1.0) to each relation.
- Be thorough but precise — quality over quantity.
"""

EXTRACTION_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "extract_knowledge",
            "description": "Extract entities and relations from a text chunk for knowledge graph construction.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entities": {
                        "type": "array",
                        "description": "List of entities found in the text",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "Canonical name of the entity"
                                },
                                "label": {
                                    "type": "string",
                                    "enum": [
                                        "PERSON", "ORGANIZATION", "TECHNOLOGY",
                                        "CONCEPT", "LOCATION", "EVENT",
                                        "PRODUCT", "METRIC", "OTHER"
                                    ],
                                    "description": "Type/category of the entity"
                                },
                                "description": {
                                    "type": "string",
                                    "description": "Brief description of the entity in context"
                                }
                            },
                            "required": ["name", "label"]
                        }
                    },
                    "relations": {
                        "type": "array",
                        "description": "List of relations between entities",
                        "items": {
                            "type": "object",
                            "properties": {
                                "source": {
                                    "type": "string",
                                    "description": "Source entity name (must match an extracted entity)"
                                },
                                "target": {
                                    "type": "string",
                                    "description": "Target entity name (must match an extracted entity)"
                                },
                                "relation_type": {
                                    "type": "string",
                                    "description": "Type of relation in UPPERCASE_SNAKE_CASE"
                                },
                                "description": {
                                    "type": "string",
                                    "description": "Brief description of the relation"
                                },
                                "confidence": {
                                    "type": "number",
                                    "description": "Confidence score 0.0-1.0"
                                }
                            },
                            "required": ["source", "target", "relation_type", "confidence"]
                        }
                    }
                },
                "required": ["entities", "relations"]
            }
        }
    }
]


# ── Client ────────────────────────────────────────────────────────────────────
_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=config.openai_api_key)
    return _client


# ── Core extraction ───────────────────────────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def extract_from_text(text: str, chunk_id: str = "") -> tuple[list[Entity], list[Relation]]:
    """
    Extract entities and relations from a single text chunk
    using OpenAI function calling.
    """
    client = _get_client()

    response = client.chat.completions.create(
        model=config.llm.model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": f"Extract entities and relations from this text:\n\n{text}"},
        ],
        tools=EXTRACTION_TOOLS,
        tool_choice={"type": "function", "function": {"name": "extract_knowledge"}},
    )

    # Parse the function call result
    tool_call = response.choices[0].message.tool_calls[0]
    result = json.loads(tool_call.function.arguments)

    # Build Entity objects
    entities = []
    entity_names = set()
    for e in result.get("entities", []):
        entity = Entity(
            name=e["name"],
            label=e["label"],
            properties={"description": e.get("description", "")},
            source_chunks=[chunk_id] if chunk_id else [],
        )
        entities.append(entity)
        entity_names.add(e["name"])

    # Build Relation objects (only if both endpoints exist)
    relations = []
    for r in result.get("relations", []):
        if r["source"] in entity_names and r["target"] in entity_names:
            relation = Relation(
                source=r["source"],
                target=r["target"],
                relation_type=r["relation_type"],
                properties={"description": r.get("description", "")},
                source_chunks=[chunk_id] if chunk_id else [],
                confidence=r.get("confidence", 0.8),
            )
            relations.append(relation)

    logger.debug(f"Chunk {chunk_id}: {len(entities)} entities, {len(relations)} relations")
    return entities, relations


def extract_from_chunks(
    chunks: list[DocumentChunk],
    progress_callback=None,
) -> tuple[list[Entity], list[Relation]]:
    """
    Extract entities and relations from all chunks.
    Deduplicates entities by normalized name.
    """
    all_entities: dict[str, Entity] = {}  # normalized_name -> Entity
    all_relations: list[Relation] = []

    for i, chunk in enumerate(chunks):
        try:
            entities, relations = extract_from_text(chunk.text, chunk.chunk_id)

            # Deduplicate entities
            for entity in entities:
                key = entity.normalized_name
                if key in all_entities:
                    # Merge source chunks
                    all_entities[key].source_chunks.extend(entity.source_chunks)
                else:
                    all_entities[key] = entity

            all_relations.extend(relations)

            if progress_callback:
                progress_callback(i + 1, len(chunks))

        except Exception as e:
            logger.warning(f"Extraction failed for chunk {chunk.chunk_id}: {e}")
            continue

    entity_list = list(all_entities.values())
    logger.info(
        f"Extraction complete: {len(entity_list)} unique entities, "
        f"{len(all_relations)} relations from {len(chunks)} chunks"
    )
    return entity_list, all_relations
