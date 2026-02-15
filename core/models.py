"""
Data models used across the Hybrid RAG system.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SourceType(str, Enum):
    VECTOR = "vector"
    GRAPH = "graph"
    HYBRID = "hybrid"


class DocType(str, Enum):
    PDF = "pdf"
    TXT = "txt"
    MD = "md"
    DOCX = "docx"
    CSV = "csv"


@dataclass
class DocumentChunk:
    """A chunk of text from a document with metadata."""
    text: str
    doc_id: str
    chunk_id: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None

    @property
    def source_file(self) -> str:
        return self.metadata.get("source", "unknown")

    @property
    def page_number(self) -> int | None:
        return self.metadata.get("page_number")


@dataclass
class Entity:
    """An entity extracted from text (person, org, concept, etc.)."""
    name: str
    label: str  # PERSON, ORG, CONCEPT, LOCATION, EVENT, TECHNOLOGY, etc.
    properties: dict[str, Any] = field(default_factory=dict)
    source_chunks: list[str] = field(default_factory=list)  # chunk_ids

    @property
    def normalized_name(self) -> str:
        return self.name.strip().lower()


@dataclass
class Relation:
    """A relation between two entities."""
    source: str  # entity name
    target: str  # entity name
    relation_type: str  # e.g., WORKS_FOR, LOCATED_IN, RELATED_TO
    properties: dict[str, Any] = field(default_factory=dict)
    source_chunks: list[str] = field(default_factory=list)
    confidence: float = 1.0


@dataclass
class KnowledgeTriple:
    """A (subject, predicate, object) triple for the KG."""
    subject: Entity
    predicate: str
    obj: Entity
    chunk_id: str
    confidence: float = 1.0


@dataclass
class RetrievedContext:
    """A piece of context retrieved from either vector or graph pipeline."""
    text: str
    source: SourceType
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    entities: list[str] = field(default_factory=list)
    relations: list[str] = field(default_factory=list)

    @property
    def source_file(self) -> str:
        return self.metadata.get("source", "unknown")


@dataclass
class HybridResult:
    """Final fused result from the hybrid retrieval pipeline."""
    contexts: list[RetrievedContext]
    vector_contexts: list[RetrievedContext]
    graph_contexts: list[RetrievedContext]
    query: str
    answer: str = ""
    total_tokens_used: int = 0
    cost_estimate: float = 0.0
