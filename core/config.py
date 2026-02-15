"""
Centralized configuration for the Hybrid RAG system.
Uses pydantic for validation and .env for secrets.
"""

import os
from pathlib import Path
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


class EmbeddingConfig(BaseModel):
    model: str = Field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))
    dimensions: int = 1536


class LLMConfig(BaseModel):
    model: str = Field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4o-mini"))
    temperature: float = Field(default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.1")))
    max_tokens: int = 2048


class ChunkingConfig(BaseModel):
    chunk_size: int = Field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "512")))
    chunk_overlap: int = Field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "64")))
    separators: list[str] = ["\n\n", "\n", ". ", " ", ""]


class Neo4jConfig(BaseModel):
    uri: str = Field(default_factory=lambda: os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    username: str = Field(default_factory=lambda: os.getenv("NEO4J_USERNAME", "neo4j"))
    password: str = Field(default_factory=lambda: os.getenv("NEO4J_PASSWORD", "password"))


class RetrievalConfig(BaseModel):
    vector_top_k: int = Field(default_factory=lambda: int(os.getenv("VECTOR_TOP_K", "5")))
    graph_top_k: int = Field(default_factory=lambda: int(os.getenv("GRAPH_TOP_K", "5")))
    hybrid_weight_vector: float = Field(
        default_factory=lambda: float(os.getenv("HYBRID_WEIGHT_VECTOR", "0.5"))
    )
    hybrid_weight_graph: float = Field(
        default_factory=lambda: float(os.getenv("HYBRID_WEIGHT_GRAPH", "0.5"))
    )
    rerank: bool = True


class AppConfig(BaseModel):
    embedding: EmbeddingConfig = EmbeddingConfig()
    llm: LLMConfig = LLMConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    neo4j: Neo4jConfig = Neo4jConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    faiss_index_path: str = "data/faiss_index"
    data_dir: str = "data"

    @property
    def openai_api_key(self) -> str:
        key = os.getenv("OPENAI_API_KEY", "")
        if not key:
            raise ValueError("OPENAI_API_KEY not set in environment")
        return key


# Singleton
config = AppConfig()
