# 🔬 Hybrid RAG — Vector Search + Knowledge Graph

A production-quality Retrieval-Augmented Generation system that **always** runs both **FAISS vector search** (semantic similarity) and **Neo4j Knowledge Graph** (structural/relational reasoning) on every query, fused via **Reciprocal Rank Fusion (RRF)**.

> **Core principle:** Every query hits both pipelines. No conditional routing. The graph pipeline uses a 3-layer fallback to guarantee results.

---

## Architecture

```
                    ┌─────────────────────────────────┐
                    │         User Query               │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │       Query Analyzer             │
                    │  (Entity + concept extraction)   │
                    └──────┬──────────────┬───────────┘
                           │              │
          ALWAYS RUNS      │              │      ALWAYS RUNS
              ┌────────────▼────┐  ┌──────▼─────────────────┐
              │  Vector Pipeline │  │  Graph Pipeline         │
              │                  │  │  (3-Layer Fallback)     │
              │ embed query      │  │                         │
              │ → FAISS search   │  │ L1: Entity subgraph     │
              │ → top-k chunks   │  │ L2: Fulltext keyword    │
              │                  │  │ L3: Relation overview   │
              └────────┬─────────┘  └──────┬─────────────────┘
                       │                   │
              ┌────────▼───────────────────▼────────┐
              │     Reciprocal Rank Fusion (RRF)     │
              │  Merges + re-ranks both result sets   │
              └──────────────────┬───────────────────┘
                                 │
              ┌──────────────────▼───────────────────┐
              │        LLM Generation (GPT-4o-mini)   │
              │  Grounded answer from fused context    │
              └──────────────────────────────────────┘
```

### Graph Pipeline — 3-Layer Fallback

| Layer | Method | Quality | When |
|-------|--------|---------|------|
| L1 | Direct entity match + subgraph traversal | Highest | Entities found in query |
| L2 | Fulltext keyword search across entities + chunks | Medium | Always runs |
| L3 | Broad relation overview from entire KG | Baseline | If L1+L2 < 3 results |

This guarantees the graph pipeline **always contributes context** as long as the KG has data.

---

## Ingestion Pipeline

```
Documents (PDF/TXT/MD/DOCX/CSV)
        │
        ▼
   Load & Parse (LangChain loaders)
        │
        ▼
   Chunk (RecursiveCharacterTextSplitter)
        │
        ├──► Embed (OpenAI text-embedding-3-small) ──► FAISS Index
        │
        └──► Entity Extraction (GPT-4o-mini + function calling)
                    │
                    ├──► Entities ──► Neo4j Nodes
                    └──► Relations ──► Neo4j Edges
                    └──► Chunks ──► Neo4j Chunk Nodes (linked to entities)
```

---

## Quick Start

### 1. Prerequisites

- **Python 3.11+**
- **uv** — install with one of:
  - **Windows (PowerShell):** `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
  - **Windows (pip):** `pip install uv`
  - **macOS/Linux:** `curl -LsSf https://astral.sh/uv/install.sh | sh`
  - After installing, **restart your terminal** so the PATH updates.
- **Neo4j** (local or [Aura Free](https://neo4j.com/cloud/aura-free/))
- **OpenAI API key**

### 2. Install with uv

```bash
cd hybrid_rag

# Create venv and install all dependencies in one shot
uv sync

# Download spaCy model
uv run python -m spacy download en_core_web_sm
```

That's it. `uv sync` reads `pyproject.toml`, creates the venv, resolves dependencies, and installs everything — typically 10-50x faster than pip.

### 3. Configure

```bash
cp .env.example .env
# Edit .env with your keys:
#   OPENAI_API_KEY=sk-...
#   NEO4J_URI=bolt://localhost:7687
#   NEO4J_USERNAME=neo4j
#   NEO4J_PASSWORD=your-password
```

### 4. Start Neo4j

**Option A: Docker (recommended)**
```bash
docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your-password \
  neo4j:5
```

**Option B: Neo4j Desktop** — Download from [neo4j.com/download](https://neo4j.com/download/)

### 5. Run

```bash
uv run streamlit run ui/app.py
```

Open [http://localhost:8501](http://localhost:8501)

### Common uv commands

```bash
uv sync                    # Install/update all dependencies
uv add <package>           # Add a new dependency
uv remove <package>        # Remove a dependency
uv run <command>           # Run a command in the project venv
uv run python script.py    # Run any Python script
uv lock                    # Regenerate the lockfile
```

---

## Project Structure

```
hybrid_rag/
├── pyproject.toml         # uv project config + dependencies
├── core/
│   ├── config.py          # Centralized config (pydantic + .env)
│   ├── models.py          # Data models (Entity, Relation, DocumentChunk, etc.)
│   └── pipeline.py        # Main orchestrator (ingest + query)
├── ingestion/
│   ├── loader.py          # Document loading + chunking
│   ├── embeddings.py      # OpenAI embedding generation
│   └── entity_extractor.py # GPT-4o-mini entity/relation extraction
├── retrieval/
│   ├── vector_store.py    # FAISS index wrapper
│   ├── query_analyzer.py  # Query → entities + concepts (no routing)
│   ├── hybrid_retriever.py # ALWAYS dual-pipeline + RRF fusion
│   └── generator.py       # LLM answer generation + streaming
├── graph/
│   └── kg_manager.py      # Neo4j KG: CRUD + entity search + fulltext + overview
├── ui/
│   └── app.py             # Streamlit interface
├── data/                  # FAISS index persistence
└── .env.example
```

---

## How It Works

### Why Both Pipelines on Every Query?

| Scenario | Vector Only | Graph Only | Both (This System) |
|----------|-----------|-----------|-------------------|
| "Explain transformers" | ✅ Finds relevant passages | ❌ May miss if no entity match | ✅ Best of both |
| "How does OpenAI relate to Microsoft?" | ⚠️ Finds passages mentioning both | ✅ Traverses direct relation | ✅ Relation + context |
| "Summarize the paper" | ✅ Semantic match | ⚠️ Keyword hits on chunks | ✅ Full coverage |
| "What technologies does Company X use?" | ⚠️ Scattered mentions | ✅ Entity → USES → edges | ✅ Structured + unstructured |

Running both eliminates blind spots. The RRF fusion naturally weights whichever pipeline was more useful for that specific query.

### Reciprocal Rank Fusion (RRF)

```
RRF_score(doc) = Σ (weight_i / (k + rank_i))
```

Items appearing in **both** result sets get a natural score boost, while items only in one pipeline still contribute.

---

## Configuration

All settings in `.env` or adjustable in the Streamlit sidebar:

| Setting | Default | Description |
|---------|---------|-------------|
| `CHUNK_SIZE` | 512 | Characters per chunk |
| `CHUNK_OVERLAP` | 64 | Overlap between chunks |
| `VECTOR_TOP_K` | 5 | Results from FAISS |
| `GRAPH_TOP_K` | 5 | Results from Neo4j |
| `HYBRID_WEIGHT_VECTOR` | 0.5 | RRF weight for vector |
| `HYBRID_WEIGHT_GRAPH` | 0.5 | RRF weight for graph |
| `LLM_MODEL` | gpt-4o-mini | Generation model |
| `EMBEDDING_MODEL` | text-embedding-3-small | Embedding model |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `uv: command not found` / `uv is not recognized` | **Windows:** `pip install uv` then restart terminal. **Mac/Linux:** `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Neo4j connection failed | Ensure Neo4j is running and credentials match `.env` |
| "No chunks" after upload | Check file format is supported (PDF/TXT/MD/DOCX/CSV) |
| Slow entity extraction | This is the bottleneck — each chunk needs an LLM call |
| High costs | Switch to `gpt-4o-mini` (default) and reduce chunk count |
| FAISS import error | `uv add faiss-cpu` (not `faiss-gpu` unless you have CUDA) |
| Windows path errors | Use forward slashes or raw strings in `.env` paths |

---

## License

MIT
