"""
Streamlit UI for the Hybrid RAG System.
Features:
  - Document upload and ingestion (PDF, TXT, MD, DOCX, CSV)
  - Chat interface with streaming responses
  - Knowledge Graph statistics and entity browser
  - Retrieval diagnostics (vector vs graph vs hybrid breakdown)
  - Cost tracking
"""

import os
import sys
import json
import tempfile
import logging
import streamlit as st
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import config
from core.pipeline import HybridRAGPipeline
from core.models import SourceType

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hybrid RAG — Vector + Knowledge Graph",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=DM+Sans:wght@400;500;700&display=swap');
    
    .stApp {
        font-family: 'DM Sans', sans-serif;
    }
    
    .main-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2rem;
        font-weight: 600;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        color: #6b7280;
        font-size: 0.95rem;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
    }
    
    .metric-card h4 {
        margin: 0;
        font-size: 0.8rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .metric-card .value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.5rem;
        font-weight: 600;
        color: #1e293b;
    }
    
    .source-tag {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .source-vector { background: #dbeafe; color: #1d4ed8; }
    .source-graph { background: #dcfce7; color: #15803d; }
    .source-hybrid { background: #fef3c7; color: #b45309; }
    
    .context-box {
        background: #fafafa;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        font-size: 0.88rem;
        line-height: 1.6;
    }
    
    .chat-msg-user {
        background: #eff6ff;
        border-radius: 12px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
    }
    
    .chat-msg-assistant {
        background: #f8fafc;
        border-left: 3px solid #667eea;
        border-radius: 0 12px 12px 0;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
    }
    
    div[data-testid="stExpander"] {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────

def init_session_state():
    defaults = {
        "pipeline": None,
        "initialized": False,
        "chat_history": [],
        "total_cost": 0.0,
        "total_tokens": 0,
        "ingestion_stats": None,
        "uploaded_files_count": 0,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session_state()


# ── Pipeline management ───────────────────────────────────────────────────────

def get_pipeline() -> HybridRAGPipeline:
    if st.session_state.pipeline is None:
        st.session_state.pipeline = HybridRAGPipeline()
    return st.session_state.pipeline


def initialize_pipeline():
    pipeline = get_pipeline()
    status = pipeline.initialize()
    st.session_state.initialized = True
    return status


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown('<div class="main-title">🔬 Hybrid RAG</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="subtitle">Vector Search + Knowledge Graph</div>',
            unsafe_allow_html=True,
        )

        # ── System Status ──
        st.markdown("### ⚙️ System Status")
        if not st.session_state.initialized:
            if st.button("🚀 Initialize System", use_container_width=True, type="primary"):
                with st.spinner("Connecting to services..."):
                    status = initialize_pipeline()
                
                if status["vector_store"]:
                    st.success("✅ Vector Store loaded")
                else:
                    st.info("📦 Vector Store empty (upload docs to populate)")

                if status["knowledge_graph"]:
                    st.success("✅ Neo4j connected")
                else:
                    st.warning("⚠️ Neo4j not available — vector-only mode")
        else:
            st.success("System initialized")
            pipeline = get_pipeline()
            stats = pipeline.get_stats()

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Vectors", stats.get("vector_store_size", 0))
            with col2:
                kg_entities = stats.get("kg_entity_count", stats.get("entities_in_memory", 0))
                st.metric("KG Entities", kg_entities)

        st.divider()

        # ── Document Upload ──
        st.markdown("### 📄 Upload Documents")
        uploaded_files = st.file_uploader(
            "Drag & drop files",
            type=["pdf", "txt", "md", "docx", "csv"],
            accept_multiple_files=True,
            help="Supported: PDF, TXT, Markdown, DOCX, CSV",
        )

        if uploaded_files and st.button(
            f"🔄 Ingest {len(uploaded_files)} file(s)",
            use_container_width=True,
            type="primary",
        ):
            if not st.session_state.initialized:
                initialize_pipeline()

            pipeline = get_pipeline()

            # Save uploaded files to temp dir
            with tempfile.TemporaryDirectory() as tmpdir:
                for uf in uploaded_files:
                    file_path = os.path.join(tmpdir, uf.name)
                    with open(file_path, "wb") as f:
                        f.write(uf.getbuffer())

                # Progress tracking
                progress_bar = st.progress(0, text="Starting ingestion...")
                status_text = st.empty()

                steps_order = [
                    "Loading documents",
                    "Generating embeddings",
                    "Building vector index",
                    "Extracting entities",
                    "Building knowledge graph",
                ]
                step_weights = {
                    "Loading documents": 0.1,
                    "Generating embeddings": 0.3,
                    "Building vector index": 0.05,
                    "Extracting entities": 0.45,
                    "Building knowledge graph": 0.1,
                }

                completed_steps = set()

                def progress_cb(step, current, total):
                    if step not in step_weights:
                        return
                    # Calculate overall progress
                    base = sum(
                        step_weights.get(s, 0)
                        for s in steps_order
                        if s in completed_steps
                    )
                    step_progress = (current / max(total, 1)) * step_weights.get(step, 0)
                    overall = min(base + step_progress, 1.0)
                    progress_bar.progress(overall, text=f"{step}...")
                    status_text.text(f"{step}: {current}/{total}")
                    if current >= total:
                        completed_steps.add(step)

                stats = pipeline.ingest_documents(tmpdir, progress_callback=progress_cb)
                progress_bar.progress(1.0, text="Complete!")
                st.session_state.ingestion_stats = stats
                st.session_state.uploaded_files_count += len(uploaded_files)

            st.success(
                f"✅ Ingested {len(uploaded_files)} files → "
                f"{stats['chunks_created']} chunks, "
                f"{stats['entities_extracted']} entities, "
                f"{stats['relations_extracted']} relations"
            )

        st.divider()

        # ── Config ──
        st.markdown("### 🎛️ Configuration")
        with st.expander("Retrieval Weights"):
            v_weight = st.slider(
                "Vector weight",
                0.0, 1.0,
                config.retrieval.hybrid_weight_vector,
                0.05,
                key="v_weight",
            )
            g_weight = st.slider(
                "Graph weight",
                0.0, 1.0,
                config.retrieval.hybrid_weight_graph,
                0.05,
                key="g_weight",
            )
            config.retrieval.hybrid_weight_vector = v_weight
            config.retrieval.hybrid_weight_graph = g_weight

        with st.expander("Model Settings"):
            st.text(f"LLM: {config.llm.model}")
            st.text(f"Embeddings: {config.embedding.model}")
            st.text(f"Chunk size: {config.chunking.chunk_size}")

        st.divider()

        # ── Cost Tracker ──
        st.markdown("### 💰 Session Cost")
        st.markdown(
            f'<div class="metric-card">'
            f'<h4>Total Tokens</h4>'
            f'<div class="value">{st.session_state.total_tokens:,}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="metric-card">'
            f'<h4>Estimated Cost</h4>'
            f'<div class="value">${st.session_state.total_cost:.4f}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()


# ── Main Chat Interface ───────────────────────────────────────────────────────

def render_chat():
    st.markdown("## 💬 Chat")

    # Display chat history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(msg["content"])

                # Show retrieval diagnostics if available
                if "diagnostics" in msg:
                    diag = msg["diagnostics"]
                    with st.expander("🔍 Retrieval Diagnostics", expanded=False):
                        render_diagnostics(diag)

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        if not st.session_state.initialized:
            st.warning("Please initialize the system first (sidebar)")
            return

        pipeline = get_pipeline()
        if pipeline.vector_store.size == 0:
            st.warning("No documents ingested yet. Upload documents in the sidebar.")
            return

        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = pipeline.query(prompt)

            st.markdown(result.answer)

            # Update cost tracking
            st.session_state.total_tokens += result.total_tokens_used
            st.session_state.total_cost += result.cost_estimate

            # Diagnostics
            diagnostics = {
                "vector_count": len(result.vector_contexts),
                "graph_count": len(result.graph_contexts),
                "fused_count": len(result.contexts),
                "tokens": result.total_tokens_used,
                "cost": result.cost_estimate,
                "contexts": [
                    {
                        "text": c.text[:300],
                        "source": c.source.value,
                        "score": c.score,
                        "metadata": {k: str(v) for k, v in c.metadata.items()},
                        "entities": c.entities[:5],
                    }
                    for c in result.contexts[:8]
                ],
            }

            with st.expander("🔍 Retrieval Diagnostics", expanded=False):
                render_diagnostics(diagnostics)

            # Save to history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": result.answer,
                "diagnostics": diagnostics,
            })


def render_diagnostics(diag: dict):
    """Render retrieval diagnostic information."""
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Vector Results", diag.get("vector_count", 0))
    with col2:
        st.metric("Graph Results", diag.get("graph_count", 0))
    with col3:
        st.metric("Fused Results", diag.get("fused_count", 0))
    with col4:
        st.metric("Tokens Used", diag.get("tokens", 0))

    contexts = diag.get("contexts", [])
    if contexts:
        st.markdown("**Retrieved Contexts (ranked by RRF):**")
        for i, ctx in enumerate(contexts):
            source = ctx.get("source", "unknown")
            source_class = f"source-{source}"
            score = ctx.get("score", 0)

            st.markdown(
                f'<div class="context-box">'
                f'<span class="source-tag {source_class}">{source.upper()}</span> '
                f'Score: {score:.4f}<br/>'
                f'{ctx.get("text", "")[:300]}...'
                f'</div>',
                unsafe_allow_html=True,
            )

            if ctx.get("entities"):
                st.caption(f"Entities: {', '.join(ctx['entities'])}")


# ── Knowledge Graph Explorer Tab ─────────────────────────────────────────────

def render_kg_explorer():
    st.markdown("## 🕸️ Knowledge Graph Explorer")

    if not st.session_state.initialized:
        st.info("Initialize the system to explore the knowledge graph.")
        return

    pipeline = get_pipeline()

    try:
        if not pipeline.kg.verify_connection():
            st.warning("Neo4j is not connected. Start Neo4j to use KG features.")
            return

        stats = pipeline.kg.get_graph_stats()

        # Stats overview
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                f'<div class="metric-card">'
                f'<h4>Entities</h4>'
                f'<div class="value">{stats["entity_count"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f'<div class="metric-card">'
                f'<h4>Relations</h4>'
                f'<div class="value">{stats["relation_count"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                f'<div class="metric-card">'
                f'<h4>Chunks Linked</h4>'
                f'<div class="value">{stats["chunk_count"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Entity label distribution
        if stats.get("label_distribution"):
            st.markdown("### Entity Types")
            for item in stats["label_distribution"]:
                label = item.get("label", "UNKNOWN")
                count = item.get("count", 0)
                st.progress(
                    min(count / max(stats["entity_count"], 1), 1.0),
                    text=f"{label}: {count}",
                )

        # Relation types
        if stats.get("relation_types"):
            st.markdown("### Relation Types")
            for item in stats["relation_types"]:
                rtype = item.get("type", "UNKNOWN")
                count = item.get("count", 0)
                st.text(f"  {rtype}: {count}")

        # Entity browser
        st.markdown("### 🔎 Browse Entities")
        entities = pipeline.kg.get_all_entities(limit=200)
        if entities:
            entity_names = [e["name"] for e in entities]
            selected = st.selectbox("Select an entity", entity_names)

            if selected:
                contexts = pipeline.kg.get_entity_context(selected, depth=2)
                for ctx in contexts:
                    st.markdown(
                        f'<div class="context-box">{ctx.text}</div>',
                        unsafe_allow_html=True,
                    )
                    if ctx.entities:
                        st.caption(f"Connected entities: {', '.join(ctx.entities[:10])}")

    except Exception as e:
        st.error(f"KG Explorer error: {e}")


# ── Ingestion Stats Tab ──────────────────────────────────────────────────────

def render_ingestion_stats():
    st.markdown("## 📊 Ingestion Statistics")

    stats = st.session_state.ingestion_stats
    if stats is None:
        st.info("No documents ingested yet. Upload files in the sidebar.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Chunks Created", stats.get("chunks_created", 0))
        st.metric("Embeddings", stats.get("embeddings_generated", 0))
    with col2:
        st.metric("Entities Extracted", stats.get("entities_extracted", 0))
        st.metric("Relations Extracted", stats.get("relations_extracted", 0))
    with col3:
        st.metric("FAISS Index Size", stats.get("vector_store_size", 0))
        st.metric("KG Entities", stats.get("kg_entities", 0))

    # Show entities in memory
    pipeline = get_pipeline()
    if pipeline.entities:
        st.markdown("### Extracted Entities")
        entity_data = [
            {
                "Name": e.name,
                "Type": e.label,
                "Description": e.properties.get("description", "")[:100],
                "Chunks": len(e.source_chunks),
            }
            for e in pipeline.entities
        ]
        st.dataframe(entity_data, use_container_width=True)

    if pipeline.relations:
        st.markdown("### Extracted Relations")
        rel_data = [
            {
                "Source": r.source,
                "Relation": r.relation_type,
                "Target": r.target,
                "Confidence": f"{r.confidence:.2f}",
            }
            for r in pipeline.relations
        ]
        st.dataframe(rel_data, use_container_width=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    render_sidebar()

    # Tabs
    tab_chat, tab_kg, tab_stats = st.tabs([
        "💬 Chat",
        "🕸️ Knowledge Graph",
        "📊 Statistics",
    ])

    with tab_chat:
        render_chat()

    with tab_kg:
        render_kg_explorer()

    with tab_stats:
        render_ingestion_stats()


if __name__ == "__main__":
    main()
