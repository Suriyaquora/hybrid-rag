"""
Document loading and chunking pipeline.
Supports: PDF, TXT, Markdown, DOCX, CSV.
Uses semantic-aware chunking with RecursiveCharacterTextSplitter.
"""

import os
import hashlib
import logging
from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    Docx2txtLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from core.config import config
from core.models import DocumentChunk, DocType

logger = logging.getLogger(__name__)

# ── Loader registry ──────────────────────────────────────────────────────────
LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".md": TextLoader,
    ".docx": Docx2txtLoader,
    ".csv": CSVLoader,
}


def detect_doc_type(file_path: str) -> DocType | None:
    ext = Path(file_path).suffix.lower()
    mapping = {
        ".pdf": DocType.PDF,
        ".txt": DocType.TXT,
        ".md": DocType.MD,
        ".docx": DocType.DOCX,
        ".csv": DocType.CSV,
    }
    return mapping.get(ext)


def generate_doc_id(file_path: str) -> str:
    """Deterministic document ID from file path + content hash."""
    with open(file_path, "rb") as f:
        content_hash = hashlib.md5(f.read()).hexdigest()[:12]
    filename = Path(file_path).stem
    return f"{filename}_{content_hash}"


def load_single_document(file_path: str) -> list[Document]:
    """Load a single document using the appropriate loader."""
    ext = Path(file_path).suffix.lower()
    loader_cls = LOADER_MAP.get(ext)
    if loader_cls is None:
        raise ValueError(f"Unsupported file type: {ext}. Supported: {list(LOADER_MAP.keys())}")

    logger.info(f"Loading {file_path} with {loader_cls.__name__}")
    loader = loader_cls(file_path)
    docs = loader.load()

    # Enrich metadata
    for i, doc in enumerate(docs):
        doc.metadata["source"] = str(file_path)
        doc.metadata["file_name"] = Path(file_path).name
        doc.metadata["doc_type"] = ext.lstrip(".")
        if "page" in doc.metadata:
            doc.metadata["page_number"] = doc.metadata["page"]

    return docs


def load_documents_from_directory(directory: str) -> list[Document]:
    """Load all supported documents from a directory."""
    all_docs = []
    supported_exts = set(LOADER_MAP.keys())

    for root, _, files in os.walk(directory):
        for file in sorted(files):
            ext = Path(file).suffix.lower()
            if ext in supported_exts:
                file_path = os.path.join(root, file)
                try:
                    docs = load_single_document(file_path)
                    all_docs.extend(docs)
                    logger.info(f"  ✓ Loaded {file} → {len(docs)} pages/sections")
                except Exception as e:
                    logger.warning(f"  ✗ Failed to load {file}: {e}")

    logger.info(f"Total documents loaded: {len(all_docs)}")
    return all_docs


def chunk_documents(
    documents: list[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[DocumentChunk]:
    """
    Split documents into chunks using RecursiveCharacterTextSplitter.
    Returns DocumentChunk objects with proper IDs and metadata.
    """
    _chunk_size = chunk_size or config.chunking.chunk_size
    _chunk_overlap = chunk_overlap or config.chunking.chunk_overlap

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=_chunk_size,
        chunk_overlap=_chunk_overlap,
        separators=config.chunking.separators,
        length_function=len,
        is_separator_regex=False,
    )

    lc_chunks = splitter.split_documents(documents)

    chunks: list[DocumentChunk] = []
    for i, lc_chunk in enumerate(lc_chunks):
        source = lc_chunk.metadata.get("source", "unknown")
        doc_id = hashlib.md5(source.encode()).hexdigest()[:12]
        chunk_id = f"{doc_id}_chunk_{i:04d}"

        chunk = DocumentChunk(
            text=lc_chunk.page_content,
            doc_id=doc_id,
            chunk_id=chunk_id,
            metadata={
                **lc_chunk.metadata,
                "chunk_index": i,
                "char_count": len(lc_chunk.page_content),
            },
        )
        chunks.append(chunk)

    logger.info(f"Chunked {len(documents)} documents → {len(chunks)} chunks "
                f"(size={_chunk_size}, overlap={_chunk_overlap})")
    return chunks


def load_and_chunk(
    source: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[DocumentChunk]:
    """
    End-to-end: load from file or directory, then chunk.
    """
    path = Path(source)
    if path.is_file():
        docs = load_single_document(str(path))
    elif path.is_dir():
        docs = load_documents_from_directory(str(path))
    else:
        raise FileNotFoundError(f"Source not found: {source}")

    return chunk_documents(docs, chunk_size, chunk_overlap)
