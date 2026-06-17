"""
PDF ingestion and FAISS index management.

Loads PDFs from the papers directory, splits them into chunks,
embeds them, and persists a FAISS vector store to disk.
"""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from dsa.config import PAPERS_DIR, FAISS_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from dsa.embeddings import get_embeddings

logger = logging.getLogger(__name__)

INDEX_NAME = "dsa_index"


def load_pdfs(papers_dir: Path | None = None) -> list:
    """Load all PDFs from the papers directory.

    Returns a flat list of LangChain Document objects (one per page).
    """
    _dir = papers_dir or PAPERS_DIR
    pdf_files = sorted(_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDFs found in %s", _dir)
        return []

    docs = []
    for pdf in pdf_files:
        logger.info("Loading %s", pdf.name)
        loader = PyPDFLoader(str(pdf))
        docs.extend(loader.load())

    logger.info("Loaded %d pages from %d PDFs", len(docs), len(pdf_files))
    return docs


def split_documents(docs: list) -> list:
    """Split documents into chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    logger.info("Split %d pages into %d chunks", len(docs), len(chunks))
    return chunks


def build_index(chunks: list) -> FAISS:
    """Build a FAISS index from document chunks and persist to disk."""
    embeddings = get_embeddings()
    index = FAISS.from_documents(chunks, embeddings)
    index.save_local(str(FAISS_DIR), index_name=INDEX_NAME)
    logger.info("FAISS index saved to %s", FAISS_DIR)
    return index


def load_index() -> FAISS:
    """Load a previously saved FAISS index from disk."""
    embeddings = get_embeddings()
    index = FAISS.from_local(
        str(FAISS_DIR),
        embeddings,
        index_name=INDEX_NAME,
        allow_dangerous_deserialization=True,
    )
    logger.info("FAISS index loaded from %s", FAISS_DIR)
    return index


def get_or_build_index() -> FAISS:
    """Load existing FAISS index, or build from PDFs if none exists."""
    index_file = FAISS_DIR / f"{INDEX_NAME}.faiss"
    if index_file.exists():
        logger.info("Found existing index: %s", index_file)
        return load_index()

    logger.info("No index found; building from PDFs...")
    docs = load_pdfs()
    if not docs:
        raise FileNotFoundError(
            f"No PDFs in {PAPERS_DIR}. Add papers before building the index."
        )
    chunks = split_documents(docs)
    return build_index(chunks)