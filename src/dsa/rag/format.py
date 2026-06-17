"""
Document formatting utilities for RAG output.

Formats retrieved documents into readable, citation-friendly text.
"""

from __future__ import annotations


def format_docs(docs: list) -> str:
    """Format retrieved documents into a numbered citation string.

    Parameters
    ----------
    docs : list
        LangChain Document objects from a retriever.

    Returns
    -------
    str
        Formatted string with source citations.
    """
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        parts.append(f"[{i}] (Source: {source}, p.{page})\n{doc.page_content}")
    return "\n\n".join(parts)