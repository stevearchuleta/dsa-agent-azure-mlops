"""
Embedding factory.

Returns a configured OpenAI embeddings instance for use
with FAISS vector stores and document retrieval.
"""

from __future__ import annotations

import logging

from langchain_openai import OpenAIEmbeddings

from dsa.config import EMBEDDING_MODEL, ensure_api_key

logger = logging.getLogger(__name__)


def get_embeddings(model: str | None = None) -> OpenAIEmbeddings:
    """Return an OpenAIEmbeddings instance.

    Parameters
    ----------
    model : str, optional
        Override the embedding model name (default: config.EMBEDDING_MODEL).

    Returns
    -------
    OpenAIEmbeddings
        Ready-to-use embedding model.
    """
    ensure_api_key()

    _model = model or EMBEDDING_MODEL
    logger.info("Embedding model: %s", _model)
    return OpenAIEmbeddings(model=_model)