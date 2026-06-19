"""
Azure OpenAI embeddings factory.

Builds AzureOpenAIEmbeddings instances for FAISS vector stores and document
retrieval. Application code uses Azure deployment names, not raw model names.
"""

from __future__ import annotations

import logging

from langchain_openai import AzureOpenAIEmbeddings

from dsa.config import get_azure_openai_config

logger = logging.getLogger(__name__)


def get_embeddings(
    deployment: str | None = None,
    *,
    model: str | None = None,
) -> AzureOpenAIEmbeddings:
    """Return an AzureOpenAIEmbeddings instance.

    Parameters
    ----------
    deployment : str, optional
        Azure OpenAI embedding deployment name. This is preferred.
    model : str, optional
        Backward-compatible alias for deployment.
    """
    settings = get_azure_openai_config(require_credentials=True)
    selected_deployment = deployment or model or settings.embedding_deployment

    logger.info("Azure OpenAI embedding deployment: %s", selected_deployment)

    return AzureOpenAIEmbeddings(
        azure_endpoint=settings.endpoint,
        api_key=settings.api_key,
        api_version=settings.api_version,
        azure_deployment=selected_deployment,
    )