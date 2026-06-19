"""
Azure OpenAI LLM factory.

Builds AzureChatOpenAI instances from config/config.local.json or environment
variables. Application code uses Azure deployment names, not raw model names.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_openai import AzureChatOpenAI

from dsa.config import LLM_TEMPERATURE, get_azure_openai_config

logger = logging.getLogger(__name__)


def get_llm(
    deployment: str | None = None,
    temperature: float | None = None,
    *,
    fallback: bool = True,
    model: str | None = None,
) -> Any:
    """Return an AzureChatOpenAI instance.

    Parameters
    ----------
    deployment : str, optional
        Azure OpenAI deployment name. This is preferred.
    temperature : float, optional
        Sampling temperature.
    fallback : bool
        Attach a fallback deployment when one is configured.
    model : str, optional
        Backward-compatible alias for deployment.
    """
    settings = get_azure_openai_config(require_credentials=True)

    selected_deployment = deployment or model or settings.chat_deployment
    selected_temperature = temperature if temperature is not None else LLM_TEMPERATURE

    primary = AzureChatOpenAI(
        azure_endpoint=settings.endpoint,
        api_key=settings.api_key,
        api_version=settings.api_version,
        azure_deployment=selected_deployment,
        temperature=selected_temperature,
    )

    logger.info(
        "Primary Azure OpenAI chat deployment: %s (temperature=%.1f)",
        selected_deployment,
        selected_temperature,
    )

    if (
        fallback
        and deployment is None
        and model is None
        and settings.chat_fallback_deployment
        and settings.chat_fallback_deployment != selected_deployment
    ):
        backup = AzureChatOpenAI(
            azure_endpoint=settings.endpoint,
            api_key=settings.api_key,
            api_version=settings.api_version,
            azure_deployment=settings.chat_fallback_deployment,
            temperature=selected_temperature,
        )

        logger.info(
            "Fallback Azure OpenAI chat deployment: %s",
            settings.chat_fallback_deployment,
        )

        return primary.with_fallbacks([backup])

    return primary