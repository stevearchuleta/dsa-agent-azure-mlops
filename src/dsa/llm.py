"""
LLM builder with automatic fallback.

Tries the primary model first; falls back to a cheaper model
if the primary is unavailable or raises an error.
"""

from __future__ import annotations

import logging

from langchain_openai import ChatOpenAI

from dsa.config import LLM_MODEL, LLM_FALLBACK, LLM_TEMPERATURE, ensure_api_key

logger = logging.getLogger(__name__)


def get_llm(
    model: str | None = None,
    temperature: float | None = None,
    *,
    fallback: bool = True,
) -> ChatOpenAI:
    """Return a ChatOpenAI instance, optionally with a fallback model.

    Parameters
    ----------
    model : str, optional
        Override the primary model name (default: config.LLM_MODEL).
    temperature : float, optional
        Override the temperature (default: config.LLM_TEMPERATURE).
    fallback : bool
        If True, attach a fallback model via .with_fallbacks().

    Returns
    -------
    ChatOpenAI
        Ready-to-use LLM (with or without fallback).
    """
    ensure_api_key()

    _model = model or LLM_MODEL
    _temp = temperature if temperature is not None else LLM_TEMPERATURE

    primary = ChatOpenAI(model=_model, temperature=_temp)
    logger.info("Primary LLM: %s (temperature=%.1f)", _model, _temp)

    if fallback and _model != LLM_FALLBACK:
        backup = ChatOpenAI(model=LLM_FALLBACK, temperature=_temp)
        logger.info("Fallback LLM: %s", LLM_FALLBACK)
        return primary.with_fallbacks([backup])

    return primary