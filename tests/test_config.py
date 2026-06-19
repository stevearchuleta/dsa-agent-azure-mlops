"""Tests for DSA configuration loading."""

from __future__ import annotations

import json


AZURE_ENV_NAMES = (
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_API_VERSION",
    "AZURE_OPENAI_CHAT_DEPLOYMENT",
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
    "AZURE_OPENAI_CHAT_FALLBACK_DEPLOYMENT",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "RETRIEVAL_K",
    "RANDOM_SEED",
    "LLM_TEMPERATURE",
)


def clear_project_env(monkeypatch):
    """Remove project environment variables for deterministic tests."""
    for env_name in AZURE_ENV_NAMES:
        monkeypatch.delenv(env_name, raising=False)


def test_load_settings_reads_json_with_bom(tmp_path, monkeypatch):
    from dsa.config import load_settings

    clear_project_env(monkeypatch)

    config_path = tmp_path / "config.local.json"

    config_data = {
        "azure_openai": {
            "endpoint": "https://example.openai.azure.com/",
            "api_key": "fake-key",
            "api_version": "2024-10-21",
            "chat_deployment": "chat-local",
            "embedding_deployment": "embed-local",
        },
        "rag": {
            "chunk_size": 111,
            "chunk_overlap": 22,
            "retrieval_k": 3,
        },
        "ml": {
            "random_seed": 692,
        },
    }

    config_text = "\ufeff" + json.dumps(config_data)
    config_path.write_text(config_text, encoding="utf-8")

    settings = load_settings(config_path, require_credentials=True)

    assert settings.azure_openai.endpoint == "https://example.openai.azure.com/"
    assert settings.azure_openai.api_key == "fake-key"
    assert settings.azure_openai.chat_deployment == "chat-local"
    assert settings.azure_openai.embedding_deployment == "embed-local"
    assert settings.rag.chunk_size == 111
    assert settings.rag.chunk_overlap == 22
    assert settings.rag.retrieval_k == 3
    assert settings.ml.random_seed == 692


def test_load_settings_uses_environment_when_json_missing(tmp_path, monkeypatch):
    from dsa.config import load_settings

    clear_project_env(monkeypatch)

    missing_config_path = tmp_path / "missing.json"

    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://env.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "fake-key")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
    monkeypatch.setenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "chat-env")
    monkeypatch.setenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "embed-env")
    monkeypatch.setenv("CHUNK_SIZE", "222")
    monkeypatch.setenv("CHUNK_OVERLAP", "33")
    monkeypatch.setenv("RETRIEVAL_K", "4")
    monkeypatch.setenv("RANDOM_SEED", "42")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.25")

    settings = load_settings(missing_config_path, require_credentials=True)

    assert settings.azure_openai.endpoint == "https://env.openai.azure.com/"
    assert settings.azure_openai.api_key == "fake-key"
    assert settings.azure_openai.chat_deployment == "chat-env"
    assert settings.azure_openai.embedding_deployment == "embed-env"
    assert settings.rag.chunk_size == 222
    assert settings.rag.chunk_overlap == 33
    assert settings.rag.retrieval_k == 4
    assert settings.ml.random_seed == 42
    assert settings.llm_temperature == 0.25


def test_load_settings_raises_when_credentials_required_and_missing(tmp_path, monkeypatch):
    import pytest

    from dsa.config import load_settings

    clear_project_env(monkeypatch)

    missing_config_path = tmp_path / "missing.json"

    with pytest.raises(RuntimeError, match="Missing Azure OpenAI settings"):
        load_settings(missing_config_path, require_credentials=True)