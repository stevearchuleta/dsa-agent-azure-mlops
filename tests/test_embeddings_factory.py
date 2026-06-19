"""Tests for Azure OpenAI embeddings factory."""

from __future__ import annotations

from dsa.config import AzureOpenAISettings


class FakeAzureOpenAIEmbeddings:
    """Small fake class that captures constructor keyword arguments."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_get_embeddings_builds_azure_openai_embeddings(monkeypatch):
    import dsa.embeddings as embeddings_module

    settings = AzureOpenAISettings(
        endpoint="https://example.openai.azure.com/",
        api_key="fake-key",
        api_version="2024-10-21",
        chat_deployment="chat-test",
        embedding_deployment="embed-test",
    )

    monkeypatch.setattr(
        embeddings_module,
        "get_azure_openai_config",
        lambda require_credentials=True: settings,
    )

    monkeypatch.setattr(
        embeddings_module,
        "AzureOpenAIEmbeddings",
        FakeAzureOpenAIEmbeddings,
    )

    embeddings = embeddings_module.get_embeddings()

    assert embeddings.kwargs["azure_endpoint"] == "https://example.openai.azure.com/"
    assert embeddings.kwargs["api_key"] == "fake-key"
    assert embeddings.kwargs["api_version"] == "2024-10-21"
    assert embeddings.kwargs["azure_deployment"] == "embed-test"


def test_get_embeddings_supports_explicit_deployment(monkeypatch):
    import dsa.embeddings as embeddings_module

    settings = AzureOpenAISettings(
        endpoint="https://example.openai.azure.com/",
        api_key="fake-key",
        api_version="2024-10-21",
        chat_deployment="chat-test",
        embedding_deployment="embed-default",
    )

    monkeypatch.setattr(
        embeddings_module,
        "get_azure_openai_config",
        lambda require_credentials=True: settings,
    )

    monkeypatch.setattr(
        embeddings_module,
        "AzureOpenAIEmbeddings",
        FakeAzureOpenAIEmbeddings,
    )

    embeddings = embeddings_module.get_embeddings(deployment="embed-explicit")

    assert embeddings.kwargs["azure_deployment"] == "embed-explicit"