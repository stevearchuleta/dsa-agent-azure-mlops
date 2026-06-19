"""Tests for Azure OpenAI LLM factory."""

from __future__ import annotations

from dsa.config import AzureOpenAISettings


class FakeAzureChatOpenAI:
    """Small fake class that captures constructor keyword arguments."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.fallbacks = []

    def with_fallbacks(self, fallbacks):
        self.fallbacks = fallbacks
        return self


def test_get_llm_builds_azure_chat_openai(monkeypatch):
    import dsa.llm as llm_module

    settings = AzureOpenAISettings(
        endpoint="https://example.openai.azure.com/",
        api_key="fake-key",
        api_version="2024-10-21",
        chat_deployment="chat-test",
        embedding_deployment="embed-test",
    )

    monkeypatch.setattr(
        llm_module,
        "get_azure_openai_config",
        lambda require_credentials=True: settings,
    )

    monkeypatch.setattr(llm_module, "AzureChatOpenAI", FakeAzureChatOpenAI)

    llm = llm_module.get_llm(fallback=False)

    assert llm.kwargs["azure_endpoint"] == "https://example.openai.azure.com/"
    assert llm.kwargs["api_key"] == "fake-key"
    assert llm.kwargs["api_version"] == "2024-10-21"
    assert llm.kwargs["azure_deployment"] == "chat-test"
    assert llm.kwargs["temperature"] == llm_module.LLM_TEMPERATURE


def test_get_llm_supports_explicit_deployment(monkeypatch):
    import dsa.llm as llm_module

    settings = AzureOpenAISettings(
        endpoint="https://example.openai.azure.com/",
        api_key="fake-key",
        api_version="2024-10-21",
        chat_deployment="chat-default",
        embedding_deployment="embed-test",
    )

    monkeypatch.setattr(
        llm_module,
        "get_azure_openai_config",
        lambda require_credentials=True: settings,
    )

    monkeypatch.setattr(llm_module, "AzureChatOpenAI", FakeAzureChatOpenAI)

    llm = llm_module.get_llm(deployment="chat-explicit", temperature=0.1)

    assert llm.kwargs["azure_deployment"] == "chat-explicit"
    assert llm.kwargs["temperature"] == 0.1