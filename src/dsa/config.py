"""
DSA Agent configuration.

Loads Azure OpenAI settings from config/config.local.json and environment
variables. The JSON reader uses utf-8-sig so Windows PowerShell UTF-8 BOM
files and standard UTF-8 files both parse correctly.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
CONFIG_LOCAL_PATH = CONFIG_DIR / "config.local.json"
DATA_DIR = PROJECT_ROOT / "data"
PAPERS_DIR = DATA_DIR / "papers"
FAISS_DIR = DATA_DIR / "faiss_index"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


# ---------------------------------------------------------------------------
# Directory creation
# ---------------------------------------------------------------------------
for _directory in (PAPERS_DIR, FAISS_DIR, ARTIFACTS_DIR):
    _directory.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Environment loading
# ---------------------------------------------------------------------------
load_dotenv(PROJECT_ROOT / ".env", override=False)


@dataclass(frozen=True)
class AzureOpenAISettings:
    """Azure OpenAI settings used by chat and embedding factories."""

    endpoint: str
    api_key: str
    api_version: str
    chat_deployment: str
    embedding_deployment: str
    chat_fallback_deployment: str | None = None


@dataclass(frozen=True)
class RagSettings:
    """RAG chunking and retrieval settings."""

    chunk_size: int
    chunk_overlap: int
    retrieval_k: int


@dataclass(frozen=True)
class MlSettings:
    """Deterministic ML settings."""

    random_seed: int


@dataclass(frozen=True)
class DSASettings:
    """Full project settings object."""

    azure_openai: AzureOpenAISettings
    rag: RagSettings
    ml: MlSettings
    llm_temperature: float


def _read_json_config(path: Path) -> dict[str, Any]:
    """Read a JSON config file with BOM-tolerant UTF-8 handling."""
    if not path.exists():
        return {}

    text = path.read_text(encoding="utf-8-sig")
    return json.loads(text)


def _section(config: dict[str, Any], section_name: str) -> dict[str, Any]:
    """Return a named JSON section as a dictionary."""
    value = config.get(section_name, {})
    if isinstance(value, dict):
        return value
    raise TypeError(f"Config section must be an object: {section_name}")


def _get_value(
    section: dict[str, Any],
    key: str,
    env_name: str,
    default: Any = None,
) -> Any:
    """Return an environment value, JSON section value, or default."""
    env_value = os.getenv(env_name)
    if env_value not in (None, ""):
        return env_value

    section_value = section.get(key)
    if section_value not in (None, ""):
        return section_value

    return default


def _get_optional_string(
    section: dict[str, Any],
    key: str,
    env_name: str,
) -> str | None:
    """Return an optional string value from environment or JSON."""
    value = _get_value(section, key, env_name, None)
    if value in (None, ""):
        return None
    return str(value)


def _get_int(
    section: dict[str, Any],
    key: str,
    env_name: str,
    default: int,
) -> int:
    """Return an integer value from environment, JSON, or default."""
    return int(_get_value(section, key, env_name, default))


def _get_float(
    section: dict[str, Any],
    key: str,
    env_name: str,
    default: float,
) -> float:
    """Return a float value from environment, JSON, or default."""
    return float(_get_value(section, key, env_name, default))


def load_settings(
    local_config_path: Path = CONFIG_LOCAL_PATH,
    *,
    require_credentials: bool = False,
) -> DSASettings:
    """Load settings from config.local.json and environment variables.

    Environment variables intentionally override JSON values. This lets
    GitHub Actions, Azure ML jobs, and local shells inject settings without
    editing config files.
    """
    config = _read_json_config(local_config_path)

    azure_section = _section(config, "azure_openai")
    rag_section = _section(config, "rag")
    ml_section = _section(config, "ml")

    azure_openai = AzureOpenAISettings(
        endpoint=str(
            _get_value(
                azure_section,
                "endpoint",
                "AZURE_OPENAI_ENDPOINT",
                "",
            )
        ),
        api_key=str(
            _get_value(
                azure_section,
                "api_key",
                "AZURE_OPENAI_API_KEY",
                "",
            )
        ),
        api_version=str(
            _get_value(
                azure_section,
                "api_version",
                "AZURE_OPENAI_API_VERSION",
                "2024-10-21",
            )
        ),
        chat_deployment=str(
            _get_value(
                azure_section,
                "chat_deployment",
                "AZURE_OPENAI_CHAT_DEPLOYMENT",
                "gpt-4o-mini-dsa",
            )
        ),
        embedding_deployment=str(
            _get_value(
                azure_section,
                "embedding_deployment",
                "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
                "text-embedding-3-small-dsa",
            )
        ),
        chat_fallback_deployment=_get_optional_string(
            azure_section,
            "chat_fallback_deployment",
            "AZURE_OPENAI_CHAT_FALLBACK_DEPLOYMENT",
        ),
    )

    if require_credentials:
        missing = []
        if not azure_openai.endpoint:
            missing.append("AZURE_OPENAI_ENDPOINT")
        if not azure_openai.api_key:
            missing.append("AZURE_OPENAI_API_KEY")
        if not azure_openai.chat_deployment:
            missing.append("AZURE_OPENAI_CHAT_DEPLOYMENT")
        if not azure_openai.embedding_deployment:
            missing.append("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

        if missing:
            missing_text = ", ".join(missing)
            raise RuntimeError(f"Missing Azure OpenAI settings: {missing_text}")

    rag = RagSettings(
        chunk_size=_get_int(rag_section, "chunk_size", "CHUNK_SIZE", 1000),
        chunk_overlap=_get_int(rag_section, "chunk_overlap", "CHUNK_OVERLAP", 200),
        retrieval_k=_get_int(rag_section, "retrieval_k", "RETRIEVAL_K", 4),
    )

    ml = MlSettings(
        random_seed=_get_int(ml_section, "random_seed", "RANDOM_SEED", 42),
    )

    llm_temperature = _get_float(
        {},
        "temperature",
        "LLM_TEMPERATURE",
        0.0,
    )

    return DSASettings(
        azure_openai=azure_openai,
        rag=rag,
        ml=ml,
        llm_temperature=llm_temperature,
    )


def get_azure_openai_config(
    *,
    require_credentials: bool = True,
) -> AzureOpenAISettings:
    """Return Azure OpenAI settings."""
    return load_settings(require_credentials=require_credentials).azure_openai


def ensure_api_key(*, prompt_if_missing: bool = False) -> str:
    """Return the Azure OpenAI API key.

    The default behavior avoids interactive prompts so CI jobs cannot hang.
    """
    settings = get_azure_openai_config(require_credentials=False)

    if settings.api_key:
        return settings.api_key

    if prompt_if_missing:
        import getpass

        key = getpass.getpass("Enter your Azure OpenAI API key: ")
        os.environ["AZURE_OPENAI_API_KEY"] = key
        return key

    raise RuntimeError("Missing Azure OpenAI API key.")


# ---------------------------------------------------------------------------
# Backward-compatible module constants
# ---------------------------------------------------------------------------
_DEFAULT_SETTINGS = load_settings(require_credentials=False)

AZURE_OPENAI_ENDPOINT = _DEFAULT_SETTINGS.azure_openai.endpoint
AZURE_OPENAI_API_KEY = _DEFAULT_SETTINGS.azure_openai.api_key
AZURE_OPENAI_API_VERSION = _DEFAULT_SETTINGS.azure_openai.api_version
AZURE_OPENAI_CHAT_DEPLOYMENT = _DEFAULT_SETTINGS.azure_openai.chat_deployment
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = _DEFAULT_SETTINGS.azure_openai.embedding_deployment
AZURE_OPENAI_CHAT_FALLBACK_DEPLOYMENT = (
    _DEFAULT_SETTINGS.azure_openai.chat_fallback_deployment
)

LLM_MODEL = AZURE_OPENAI_CHAT_DEPLOYMENT
LLM_FALLBACK = AZURE_OPENAI_CHAT_FALLBACK_DEPLOYMENT or ""
LLM_TEMPERATURE = _DEFAULT_SETTINGS.llm_temperature
EMBEDDING_MODEL = AZURE_OPENAI_EMBEDDING_DEPLOYMENT

CHUNK_SIZE = _DEFAULT_SETTINGS.rag.chunk_size
CHUNK_OVERLAP = _DEFAULT_SETTINGS.rag.chunk_overlap
RETRIEVAL_K = _DEFAULT_SETTINGS.rag.retrieval_k
RANDOM_SEED = _DEFAULT_SETTINGS.ml.random_seed