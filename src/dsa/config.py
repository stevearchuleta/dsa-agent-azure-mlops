"""
DSA Agent configuration.

Loads secrets from .env (or getpass fallback), defines project paths,
model preferences, and RAG parameters.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # data-agent/
DATA_DIR = PROJECT_ROOT / "data"
PAPERS_DIR = DATA_DIR / "papers"
FAISS_DIR = DATA_DIR / "faiss_index"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# Ensure directories exist
for _d in (PAPERS_DIR, FAISS_DIR, ARTIFACTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Secrets
# ---------------------------------------------------------------------------
load_dotenv(PROJECT_ROOT / ".env")


def ensure_api_key() -> str:
    """Return the OpenAI API key, prompting interactively if not set."""
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        import getpass
        key = getpass.getpass("Enter your OpenAI API key: ")
        os.environ["OPENAI_API_KEY"] = key
    return key


# ---------------------------------------------------------------------------
# Model preferences
# ---------------------------------------------------------------------------
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")
LLM_FALLBACK = os.getenv("LLM_FALLBACK", "gpt-4o-mini")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")

# ---------------------------------------------------------------------------
# RAG parameters
# ---------------------------------------------------------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "4"))