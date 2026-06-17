"""Smoke tests for dsa package imports."""


def test_config_imports():
    from dsa.config import ensure_api_key, PROJECT_ROOT, PAPERS_DIR
    assert PROJECT_ROOT.exists()
    assert callable(ensure_api_key)
    assert PAPERS_DIR is not None


def test_llm_imports():
    from dsa.llm import get_llm
    assert callable(get_llm)


def test_embeddings_imports():
    from dsa.embeddings import get_embeddings
    assert callable(get_embeddings)


def test_rag_imports():
    from dsa.rag.ingest import load_pdfs, get_or_build_index
    from dsa.rag.format import format_docs
    assert callable(load_pdfs)
    assert callable(get_or_build_index)
    assert callable(format_docs)


def test_agents_imports():
    from dsa.agents.pandas_agent import build_pandas_agent
    assert callable(build_pandas_agent)


def test_export_imports():
    from dsa.export.artifacts import save_figure, save_table
    assert callable(save_figure)
    assert callable(save_table)