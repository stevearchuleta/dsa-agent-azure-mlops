"""
Citation-aware RAG chain.

Builds a retrieval-augmented generation chain that answers
questions using PDF content and includes source citations.
"""

from __future__ import annotations

import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from dsa.config import RETRIEVAL_K
from dsa.llm import get_llm
from dsa.rag.ingest import get_or_build_index
from dsa.rag.format import format_docs

logger = logging.getLogger(__name__)

RAG_SYSTEM_PROMPT = (
    "You are a research assistant. Use the following retrieved context "
    "to answer the question. Always cite sources using the bracketed "
    "numbers provided. If the context does not contain the answer, "
    "say so clearly.\n\n"
    "Context:\n{context}\n"
)


def build_rag_chain():
    """Build and return a citation-aware RAG chain.

    Returns
    -------
    Runnable
        A LangChain chain: question -> retriever -> LLM -> answer.
    """
    index = get_or_build_index()
    retriever = index.as_retriever(search_kwargs={"k": RETRIEVAL_K})

    prompt = ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | get_llm()
        | StrOutputParser()
    )

    logger.info("RAG chain built (k=%d)", RETRIEVAL_K)
    return chain