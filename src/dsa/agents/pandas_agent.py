"""
Pandas-based EDA agent.

Wraps LangChain's experimental Pandas agent to answer
natural-language questions about DataFrames.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

from dsa.llm import get_llm

logger = logging.getLogger(__name__)


def build_pandas_agent(
    df: pd.DataFrame,
    *,
    verbose: bool = True,
    allow_dangerous_code: bool = True,
) -> Any:
    """Create a Pandas DataFrame agent.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame the agent will query.
    verbose : bool
        If True, print intermediate reasoning steps.
    allow_dangerous_code : bool
        Required True for LangChain >= 0.3 Pandas agents.

    Returns
    -------
    AgentExecutor
        A LangChain agent that can answer questions about the DataFrame.
    """
    llm = get_llm(fallback=False)
    agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=verbose,
        allow_dangerous_code=allow_dangerous_code,
        agent_type="openai-tools",
    )
    logger.info("Pandas agent built for DataFrame with %d rows, %d cols", *df.shape)
    return agent