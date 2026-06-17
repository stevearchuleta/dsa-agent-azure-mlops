"""
Artifact export utilities.

Save figures and tables to the artifacts/ directory
for inclusion in the capstone report.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from dsa.config import ARTIFACTS_DIR

logger = logging.getLogger(__name__)


def save_figure(
    fig: plt.Figure | None = None,
    filename: str = "figure.png",
    *,
    dpi: int = 150,
    tight: bool = True,
) -> Path:
    """Save a matplotlib figure to the artifacts directory.

    Parameters
    ----------
    fig : Figure, optional
        The figure to save. If None, saves the current active figure.
    filename : str
        Output filename (e.g., 'volatility_plot.png').
    dpi : int
        Resolution in dots per inch.
    tight : bool
        If True, use tight_layout before saving.

    Returns
    -------
    Path
        Absolute path to the saved file.
    """
    _fig = fig or plt.gcf()
    if tight:
        _fig.tight_layout()
    path = ARTIFACTS_DIR / filename
    _fig.savefig(path, dpi=dpi, bbox_inches="tight")
    logger.info("Figure saved: %s", path)
    return path


def save_table(
    df: pd.DataFrame,
    filename: str = "table.csv",
) -> Path:
    """Save a DataFrame to the artifacts directory as CSV.

    Parameters
    ----------
    df : pd.DataFrame
        The data to export.
    filename : str
        Output filename (e.g., 'summary_stats.csv').

    Returns
    -------
    Path
        Absolute path to the saved file.
    """
    path = ARTIFACTS_DIR / filename
    df.to_csv(path, index=True)
    logger.info("Table saved: %s (%d rows)", path, len(df))
    return path