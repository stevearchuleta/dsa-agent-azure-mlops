"""
Deterministic plotting tools for pandas DataFrames.

These functions generate stable matplotlib artifacts without asking an LLM to
write arbitrary plotting code. The functions are intended for safe agent tool
use and CI-friendly tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dsa.export.artifacts import save_figure


@dataclass(frozen=True)
class PlotResult:
    """Metadata returned by deterministic plotting functions."""

    path: Path
    plot_type: str
    value_column: str | None = None
    target_column: str | None = None
    correlation_matrix: pd.DataFrame | None = None


def _validate_dataframe(df: pd.DataFrame) -> None:
    """Raise a helpful error when the supplied object is not a DataFrame."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")


def _validate_column(df: pd.DataFrame, column: str) -> None:
    """Raise a helpful error when a required column is missing."""
    if column not in df.columns:
        raise KeyError(f"Column not found: {column}")


def _validate_numeric_column(df: pd.DataFrame, column: str) -> None:
    """Raise a helpful error when a column is missing or non-numeric."""
    _validate_column(df, column)

    if not pd.api.types.is_numeric_dtype(df[column]):
        raise TypeError(f"Column must be numeric: {column}")


def _sorted_group_values(series: pd.Series) -> list[Any]:
    """Return stable group values with missing values removed."""
    values = series.dropna().unique().tolist()
    return sorted(values, key=lambda value: str(value))


def _save_plot(
    fig: plt.Figure,
    output_path: str | Path | None,
    default_filename: str,
) -> Path:
    """Save a matplotlib figure and close the figure."""
    if output_path is None:
        saved_path = save_figure(fig=fig, filename=default_filename)
        plt.close(fig)
        return saved_path

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _gaussian_kde(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Return a simple deterministic Gaussian KDE without scipy."""
    clean_values = values[np.isfinite(values)]

    if clean_values.size == 0:
        raise ValueError("KDE requires at least one finite numeric value.")

    value_std = float(np.std(clean_values, ddof=1)) if clean_values.size > 1 else 0.0

    if value_std > 0.0:
        bandwidth = 1.06 * value_std * clean_values.size ** (-1.0 / 5.0)
    else:
        bandwidth = max(abs(float(clean_values.mean())) * 0.1, 1.0)

    scaled = (grid[:, None] - clean_values[None, :]) / bandwidth
    density = np.exp(-0.5 * scaled**2).sum(axis=1)
    density = density / (clean_values.size * bandwidth * np.sqrt(2.0 * np.pi))

    return density


def plot_histogram_by_target(
    df: pd.DataFrame,
    value_col: str,
    target_col: str,
    output_path: str | Path | None = None,
    *,
    bins: int = 20,
) -> PlotResult:
    """Create a deterministic histogram grouped by a target column."""
    _validate_dataframe(df)
    _validate_numeric_column(df, value_col)
    _validate_column(df, target_col)

    if bins <= 0:
        raise ValueError("bins must be positive.")

    groups = _sorted_group_values(df[target_col])

    if not groups:
        raise ValueError("Histogram requires at least one non-missing target group.")

    fig, ax = plt.subplots(figsize=(8, 5))

    for group_value in groups:
        group_series = df.loc[df[target_col] == group_value, value_col].dropna()

        if group_series.empty:
            continue

        ax.hist(
            group_series.to_numpy(dtype=float),
            bins=bins,
            alpha=0.5,
            label=str(group_value),
        )

    ax.set_title(f"Histogram of {value_col} by {target_col}")
    ax.set_xlabel(value_col)
    ax.set_ylabel("Frequency")
    ax.legend(title=target_col)

    path = _save_plot(
        fig,
        output_path,
        f"histogram_{value_col}_by_{target_col}.png",
    )

    return PlotResult(
        path=path,
        plot_type="histogram_by_target",
        value_column=value_col,
        target_column=target_col,
    )


def plot_kde_by_target(
    df: pd.DataFrame,
    value_col: str,
    target_col: str,
    output_path: str | Path | None = None,
) -> PlotResult:
    """Create a deterministic KDE plot grouped by a target column."""
    _validate_dataframe(df)
    _validate_numeric_column(df, value_col)
    _validate_column(df, target_col)

    all_values = df[value_col].dropna().to_numpy(dtype=float)

    if all_values.size == 0:
        raise ValueError("KDE plot requires at least one non-missing numeric value.")

    value_min = float(np.min(all_values))
    value_max = float(np.max(all_values))

    if value_min == value_max:
        value_min -= 1.0
        value_max += 1.0

    grid = np.linspace(value_min, value_max, 200)
    groups = _sorted_group_values(df[target_col])

    if not groups:
        raise ValueError("KDE plot requires at least one non-missing target group.")

    fig, ax = plt.subplots(figsize=(8, 5))

    for group_value in groups:
        group_values = df.loc[df[target_col] == group_value, value_col].dropna()
        numeric_group_values = group_values.to_numpy(dtype=float)

        if numeric_group_values.size == 0:
            continue

        density = _gaussian_kde(numeric_group_values, grid)
        ax.plot(grid, density, label=str(group_value))

    ax.set_title(f"KDE of {value_col} by {target_col}")
    ax.set_xlabel(value_col)
    ax.set_ylabel("Density")
    ax.legend(title=target_col)

    path = _save_plot(
        fig,
        output_path,
        f"kde_{value_col}_by_{target_col}.png",
    )

    return PlotResult(
        path=path,
        plot_type="kde_by_target",
        value_column=value_col,
        target_column=target_col,
    )


def plot_correlation_heatmap(
    df: pd.DataFrame,
    output_path: str | Path | None = None,
    *,
    method: str = "pearson",
) -> PlotResult:
    """Create a deterministic correlation heatmap for numeric columns."""
    _validate_dataframe(df)

    numeric_df = df.select_dtypes(include="number")

    if numeric_df.shape[1] < 2:
        raise ValueError("Correlation heatmap requires at least two numeric columns.")

    correlation_matrix = numeric_df.corr(method=method)

    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(correlation_matrix.to_numpy(), aspect="auto")

    ax.set_title(f"{method.title()} Correlation Heatmap")
    ax.set_xticks(range(len(correlation_matrix.columns)))
    ax.set_yticks(range(len(correlation_matrix.index)))
    ax.set_xticklabels(correlation_matrix.columns, rotation=45, ha="right")
    ax.set_yticklabels(correlation_matrix.index)

    for row_index, row_name in enumerate(correlation_matrix.index):
        for col_index, col_name in enumerate(correlation_matrix.columns):
            value = correlation_matrix.loc[row_name, col_name]
            ax.text(
                col_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
            )

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    path = _save_plot(
        fig,
        output_path,
        f"correlation_heatmap_{method}.png",
    )

    return PlotResult(
        path=path,
        plot_type="correlation_heatmap",
        correlation_matrix=correlation_matrix,
    )


def describe_plot(
    plot_type: str,
    dataframe_profile: dict[str, Any],
) -> str:
    """Return deterministic plain-English plot commentary."""
    row_count = dataframe_profile.get("row_count", "unknown")
    column_count = dataframe_profile.get("column_count", "unknown")
    numeric_columns = dataframe_profile.get("numeric_columns", [])

    if plot_type == "histogram_by_target":
        return (
            "Histogram by target compares value distributions across groups. "
            f"The profiled data has {row_count} rows and {column_count} columns."
        )

    if plot_type == "kde_by_target":
        return (
            "KDE by target compares smoothed distributions across groups. "
            f"The profiled data has {row_count} rows and {len(numeric_columns)} "
            "numeric columns."
        )

    if plot_type == "correlation_heatmap":
        return (
            "Correlation heatmap shows pairwise numeric relationships. "
            f"The profiled data has {len(numeric_columns)} numeric columns."
        )

    return (
        f"{plot_type} plot generated for a data profile with {row_count} rows "
        f"and {column_count} columns."
    )