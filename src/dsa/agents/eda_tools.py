"""
Deterministic EDA tools for pandas DataFrames.

These utilities provide stable, testable DataFrame summaries that an agent can
call safely. The functions do not execute arbitrary generated code and do not
make network calls.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def _validate_dataframe(df: pd.DataFrame) -> None:
    """Raise a helpful error when the supplied object is not a DataFrame."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")


def _column_names(df: pd.DataFrame) -> list[str]:
    """Return DataFrame column names as strings."""
    return [str(column) for column in df.columns]


def dataframe_overview(df: pd.DataFrame) -> dict[str, Any]:
    """Return a compact overview of a DataFrame."""
    _validate_dataframe(df)

    numeric_columns = df.select_dtypes(include="number").columns
    categorical_columns = df.select_dtypes(exclude="number").columns

    return {
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "columns": _column_names(df),
        "numeric_columns": [str(column) for column in numeric_columns],
        "categorical_columns": [str(column) for column in categorical_columns],
        "missing_cell_count": int(df.isna().sum().sum()),
        "duplicate_row_count": int(df.duplicated().sum()),
    }


def missing_values_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return missing-value counts and percentages by column."""
    _validate_dataframe(df)

    missing_counts = df.isna().sum()
    missing_percent = df.isna().mean().mul(100).round(2)

    return pd.DataFrame(
        {
            "column": [str(column) for column in df.columns],
            "missing_count": missing_counts.astype(int).to_list(),
            "missing_percent": missing_percent.astype(float).to_list(),
        }
    )


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return deterministic summary statistics for numeric columns."""
    _validate_dataframe(df)

    numeric_df = df.select_dtypes(include="number")

    summary_columns = [
        "column",
        "count",
        "mean",
        "std",
        "min",
        "25%",
        "50%",
        "75%",
        "max",
    ]

    if numeric_df.empty:
        return pd.DataFrame(columns=summary_columns)

    summary = numeric_df.describe().T.reset_index()
    summary = summary.rename(columns={"index": "column"})
    summary["column"] = summary["column"].astype(str)

    return summary[summary_columns]


def categorical_summary(
    df: pd.DataFrame,
    *,
    max_categories: int = 10,
) -> pd.DataFrame:
    """Return deterministic summary statistics for non-numeric columns."""
    _validate_dataframe(df)

    if max_categories <= 0:
        raise ValueError("max_categories must be positive.")

    categorical_df = df.select_dtypes(exclude="number")

    records: list[dict[str, Any]] = []

    for column in categorical_df.columns:
        series = categorical_df[column]
        normalized_series = series.astype("string").fillna("<NA>")
        value_counts = normalized_series.value_counts(dropna=False).head(max_categories)

        top_values = ", ".join(
            f"{str(value)}: {int(count)}" for value, count in value_counts.items()
        )

        records.append(
            {
                "column": str(column),
                "count": int(series.count()),
                "missing_count": int(series.isna().sum()),
                "unique_count": int(series.nunique(dropna=True)),
                "top_values": top_values,
            }
        )

    return pd.DataFrame(
        records,
        columns=[
            "column",
            "count",
            "missing_count",
            "unique_count",
            "top_values",
        ],
    )


def target_summary(
    df: pd.DataFrame,
    target_column: str,
) -> dict[str, Any]:
    """Return a deterministic summary for a target column."""
    _validate_dataframe(df)

    if target_column not in df.columns:
        raise KeyError(f"Target column not found: {target_column}")

    series = df[target_column]
    normalized_series = series.astype("string").fillna("<NA>")
    value_counts = normalized_series.value_counts(dropna=False)

    return {
        "target_column": str(target_column),
        "row_count": int(series.shape[0]),
        "missing_count": int(series.isna().sum()),
        "unique_count": int(series.nunique(dropna=True)),
        "value_counts": {
            str(value): int(count) for value, count in value_counts.items()
        },
    }