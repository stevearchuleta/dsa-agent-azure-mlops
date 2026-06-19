"""Tests for deterministic EDA tools."""

from __future__ import annotations

import pandas as pd
import pytest

from dsa.agents.eda_tools import (
    categorical_summary,
    dataframe_overview,
    missing_values_summary,
    numeric_summary,
    target_summary,
)


def make_sample_dataframe() -> pd.DataFrame:
    """Create a small deterministic DataFrame for EDA tests."""
    return pd.DataFrame(
        {
            "age": [10.0, 20.0, None, 20.0],
            "income": [100.0, 200.0, 300.0, None],
            "city": ["Austin", "Boston", "Austin", None],
            "target": [1, 0, 1, 0],
        }
    )


def test_dataframe_overview_returns_expected_counts():
    df = make_sample_dataframe()

    overview = dataframe_overview(df)

    assert overview["row_count"] == 4
    assert overview["column_count"] == 4
    assert overview["columns"] == ["age", "income", "city", "target"]
    assert overview["numeric_columns"] == ["age", "income", "target"]
    assert overview["categorical_columns"] == ["city"]
    assert overview["missing_cell_count"] == 3
    assert overview["duplicate_row_count"] == 0


def test_missing_values_summary_returns_column_level_counts():
    df = make_sample_dataframe()

    summary = missing_values_summary(df).set_index("column")

    assert summary.loc["age", "missing_count"] == 1
    assert summary.loc["income", "missing_count"] == 1
    assert summary.loc["city", "missing_count"] == 1
    assert summary.loc["target", "missing_count"] == 0
    assert summary.loc["age", "missing_percent"] == 25.0


def test_numeric_summary_returns_describe_output():
    df = make_sample_dataframe()

    summary = numeric_summary(df).set_index("column")

    assert summary.loc["age", "count"] == 3
    assert summary.loc["age", "mean"] == pytest.approx(50.0 / 3.0)
    assert summary.loc["income", "max"] == 300.0
    assert summary.loc["target", "mean"] == 0.5


def test_categorical_summary_returns_top_values():
    df = make_sample_dataframe()

    summary = categorical_summary(df).set_index("column")

    assert summary.loc["city", "count"] == 3
    assert summary.loc["city", "missing_count"] == 1
    assert summary.loc["city", "unique_count"] == 2
    assert "Austin: 2" in summary.loc["city", "top_values"]


def test_target_summary_returns_value_counts():
    df = make_sample_dataframe()

    summary = target_summary(df, "target")

    assert summary["target_column"] == "target"
    assert summary["row_count"] == 4
    assert summary["missing_count"] == 0
    assert summary["unique_count"] == 2
    assert summary["value_counts"] == {"0": 2, "1": 2}


def test_target_summary_raises_for_missing_target():
    df = make_sample_dataframe()

    with pytest.raises(KeyError, match="Target column not found"):
        target_summary(df, "missing_target")


def test_eda_tools_reject_non_dataframe():
    with pytest.raises(TypeError, match="Expected a pandas DataFrame"):
        dataframe_overview({"not": "a dataframe"})  # type: ignore[arg-type]