"""Tests for deterministic plot agent tools."""

from __future__ import annotations

import pandas as pd
import pytest

from dsa.agents.plot_agent import (
    describe_plot,
    plot_correlation_heatmap,
    plot_histogram_by_target,
    plot_kde_by_target,
)


def make_plot_dataframe() -> pd.DataFrame:
    """Create a small deterministic DataFrame for plot tests."""
    return pd.DataFrame(
        {
            "age": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "income": [100.0, 180.0, 260.0, 330.0, 410.0, 500.0],
            "score": [1.0, 1.5, 2.5, 3.5, 4.0, 5.0],
            "target": [0, 0, 1, 1, 1, 0],
            "city": ["Austin", "Boston", "Austin", "Denver", "Denver", "Boston"],
        }
    )


def assert_png_file(path):
    """Assert that a path points to a non-empty PNG file."""
    assert path.exists()
    assert path.suffix == ".png"
    assert path.stat().st_size > 0


def test_plot_histogram_by_target_creates_png(tmp_path):
    df = make_plot_dataframe()
    output_path = tmp_path / "histogram.png"

    result = plot_histogram_by_target(
        df,
        value_col="age",
        target_col="target",
        output_path=output_path,
        bins=3,
    )

    assert result.plot_type == "histogram_by_target"
    assert result.value_column == "age"
    assert result.target_column == "target"
    assert result.path == output_path
    assert_png_file(result.path)


def test_plot_kde_by_target_creates_png(tmp_path):
    df = make_plot_dataframe()
    output_path = tmp_path / "kde.png"

    result = plot_kde_by_target(
        df,
        value_col="income",
        target_col="target",
        output_path=output_path,
    )

    assert result.plot_type == "kde_by_target"
    assert result.value_column == "income"
    assert result.target_column == "target"
    assert result.path == output_path
    assert_png_file(result.path)


def test_plot_correlation_heatmap_creates_png_and_matrix(tmp_path):
    df = make_plot_dataframe()
    output_path = tmp_path / "correlation.png"

    result = plot_correlation_heatmap(df, output_path=output_path)

    assert result.plot_type == "correlation_heatmap"
    assert result.correlation_matrix is not None
    assert list(result.correlation_matrix.columns) == ["age", "income", "score", "target"]
    assert list(result.correlation_matrix.index) == ["age", "income", "score", "target"]
    assert result.correlation_matrix.loc["age", "age"] == pytest.approx(1.0)
    assert_png_file(result.path)


def test_describe_plot_returns_deterministic_commentary():
    dataframe_profile = {
        "row_count": 6,
        "column_count": 5,
        "numeric_columns": ["age", "income", "score", "target"],
    }

    description = describe_plot("correlation_heatmap", dataframe_profile)

    assert "Correlation heatmap" in description
    assert "4 numeric columns" in description


def test_plot_histogram_rejects_missing_column(tmp_path):
    df = make_plot_dataframe()

    with pytest.raises(KeyError, match="Column not found"):
        plot_histogram_by_target(
            df,
            value_col="missing",
            target_col="target",
            output_path=tmp_path / "bad.png",
        )


def test_plot_kde_rejects_non_numeric_value_column(tmp_path):
    df = make_plot_dataframe()

    with pytest.raises(TypeError, match="Column must be numeric"):
        plot_kde_by_target(
            df,
            value_col="city",
            target_col="target",
            output_path=tmp_path / "bad.png",
        )