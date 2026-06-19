"""Tests for deterministic stats agent tools."""

from __future__ import annotations

import pandas as pd
import pytest

from dsa.agents.stats_agent import (
    format_hypothesis_result,
    interpret_p_value,
    run_independent_ttest,
)


def make_ttest_dataframe() -> pd.DataFrame:
    """Create a deterministic DataFrame for t-test checks."""
    return pd.DataFrame(
        {
            "group": ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"],
            "score": [1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            "city": [
                "Austin",
                "Austin",
                "Boston",
                "Boston",
                "Denver",
                "Austin",
                "Boston",
                "Denver",
                "Denver",
                "Denver",
            ],
        }
    )


def test_run_independent_ttest_returns_known_statistic_range():
    df = make_ttest_dataframe()

    result = run_independent_ttest(
        df,
        value_col="score",
        group_col="group",
        group_a="A",
        group_b="B",
    )

    assert result.value_column == "score"
    assert result.group_column == "group"
    assert result.group_a == "A"
    assert result.group_b == "B"
    assert result.group_a_count == 5
    assert result.group_b_count == 5
    assert result.group_a_mean == pytest.approx(3.0)
    assert result.group_b_mean == pytest.approx(7.0)
    assert result.t_statistic == pytest.approx(-4.0)
    assert 0.003 < result.p_value < 0.005
    assert result.reject_null is True
    assert result.conclusion == "Reject the null hypothesis."


def test_format_hypothesis_result_contains_required_fields():
    summary = format_hypothesis_result(
        null_hypothesis="Group means are equal.",
        alternate_hypothesis="Group means are different.",
        t_stat=-4.0,
        p_value=0.004,
        alpha=0.05,
    )

    assert "Null hypothesis: Group means are equal." in summary
    assert "Alternate hypothesis: Group means are different." in summary
    assert "t-statistic: -4.0000" in summary
    assert "p-value: 0.0040" in summary
    assert "alpha: 0.0500" in summary
    assert "conclusion: Reject the null hypothesis." in summary


def test_interpret_p_value_rejects_below_alpha():
    conclusion = interpret_p_value(0.01, alpha=0.05)

    assert conclusion == "Reject the null hypothesis."


def test_interpret_p_value_fails_to_reject_at_or_above_alpha():
    conclusion = interpret_p_value(0.05, alpha=0.05)

    assert conclusion == "Fail to reject the null hypothesis."


def test_run_independent_ttest_accepts_custom_hypotheses():
    df = make_ttest_dataframe()

    result = run_independent_ttest(
        df,
        value_col="score",
        group_col="group",
        group_a="A",
        group_b="B",
        null_hypothesis="Mean score is equal for A and B.",
        alternate_hypothesis="Mean score differs for A and B.",
    )

    assert result.null_hypothesis == "Mean score is equal for A and B."
    assert result.alternate_hypothesis == "Mean score differs for A and B."


def test_run_independent_ttest_rejects_missing_column():
    df = make_ttest_dataframe()

    with pytest.raises(KeyError, match="Column not found"):
        run_independent_ttest(
            df,
            value_col="missing_score",
            group_col="group",
            group_a="A",
            group_b="B",
        )


def test_run_independent_ttest_rejects_non_numeric_value_column():
    df = make_ttest_dataframe()

    with pytest.raises(TypeError, match="Column must be numeric"):
        run_independent_ttest(
            df,
            value_col="city",
            group_col="group",
            group_a="A",
            group_b="B",
        )


def test_run_independent_ttest_rejects_too_small_group():
    df = pd.DataFrame(
        {
            "group": ["A", "B", "B"],
            "score": [1.0, 2.0, 3.0],
        }
    )

    with pytest.raises(ValueError, match="group_a must contain at least two"):
        run_independent_ttest(
            df,
            value_col="score",
            group_col="group",
            group_a="A",
            group_b="B",
        )


def test_interpret_p_value_rejects_invalid_alpha():
    with pytest.raises(ValueError, match="alpha must be greater than 0"):
        interpret_p_value(0.01, alpha=0.0)