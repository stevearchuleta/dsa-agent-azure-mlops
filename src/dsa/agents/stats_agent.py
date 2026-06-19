"""
Deterministic statistical testing tools for pandas DataFrames.

These utilities provide safe hypothesis-testing functionality without asking an
LLM to generate arbitrary statistical code. The first supported test is Welch's
independent two-sample t-test.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class TTestResult:
    """Structured result from an independent two-sample t-test."""

    value_column: str
    group_column: str
    group_a: Any
    group_b: Any
    null_hypothesis: str
    alternate_hypothesis: str
    t_statistic: float
    p_value: float
    alpha: float
    reject_null: bool
    conclusion: str
    group_a_count: int
    group_b_count: int
    group_a_mean: float
    group_b_mean: float
    equal_var: bool

    def to_dict(self) -> dict[str, Any]:
        """Return the result as a plain dictionary."""
        return asdict(self)


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


def _validate_alpha(alpha: float) -> None:
    """Raise a helpful error when alpha is outside the open interval (0, 1)."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be greater than 0 and less than 1.")


def _default_null_hypothesis(
    value_col: str,
    group_col: str,
    group_a: Any,
    group_b: Any,
) -> str:
    """Create a default null hypothesis."""
    return (
        f"The mean {value_col} is equal for {group_col}={group_a} "
        f"and {group_col}={group_b}."
    )


def _default_alternate_hypothesis(
    value_col: str,
    group_col: str,
    group_a: Any,
    group_b: Any,
) -> str:
    """Create a default alternate hypothesis."""
    return (
        f"The mean {value_col} differs for {group_col}={group_a} "
        f"and {group_col}={group_b}."
    )


def interpret_p_value(p_value: float, alpha: float = 0.05) -> str:
    """Return the deterministic hypothesis-test decision."""
    _validate_alpha(alpha)

    if p_value < alpha:
        return "Reject the null hypothesis."

    return "Fail to reject the null hypothesis."


def format_hypothesis_result(
    null_hypothesis: str,
    alternate_hypothesis: str,
    t_stat: float,
    p_value: float,
    alpha: float = 0.05,
) -> str:
    """Return a deterministic plain-English hypothesis-test summary."""
    conclusion = interpret_p_value(p_value, alpha)

    return (
        f"Null hypothesis: {null_hypothesis}\n"
        f"Alternate hypothesis: {alternate_hypothesis}\n"
        f"t-statistic: {t_stat:.4f}\n"
        f"p-value: {p_value:.4f}\n"
        f"alpha: {alpha:.4f}\n"
        f"conclusion: {conclusion}"
    )


def run_independent_ttest(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    group_a: Any,
    group_b: Any,
    *,
    alpha: float = 0.05,
    equal_var: bool = False,
    null_hypothesis: str | None = None,
    alternate_hypothesis: str | None = None,
) -> TTestResult:
    """Run a deterministic independent two-sample t-test.

    Welch's t-test is used by default because equal population variance is often
    not a safe assumption in exploratory data analysis.
    """
    _validate_dataframe(df)
    _validate_numeric_column(df, value_col)
    _validate_column(df, group_col)
    _validate_alpha(alpha)

    group_a_values = df.loc[df[group_col] == group_a, value_col].dropna().astype(float)
    group_b_values = df.loc[df[group_col] == group_b, value_col].dropna().astype(float)

    if group_a_values.shape[0] < 2:
        raise ValueError("group_a must contain at least two non-missing values.")

    if group_b_values.shape[0] < 2:
        raise ValueError("group_b must contain at least two non-missing values.")

    t_statistic, p_value = stats.ttest_ind(
        group_a_values.to_numpy(),
        group_b_values.to_numpy(),
        equal_var=equal_var,
        nan_policy="omit",
    )

    resolved_null_hypothesis = null_hypothesis or _default_null_hypothesis(
        value_col,
        group_col,
        group_a,
        group_b,
    )
    resolved_alternate_hypothesis = (
        alternate_hypothesis
        or _default_alternate_hypothesis(value_col, group_col, group_a, group_b)
    )

    numeric_t_statistic = float(t_statistic)
    numeric_p_value = float(p_value)
    reject_null = numeric_p_value < alpha
    conclusion = interpret_p_value(numeric_p_value, alpha)

    return TTestResult(
        value_column=value_col,
        group_column=group_col,
        group_a=group_a,
        group_b=group_b,
        null_hypothesis=resolved_null_hypothesis,
        alternate_hypothesis=resolved_alternate_hypothesis,
        t_statistic=numeric_t_statistic,
        p_value=numeric_p_value,
        alpha=alpha,
        reject_null=reject_null,
        conclusion=conclusion,
        group_a_count=int(group_a_values.shape[0]),
        group_b_count=int(group_b_values.shape[0]),
        group_a_mean=float(group_a_values.mean()),
        group_b_mean=float(group_b_values.mean()),
        equal_var=equal_var,
    )