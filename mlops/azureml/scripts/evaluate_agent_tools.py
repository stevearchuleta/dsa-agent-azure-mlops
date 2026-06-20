"""Evaluate deterministic DSA agent tools inside an Azure ML job."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dsa.agents.eda_tools import (
    categorical_summary,
    dataframe_overview,
    missing_values_summary,
    numeric_summary,
    target_summary,
)
from dsa.agents.ml_agent import (
    evaluate_binary_classifier,
    prepare_binary_classification_data,
    train_logistic_regression,
    train_random_forest,
)
from dsa.agents.plot_agent import (
    describe_plot,
    plot_correlation_heatmap,
    plot_histogram_by_target,
    plot_kde_by_target,
)
from dsa.agents.sql_agent import (
    create_sqlite_engine,
    execute_read_only_query,
)
from dsa.agents.stats_agent import run_independent_ttest


def json_safe(value: Any) -> Any:
    """Convert objects into JSON-safe values."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}

    if isinstance(value, list | tuple):
        return [json_safe(item) for item in value]

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")

    if isinstance(value, np.generic):
        return value.item()

    if hasattr(value, "to_dict"):
        return json_safe(value.to_dict())

    return value


def write_markdown_report(report: dict[str, Any], output_path: Path) -> None:
    """Write a deterministic Markdown summary."""
    lines = [
        "# DSA Agent Azure ML Evaluation Report",
        "",
        f"Overall passed: `{report['passed']}`",
        "",
        "## Checks",
        "",
        "| Check | Passed |",
        "|---|---:|",
    ]

    for check_name, passed in report["checks"].items():
        lines.append(f"| {check_name} | {passed} |")

    lines.extend(
        [
            "",
            "## Metrics",
            "",
            f"- Logistic regression accuracy: {report['ml']['logistic_metrics']['accuracy']:.4f}",
            f"- Random forest accuracy: {report['ml']['random_forest_metrics']['accuracy']:.4f}",
            f"- SQL average salary: {report['sql']['average_salary']:.2f}",
            f"- T-test p-value: {report['stats']['p_value']:.6f}",
        ]
    )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_agent_tools(input_data: str | Path, output_report: str | Path) -> dict[str, Any]:
    """Run deterministic agent checks and write report artifacts."""
    input_path = Path(input_data)
    output_path = Path(output_report)
    plot_path = output_path / "plots"

    output_path.mkdir(parents=True, exist_ok=True)
    plot_path.mkdir(parents=True, exist_ok=True)

    sample_csv_path = input_path / "sample_agents.csv"
    sample_db_path = input_path / "sample.db"

    df = pd.read_csv(sample_csv_path)

    overview = dataframe_overview(df)
    missing_summary = missing_values_summary(df)
    numeric_stats = numeric_summary(df)
    category_stats = categorical_summary(df)
    target_stats = target_summary(df, "target")

    histogram = plot_histogram_by_target(
        df,
        value_col="score",
        target_col="target",
        output_path=plot_path / "histogram_score_by_target.png",
    )
    kde = plot_kde_by_target(
        df,
        value_col="score",
        target_col="target",
        output_path=plot_path / "kde_score_by_target.png",
    )
    heatmap = plot_correlation_heatmap(
        df,
        output_path=plot_path / "correlation_heatmap.png",
    )

    plot_description = describe_plot("correlation_heatmap", overview)

    ttest = run_independent_ttest(
        df,
        value_col="score",
        group_col="group",
        group_a="A",
        group_b="B",
    )

    ml_data = prepare_binary_classification_data(
        df,
        target_col="target",
        feature_cols=["age", "income", "score", "segment", "city"],
        seed=42,
    )
    logistic_model = train_logistic_regression(ml_data, seed=42)
    random_forest_model = train_random_forest(ml_data, seed=42, n_estimators=20)

    logistic_metrics = evaluate_binary_classifier(logistic_model, ml_data)
    random_forest_metrics = evaluate_binary_classifier(random_forest_model, ml_data)

    engine = create_sqlite_engine(sample_db_path)
    salary_rows = execute_read_only_query(
        engine,
        """
        SELECT
            COUNT(*) AS employee_count,
            AVG(Salary) AS average_salary,
            SUM(Salary) AS total_salary
        FROM Employees
        """,
    )
    salary_summary = salary_rows[0]

    checks = {
        "overview_has_20_rows": overview["row_count"] == 20,
        "missing_summary_has_rows": not missing_summary.empty,
        "numeric_summary_has_rows": not numeric_stats.empty,
        "categorical_summary_has_rows": not category_stats.empty,
        "target_summary_has_two_classes": target_stats["unique_count"] == 2,
        "histogram_png_exists": histogram.path.exists(),
        "kde_png_exists": kde.path.exists(),
        "heatmap_png_exists": heatmap.path.exists(),
        "ttest_rejects_null": bool(ttest.reject_null),
        "logistic_accuracy_in_range": 0.0 <= logistic_metrics["accuracy"] <= 1.0,
        "random_forest_accuracy_in_range": 0.0 <= random_forest_metrics["accuracy"] <= 1.0,
        "sql_employee_count_is_4": salary_summary["employee_count"] == 4,
    }

    report = {
        "passed": all(checks.values()),
        "checks": checks,
        "eda": {
            "overview": overview,
            "missing_summary": missing_summary,
            "numeric_summary": numeric_stats,
            "categorical_summary": category_stats,
            "target_summary": target_stats,
        },
        "plots": {
            "histogram": histogram.path,
            "kde": kde.path,
            "heatmap": heatmap.path,
            "description": plot_description,
        },
        "stats": ttest.to_dict(),
        "ml": {
            "logistic_metrics": logistic_metrics,
            "random_forest_metrics": random_forest_metrics,
        },
        "sql": salary_summary,
    }

    report_json_path = output_path / "evaluation_report.json"
    report_md_path = output_path / "evaluation_report.md"

    report_json_path.write_text(
        json.dumps(json_safe(report), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_markdown_report(json_safe(report), report_md_path)

    if not report["passed"]:
        raise SystemExit("DSA agent evaluation failed.")

    return report


def main() -> None:
    """Run deterministic agent evaluation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", required=True)
    parser.add_argument("--output_report", required=True)
    args = parser.parse_args()

    report = evaluate_agent_tools(args.input_data, args.output_report)

    print("DSA agent evaluation completed.")
    print(f"passed={report['passed']}")
    print(f"check_count={len(report['checks'])}")


if __name__ == "__main__":
    main()