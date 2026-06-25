"""Command-line interface for the deterministic DSA agent router."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from dsa.agents.router import (
    format_agent_response,
    run_agent,
    supported_routes,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the DSA agent CLI argument parser."""

    parser = argparse.ArgumentParser(
        prog="dsa-agent",
        description="Route a data science question and dispatch when possible.",
    )
    parser.add_argument(
        "question",
        nargs="*",
        help="Natural-language question to route.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Print the agent response as JSON.",
    )
    parser.add_argument(
        "--list-routes",
        action="store_true",
        help="Print supported deterministic route categories.",
    )
    parser.add_argument(
        "--no-dispatch",
        action="store_true",
        help="Classify the question without calling a selected tool.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="CSV file to load as the DataFrame input.",
    )
    parser.add_argument(
        "--database-path",
        "--db",
        type=Path,
        help="SQLite database path for SQL dispatch.",
    )
    parser.add_argument(
        "--sql",
        help="Read-only SQL statement for SQL dispatch.",
    )
    parser.add_argument(
        "--table-name",
        help="Table name for SQL schema dispatch.",
    )
    parser.add_argument(
        "--target-column",
        "--target-col",
        dest="target_col",
        help="Target column for EDA, plots, or ML.",
    )
    parser.add_argument(
        "--value-column",
        "--value-col",
        dest="value_col",
        help="Numeric value column for plots or statistical tests.",
    )
    parser.add_argument(
        "--group-column",
        "--group-col",
        dest="group_col",
        help="Grouping column for statistical tests.",
    )
    parser.add_argument(
        "--group-a",
        help="First group value for an independent t-test.",
    )
    parser.add_argument(
        "--group-b",
        help="Second group value for an independent t-test.",
    )
    parser.add_argument(
        "--p-value",
        type=float,
        help="P-value for p-value interpretation dispatch.",
    )
    parser.add_argument(
        "--feature-columns",
        "--feature-cols",
        nargs="*",
        dest="feature_cols",
        help="Feature columns for ML dispatch.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        help="Output path for plot dispatch.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=100,
        help="Maximum SQL rows to return.",
    )
    parser.add_argument(
        "--max-categories",
        type=int,
        default=10,
        help="Maximum categorical values to show in EDA summaries.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=20,
        help="Histogram bin count.",
    )
    parser.add_argument(
        "--method",
        default="pearson",
        help="Correlation method for heatmap dispatch.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Alpha value for statistics dispatch.",
    )
    parser.add_argument(
        "--equal-var",
        action="store_true",
        help="Use equal variance for independent t-test dispatch.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Test-set fraction for ML dispatch.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic ML dispatch.",
    )

    return parser


def _load_dataframe(csv_path: Path | None) -> Any:
    """Load a CSV file only when a CSV path is provided."""

    if csv_path is None:
        return None

    import pandas as pd

    return pd.read_csv(csv_path)


def format_supported_routes() -> str:
    """Format supported route categories for command-line display."""

    lines = ["Supported routes:"]

    for route_rule in supported_routes():
        lines.append(f"- {route_rule.category}: {route_rule.description}")

    lines.append("- help: Show supported route categories.")
    lines.append("- unknown: No deterministic route matched.")

    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the deterministic DSA agent router CLI."""

    parser = build_parser()
    parsed_args = parser.parse_args(argv)

    if parsed_args.list_routes:
        print(format_supported_routes())
        return 0

    question = " ".join(parsed_args.question).strip()
    df = _load_dataframe(parsed_args.csv)

    agent_response = run_agent(
        question,
        dispatch=not parsed_args.no_dispatch,
        df=df,
        database_path=parsed_args.database_path,
        sql=parsed_args.sql,
        table_name=parsed_args.table_name,
        target_col=parsed_args.target_col,
        value_col=parsed_args.value_col,
        group_col=parsed_args.group_col,
        group_a=parsed_args.group_a,
        group_b=parsed_args.group_b,
        p_value=parsed_args.p_value,
        feature_cols=parsed_args.feature_cols,
        output_path=parsed_args.output_path,
        max_rows=parsed_args.max_rows,
        max_categories=parsed_args.max_categories,
        bins=parsed_args.bins,
        method=parsed_args.method,
        alpha=parsed_args.alpha,
        equal_var=parsed_args.equal_var,
        test_size=parsed_args.test_size,
        seed=parsed_args.seed,
    )

    if parsed_args.as_json:
        print(json.dumps(agent_response.as_dict(), indent=2, sort_keys=True))
    else:
        print(format_agent_response(agent_response))

    return 1 if agent_response.error else 0


if __name__ == "__main__":
    raise SystemExit(main())