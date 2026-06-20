"""Prepare deterministic sample data for Azure ML pipeline runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from dsa.agents.sql_agent import create_sample_employees_database


def build_sample_dataframe() -> pd.DataFrame:
    """Return deterministic sample data for agent evaluation."""
    return pd.DataFrame(
        {
            "age": [
                25,
                27,
                29,
                31,
                33,
                35,
                37,
                39,
                41,
                43,
                45,
                47,
                49,
                51,
                53,
                55,
                57,
                59,
                61,
                63,
            ],
            "income": [
                42000,
                44000,
                46000,
                48000,
                50000,
                52000,
                54000,
                56000,
                58000,
                60000,
                72000,
                74000,
                76000,
                78000,
                80000,
                82000,
                84000,
                86000,
                88000,
                90000,
            ],
            "score": [
                1.0,
                1.2,
                1.3,
                1.5,
                1.7,
                1.8,
                2.0,
                2.1,
                2.3,
                2.5,
                4.0,
                4.2,
                4.3,
                4.5,
                4.7,
                4.8,
                5.0,
                5.1,
                5.3,
                5.5,
            ],
            "group": [
                "A",
                "A",
                "A",
                "A",
                "A",
                "A",
                "A",
                "A",
                "A",
                "A",
                "B",
                "B",
                "B",
                "B",
                "B",
                "B",
                "B",
                "B",
                "B",
                "B",
            ],
            "segment": [
                "low",
                "low",
                "low",
                "low",
                "low",
                "low",
                "low",
                "low",
                "low",
                "low",
                "high",
                "high",
                "high",
                "high",
                "high",
                "high",
                "high",
                "high",
                "high",
                "high",
            ],
            "city": [
                "Austin",
                "Boston",
                "Austin",
                "Denver",
                "Boston",
                "Austin",
                "Denver",
                "Boston",
                "Austin",
                "Denver",
                "Boston",
                "Austin",
                "Denver",
                "Boston",
                "Austin",
                "Denver",
                "Boston",
                "Austin",
                "Denver",
                "Boston",
            ],
            "target": [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
            ],
        }
    )


def write_eval_questions(path: Path) -> None:
    """Write deterministic evaluation prompts as JSONL."""
    questions = [
        {
            "id": "eda_rows",
            "question": "How many rows and columns are in the sample data?",
        },
        {
            "id": "stats_ttest",
            "question": "Does score differ by group?",
        },
        {
            "id": "sql_salary",
            "question": "What is the average employee salary?",
        },
    ]

    with path.open("w", encoding="utf-8") as file:
        for question in questions:
            file.write(json.dumps(question, sort_keys=True) + "\n")


def prepare_sample_data(output_data: str | Path) -> dict[str, str]:
    """Create sample CSV, SQLite database, eval questions, and manifest."""
    output_path = Path(output_data)
    output_path.mkdir(parents=True, exist_ok=True)

    sample_csv_path = output_path / "sample_agents.csv"
    sample_db_path = output_path / "sample.db"
    eval_questions_path = output_path / "eval_questions.jsonl"
    manifest_path = output_path / "manifest.json"

    sample_df = build_sample_dataframe()
    sample_df.to_csv(sample_csv_path, index=False)

    create_sample_employees_database(sample_db_path)
    write_eval_questions(eval_questions_path)

    manifest = {
        "sample_csv": sample_csv_path.name,
        "sample_db": sample_db_path.name,
        "eval_questions": eval_questions_path.name,
        "row_count": int(sample_df.shape[0]),
        "column_count": int(sample_df.shape[1]),
    }

    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return {
        "sample_csv": str(sample_csv_path),
        "sample_db": str(sample_db_path),
        "eval_questions": str(eval_questions_path),
        "manifest": str(manifest_path),
    }


def main() -> None:
    """Run the sample data preparation entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_data", required=True)
    args = parser.parse_args()

    outputs = prepare_sample_data(args.output_data)

    print("Prepared deterministic Azure ML sample data.")
    for output_name, output_path in outputs.items():
        print(f"{output_name}: {output_path}")


if __name__ == "__main__":
    main()