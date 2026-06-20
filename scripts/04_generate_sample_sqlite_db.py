"""
Generate the local sample SQLite database used by the SQL agent demo.

The generated .db file is intentionally ignored by Git. Commit this generator
script, not the binary database.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dsa.agents.sql_agent import create_sample_employees_database


def main() -> None:
    """Generate a deterministic sample SQLite database."""
    parser = argparse.ArgumentParser(description="Generate sample SQLite database.")
    parser.add_argument(
        "--output",
        default="data/sample/sample.db",
        help="Output SQLite database path.",
    )

    args = parser.parse_args()
    output_path = Path(args.output)
    generated_path = create_sample_employees_database(output_path)

    print(f"Generated sample SQLite database: {generated_path}")


if __name__ == "__main__":
    main()