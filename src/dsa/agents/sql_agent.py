"""
Deterministic SQL tools for SQLite databases.

These utilities provide safe, read-only SQL querying through SQLAlchemy.
The module rejects mutating SQL and avoids arbitrary LLM-generated database
execution paths.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import Engine, create_engine, inspect, text


EMPLOYEE_ROWS: list[tuple[str, str, str, str, float]] = [
    ("John", "Doe", "1990-05-15", "HR", 50000.00),
    ("Jane", "Smith", "1985-12-10", "Sales", 55000.00),
    ("Bob", "Johnson", "1992-08-25", "Engineering", 60000.00),
    ("Alice", "Brown", "1988-04-03", "Marketing", 52000.00),
]


DANGEROUS_SQL_PATTERN = re.compile(
    r"\b(alter|attach|create|delete|detach|drop|insert|pragma|replace|"
    r"truncate|update|vacuum)\b",
    re.IGNORECASE,
)

COMMENT_SQL_PATTERN = re.compile(r"(--|/\*|\*/)")


def create_sample_employees_database(output_path: str | Path) -> Path:
    """Create a deterministic sample SQLite database for local tests and demos."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        path.unlink()

    connection = sqlite3.connect(path)

    try:
        cursor = connection.cursor()

        cursor.execute(
            """
            CREATE TABLE Employees (
                EmployeeID INTEGER PRIMARY KEY AUTOINCREMENT,
                FirstName TEXT NOT NULL,
                LastName TEXT NOT NULL,
                Birthdate TEXT NOT NULL,
                Department TEXT NOT NULL,
                Salary REAL NOT NULL
            )
            """
        )

        cursor.executemany(
            """
            INSERT INTO Employees
                (FirstName, LastName, Birthdate, Department, Salary)
            VALUES (?, ?, ?, ?, ?)
            """,
            EMPLOYEE_ROWS,
        )

        connection.commit()
    finally:
        connection.close()

    return path


def create_sqlite_engine(database_path: str | Path) -> Engine:
    """Create a SQLAlchemy engine for a local SQLite database."""
    path = Path(database_path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"SQLite database not found: {path}")

    return create_engine(f"sqlite:///{path.as_posix()}", future=True)


def list_tables(engine: Engine) -> list[str]:
    """Return table names from a SQLAlchemy engine."""
    return sorted(inspect(engine).get_table_names())


def get_table_schema(engine: Engine, table_name: str) -> list[dict[str, Any]]:
    """Return schema metadata for a table."""
    table_names = list_tables(engine)

    if table_name not in table_names:
        raise KeyError(f"Table not found: {table_name}")

    columns = inspect(engine).get_columns(table_name)

    return [
        {
            "name": str(column["name"]),
            "type": str(column["type"]),
            "nullable": bool(column["nullable"]),
            "default": column.get("default"),
            "primary_key": bool(column.get("primary_key", False)),
        }
        for column in columns
    ]


def validate_read_only_sql(sql: str) -> str:
    """Validate and normalize a single read-only SQL statement."""
    if not isinstance(sql, str):
        raise TypeError("SQL must be a string.")

    stripped_sql = sql.strip()

    if not stripped_sql:
        raise ValueError("SQL query cannot be empty.")

    if COMMENT_SQL_PATTERN.search(stripped_sql):
        raise ValueError("SQL comments are not allowed.")

    normalized_sql = stripped_sql[:-1].strip() if stripped_sql.endswith(";") else stripped_sql

    if ";" in normalized_sql:
        raise ValueError("Only one SQL statement is allowed.")

    first_token = normalized_sql.split(maxsplit=1)[0].lower()

    if first_token not in {"select", "with"}:
        raise ValueError("Only read-only SELECT queries are allowed.")

    if DANGEROUS_SQL_PATTERN.search(normalized_sql):
        raise ValueError("Only read-only SELECT queries are allowed.")

    return normalized_sql


def execute_read_only_query(
    engine: Engine,
    sql: str,
    *,
    max_rows: int = 100,
) -> list[dict[str, Any]]:
    """Execute a validated read-only SQL query and return rows as dictionaries."""
    if max_rows <= 0:
        raise ValueError("max_rows must be positive.")

    safe_sql = validate_read_only_sql(sql)

    with engine.connect() as connection:
        result = connection.execute(text(safe_sql))
        rows = result.mappings().fetchmany(max_rows)

    return [dict(row) for row in rows]


def query_to_dataframe(
    engine: Engine,
    sql: str,
    *,
    max_rows: int = 100,
) -> pd.DataFrame:
    """Execute a read-only SQL query and return a DataFrame."""
    rows = execute_read_only_query(engine, sql, max_rows=max_rows)
    return pd.DataFrame(rows)