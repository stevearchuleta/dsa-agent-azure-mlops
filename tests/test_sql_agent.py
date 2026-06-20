"""Tests for deterministic SQL agent tools."""

from __future__ import annotations

import pandas as pd
import pytest

from dsa.agents.sql_agent import (
    create_sample_employees_database,
    create_sqlite_engine,
    execute_read_only_query,
    get_table_schema,
    list_tables,
    query_to_dataframe,
    validate_read_only_sql,
)


def make_test_engine(tmp_path):
    """Create a deterministic sample database and SQLAlchemy engine."""
    database_path = create_sample_employees_database(tmp_path / "sample.db")
    return create_sqlite_engine(database_path)


def test_sample_database_has_employees_table(tmp_path):
    engine = make_test_engine(tmp_path)

    tables = list_tables(engine)

    assert tables == ["Employees"]


def test_employees_schema_returns_expected_columns(tmp_path):
    engine = make_test_engine(tmp_path)

    schema = get_table_schema(engine, "Employees")
    column_names = [column["name"] for column in schema]

    assert column_names == [
        "EmployeeID",
        "FirstName",
        "LastName",
        "Birthdate",
        "Department",
        "Salary",
    ]


def test_salary_aggregate_query_returns_expected_values(tmp_path):
    engine = make_test_engine(tmp_path)

    rows = execute_read_only_query(
        engine,
        """
        SELECT
            COUNT(*) AS employee_count,
            AVG(Salary) AS average_salary,
            SUM(Salary) AS total_salary,
            MAX(Salary) AS max_salary
        FROM Employees
        """,
    )

    assert rows[0]["employee_count"] == 4
    assert rows[0]["average_salary"] == pytest.approx(54250.0)
    assert rows[0]["total_salary"] == pytest.approx(217000.0)
    assert rows[0]["max_salary"] == pytest.approx(60000.0)


def test_query_to_dataframe_returns_dataframe(tmp_path):
    engine = make_test_engine(tmp_path)

    df = query_to_dataframe(
        engine,
        """
        SELECT Department, Salary
        FROM Employees
        WHERE Department = 'Engineering'
        """,
    )

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (1, 2)
    assert df.loc[0, "Department"] == "Engineering"
    assert df.loc[0, "Salary"] == pytest.approx(60000.0)


@pytest.mark.parametrize(
    "dangerous_sql",
    [
        "DROP TABLE Employees",
        "DELETE FROM Employees",
        "UPDATE Employees SET Salary = 0",
        "INSERT INTO Employees (FirstName) VALUES ('Bad')",
        "CREATE TABLE BadTable (id INTEGER)",
    ],
)
def test_dangerous_sql_is_rejected(tmp_path, dangerous_sql):
    engine = make_test_engine(tmp_path)

    with pytest.raises(ValueError, match="read-only SELECT"):
        execute_read_only_query(engine, dangerous_sql)


def test_multiple_sql_statements_are_rejected(tmp_path):
    engine = make_test_engine(tmp_path)

    with pytest.raises(ValueError, match="Only one SQL statement"):
        execute_read_only_query(
            engine,
            "SELECT * FROM Employees; SELECT * FROM Employees",
        )


def test_sql_comments_are_rejected():
    with pytest.raises(ValueError, match="comments are not allowed"):
        validate_read_only_sql("SELECT * FROM Employees -- hidden comment")


def test_missing_table_schema_raises_key_error(tmp_path):
    engine = make_test_engine(tmp_path)

    with pytest.raises(KeyError, match="Table not found"):
        get_table_schema(engine, "MissingTable")