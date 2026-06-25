import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from dsa.agents.router import (
    SUPPORTED_CATEGORIES,
    format_agent_response,
    route_question,
    run_agent,
    supported_routes,
)


def _create_router_test_database(database_path: Path) -> Path:
    connection = sqlite3.connect(database_path)

    try:
        connection.execute(
            """
            create table employees (
                id integer primary key,
                name text not null,
                department text not null
            )
            """
        )
        connection.executemany(
            "insert into employees (name, department) values (?, ?)",
            [
                ("Ava", "Analytics"),
                ("Ben", "Engineering"),
                ("Cy", "Analytics"),
            ],
        )
        connection.commit()
    finally:
        connection.close()

    return database_path


@pytest.mark.parametrize(
    ("question", "expected_category"),
    [
        ("show a dataframe overview and missing values", "eda"),
        ("plot a histogram of age by target", "plot"),
        ("run an independent t-test and interpret the p-value", "stats"),
        ("train a random forest classifier and report AUC", "ml"),
        ("SQL: select * from employees", "sql"),
        ("retrieve document chunks from a FAISS vector index", "rag"),
    ],
)
def test_route_question_returns_expected_category(
    question: str,
    expected_category: str,
) -> None:
    route_result = route_question(question)

    assert route_result.category == expected_category
    assert route_result.label
    assert route_result.explanation
    assert route_result.available_categories == SUPPORTED_CATEGORIES


def test_unknown_question_returns_unknown_with_categories() -> None:
    route_result = route_question("please make coffee")

    assert route_result.category == "unknown"
    assert "eda" in route_result.available_categories
    assert "unknown" in route_result.available_categories


def test_empty_question_returns_help_with_categories() -> None:
    route_result = route_question("")

    assert route_result.category == "help"
    assert "sql" in route_result.available_categories
    assert route_result.required_inputs == ()


def test_supported_routes_are_in_priority_order() -> None:
    route_categories = [route_rule.category for route_rule in supported_routes()]

    assert route_categories == ["sql", "stats", "ml", "plot", "eda", "rag"]


def test_run_agent_dispatches_eda_when_dataframe_is_present() -> None:
    df = pd.DataFrame(
        {
            "age": [31, 42, 28],
            "target": ["yes", "no", "yes"],
        }
    )

    agent_response = run_agent("show a dataframe overview", df=df)

    assert agent_response.route.category == "eda"
    assert agent_response.dispatched is True
    assert agent_response.tool_name == "dataframe_overview"
    assert agent_response.result["row_count"] == 3


def test_run_agent_gates_plot_when_inputs_are_missing() -> None:
    agent_response = run_agent("plot a histogram of age by target")

    assert agent_response.route.category == "plot"
    assert agent_response.dispatched is False
    assert agent_response.tool_name == "plot_histogram_by_target"
    assert agent_response.missing_inputs == ("df", "value_col", "target_col")


def test_run_agent_dispatches_sql_through_read_only_path(tmp_path) -> None:
    database_path = _create_router_test_database(tmp_path / "employees.sqlite")

    agent_response = run_agent(
        "SQL: select * from employees",
        database_path=database_path,
        max_rows=2,
    )

    assert agent_response.route.category == "sql"
    assert agent_response.dispatched is True
    assert agent_response.tool_name == "execute_read_only_query"
    assert isinstance(agent_response.result, list)
    assert len(agent_response.result) <= 2


def test_run_agent_returns_error_for_unsafe_sql(tmp_path) -> None:
    database_path = _create_router_test_database(tmp_path / "employees.sqlite")

    agent_response = run_agent(
        "SQL: drop table employees",
        database_path=database_path,
        sql="drop table employees",
    )

    assert agent_response.route.category == "sql"
    assert agent_response.dispatched is False
    assert agent_response.error is not None
    assert "read-only" in agent_response.error


def test_format_agent_response_includes_missing_inputs() -> None:
    agent_response = run_agent("train a random forest classifier")

    formatted_response = format_agent_response(agent_response)

    assert "Category: ml" in formatted_response
    assert "Dispatched: no" in formatted_response
    assert "Missing inputs: df, target_col" in formatted_response