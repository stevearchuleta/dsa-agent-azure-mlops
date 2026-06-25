"""Deterministic router and gated dispatcher for DSA agent tools."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Final


SUPPORTED_CATEGORIES: Final[tuple[str, ...]] = (
    "eda",
    "plot",
    "stats",
    "ml",
    "sql",
    "rag",
    "help",
    "unknown",
)


@dataclass(frozen=True)
class RouteRule:
    """Keyword rule for a deterministic route category."""

    category: str
    label: str
    description: str
    keywords: tuple[str, ...]


@dataclass(frozen=True)
class RouteResult:
    """Classification result from deterministic routing."""

    category: str
    label: str
    explanation: str
    matched_keywords: tuple[str, ...]
    selected_tool: str | None
    required_inputs: tuple[str, ...]
    available_categories: tuple[str, ...] = SUPPORTED_CATEGORIES

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe route dictionary."""

        return {
            "category": self.category,
            "label": self.label,
            "explanation": self.explanation,
            "matched_keywords": list(self.matched_keywords),
            "selected_tool": self.selected_tool,
            "required_inputs": list(self.required_inputs),
            "available_categories": list(self.available_categories),
        }


@dataclass(frozen=True)
class AgentResponse:
    """Classification plus optional gated dispatch response."""

    route: RouteResult
    dispatched: bool
    tool_name: str | None
    missing_inputs: tuple[str, ...]
    result: Any
    error: str | None
    message: str

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe agent response dictionary."""

        return {
            "route": self.route.as_dict(),
            "dispatched": self.dispatched,
            "tool_name": self.tool_name,
            "missing_inputs": list(self.missing_inputs),
            "result": _json_safe(self.result),
            "error": self.error,
            "message": self.message,
        }


ROUTE_RULES: Final[tuple[RouteRule, ...]] = (
    RouteRule(
        category="sql",
        label="SQL",
        description="Read-only database queries and schema inspection.",
        keywords=(
            "sql",
            "select",
            "query",
            "sqlite",
            "database",
            "table",
            "tables",
            "schema",
            "join",
            "where",
            "group by",
            "read only",
            "read-only",
        ),
    ),
    RouteRule(
        category="stats",
        label="Statistics",
        description="Hypothesis tests, p-values, and statistical decisions.",
        keywords=(
            "t test",
            "ttest",
            "hypothesis",
            "p value",
            "p-value",
            "alpha",
            "significant",
            "statistical",
            "statistics",
            "null hypothesis",
            "alternate hypothesis",
        ),
    ),
    RouteRule(
        category="ml",
        label="Machine learning",
        description="Binary classification preparation, training, and metrics.",
        keywords=(
            "machine learning",
            "ml",
            "train",
            "train model",
            "model",
            "classifier",
            "classification",
            "logistic regression",
            "random forest",
            "predict",
            "accuracy",
            "roc",
            "auc",
            "binary target",
        ),
    ),
    RouteRule(
        category="plot",
        label="Plotting",
        description="Deterministic matplotlib visualizations.",
        keywords=(
            "plot",
            "chart",
            "visualize",
            "visualization",
            "histogram",
            "kde",
            "heatmap",
            "correlation",
            "matplotlib",
            "figure",
            "graph",
        ),
    ),
    RouteRule(
        category="eda",
        label="Exploratory data analysis",
        description="DataFrame overview, missing values, and summaries.",
        keywords=(
            "eda",
            "overview",
            "dataframe",
            "data frame",
            "missing",
            "missing values",
            "nulls",
            "numeric summary",
            "categorical summary",
            "describe",
            "profile",
            "columns",
            "target summary",
        ),
    ),
    RouteRule(
        category="rag",
        label="Retrieval augmented generation",
        description="Document retrieval, embeddings, FAISS, and citations.",
        keywords=(
            "rag",
            "retrieval",
            "retrieve",
            "document",
            "documents",
            "paper",
            "papers",
            "pdf",
            "faiss",
            "vector",
            "embedding",
            "embeddings",
            "cite",
            "sources",
            "source",
        ),
    ),
)

HELP_KEYWORDS: Final[set[str]] = {
    "help",
    "route",
    "routes",
    "capabilities",
    "what can you do",
}


def _normalize_text(text: str) -> str:
    """Normalize text for deterministic keyword matching."""

    lowered_text = text.casefold()
    normalized_text = re.sub(r"[^a-z0-9]+", " ", lowered_text)
    return re.sub(r"\s+", " ", normalized_text).strip()


def _keyword_matches(normalized_question: str, keyword: str) -> bool:
    """Return True when a normalized keyword appears in a question."""

    normalized_keyword = _normalize_text(keyword)

    if not normalized_keyword:
        return False

    keyword_pattern = rf"\b{re.escape(normalized_keyword)}\b"
    return re.search(keyword_pattern, normalized_question) is not None


def _matched_keywords(question: str, route_rule: RouteRule) -> tuple[str, ...]:
    """Return all keywords from a rule that match a question."""

    normalized_question = _normalize_text(question)

    return tuple(
        keyword
        for keyword in route_rule.keywords
        if _keyword_matches(normalized_question, keyword)
    )


def _select_tool(category: str, question: str) -> str | None:
    """Select a deterministic tool name inside a routed category."""

    normalized_question = _normalize_text(question)

    if category == "eda":
        if "target" in normalized_question:
            return "target_summary"
        if "missing" in normalized_question or "null" in normalized_question:
            return "missing_values_summary"
        if "categorical" in normalized_question:
            return "categorical_summary"
        if "numeric" in normalized_question or "describe" in normalized_question:
            return "numeric_summary"
        return "dataframe_overview"

    if category == "plot":
        if "heatmap" in normalized_question or "correlation" in normalized_question:
            return "plot_correlation_heatmap"
        if "kde" in normalized_question:
            return "plot_kde_by_target"
        return "plot_histogram_by_target"

    if category == "stats":
        if "interpret" in normalized_question and "p value" in normalized_question:
            return "interpret_p_value"
        return "run_independent_ttest"

    if category == "ml":
        if "logistic" in normalized_question:
            return "train_logistic_regression"
        return "train_random_forest"

    if category == "sql":
        list_tables_requested = (
            "list tables" in normalized_question
            or "show tables" in normalized_question
        )
        if "schema" in normalized_question:
            return "get_table_schema"
        if list_tables_requested:
            return "list_tables"
        return "execute_read_only_query"

    if category == "rag":
        return None

    return None


def _required_inputs(selected_tool: str | None) -> tuple[str, ...]:
    """Return required dispatcher inputs for a selected tool."""

    required_by_tool: dict[str, tuple[str, ...]] = {
        "dataframe_overview": ("df",),
        "missing_values_summary": ("df",),
        "numeric_summary": ("df",),
        "categorical_summary": ("df",),
        "target_summary": ("df", "target_col"),
        "plot_histogram_by_target": ("df", "value_col", "target_col"),
        "plot_kde_by_target": ("df", "value_col", "target_col"),
        "plot_correlation_heatmap": ("df",),
        "interpret_p_value": ("p_value",),
        "run_independent_ttest": (
            "df",
            "value_col",
            "group_col",
            "group_a",
            "group_b",
        ),
        "train_logistic_regression": ("df", "target_col"),
        "train_random_forest": ("df", "target_col"),
        "list_tables": ("database_path",),
        "get_table_schema": ("database_path", "table_name"),
        "execute_read_only_query": ("database_path", "sql"),
    }

    if selected_tool is None:
        return ()

    return required_by_tool[selected_tool]


def supported_routes() -> tuple[RouteRule, ...]:
    """Return deterministic route rules in priority order."""

    return ROUTE_RULES


def route_question(question: str) -> RouteResult:
    """Classify a natural-language question into one deterministic category."""

    clean_question = question.strip()
    normalized_question = _normalize_text(clean_question)

    if not normalized_question or normalized_question in HELP_KEYWORDS:
        return RouteResult(
            category="help",
            label="Help",
            explanation="The question is empty or asks for supported routes.",
            matched_keywords=(),
            selected_tool=None,
            required_inputs=(),
        )

    scored_rules = [
        (route_rule, _matched_keywords(clean_question, route_rule))
        for route_rule in ROUTE_RULES
    ]
    matching_rules = [
        (route_rule, matches)
        for route_rule, matches in scored_rules
        if len(matches) > 0
    ]

    if not matching_rules:
        return RouteResult(
            category="unknown",
            label="Unknown",
            explanation="No deterministic keyword rule matched the question.",
            matched_keywords=(),
            selected_tool=None,
            required_inputs=(),
        )

    best_rule, best_matches = max(
        matching_rules,
        key=lambda rule_and_matches: len(rule_and_matches[1]),
    )
    selected_tool = _select_tool(best_rule.category, clean_question)

    return RouteResult(
        category=best_rule.category,
        label=best_rule.label,
        explanation=(
            f"Matched {best_rule.category} keyword(s): "
            f"{', '.join(best_matches)}."
        ),
        matched_keywords=best_matches,
        selected_tool=selected_tool,
        required_inputs=_required_inputs(selected_tool),
    )


def _extract_sql_candidate(question: str) -> str | None:
    """Extract an explicit SQL statement only when the question provides one."""

    marker_match = re.search(
        r"(?is)\bsql\s*:\s*((?:select|with)\b.*)",
        question,
    )

    if marker_match:
        return marker_match.group(1).strip()

    if re.match(r"(?is)^\s*(select|with)\b", question):
        return question.strip()

    return None


def _input_is_missing(value: Any) -> bool:
    """Return True when a dispatch input is absent."""

    if value is None:
        return True

    if isinstance(value, str):
        return value.strip() == ""

    return False


def _collect_missing_inputs(
    required_inputs: tuple[str, ...],
    input_values: dict[str, Any],
) -> tuple[str, ...]:
    """Return required inputs that are missing from a dispatch request."""

    return tuple(
        input_name
        for input_name in required_inputs
        if _input_is_missing(input_values.get(input_name))
    )


def _json_safe(value: Any) -> Any:
    """Convert common tool outputs into JSON-safe values."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Path):
        return str(value)

    if is_dataclass(value):
        return _json_safe(asdict(value))

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]

    if value.__class__.__name__ == "DataFrame" and hasattr(value, "to_dict"):
        return value.to_dict(orient="records")

    if value.__class__.__name__ == "Series" and hasattr(value, "to_list"):
        return value.to_list()

    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            return str(value)

    return str(value)


def _dispatch_eda(
    selected_tool: str,
    *,
    df: Any,
    target_col: str | None,
    max_categories: int,
) -> Any:
    """Dispatch an EDA route to the selected deterministic EDA tool."""

    from dsa.agents.eda_tools import (
        categorical_summary,
        dataframe_overview,
        missing_values_summary,
        numeric_summary,
        target_summary,
    )

    if selected_tool == "missing_values_summary":
        return missing_values_summary(df)

    if selected_tool == "numeric_summary":
        return numeric_summary(df)

    if selected_tool == "categorical_summary":
        return categorical_summary(df, max_categories=max_categories)

    if selected_tool == "target_summary":
        return target_summary(df, target_col or "")

    return dataframe_overview(df)


def _dispatch_plot(
    selected_tool: str,
    *,
    df: Any,
    value_col: str | None,
    target_col: str | None,
    output_path: str | Path | None,
    bins: int,
    method: str,
) -> Any:
    """Dispatch a plotting route to the selected deterministic plot tool."""

    from dsa.agents.plot_agent import (
        plot_correlation_heatmap,
        plot_histogram_by_target,
        plot_kde_by_target,
    )

    if selected_tool == "plot_correlation_heatmap":
        return plot_correlation_heatmap(df, output_path, method=method)

    if selected_tool == "plot_kde_by_target":
        return plot_kde_by_target(df, value_col or "", target_col or "", output_path)

    return plot_histogram_by_target(
        df,
        value_col or "",
        target_col or "",
        output_path,
        bins=bins,
    )


def _dispatch_stats(
    selected_tool: str,
    *,
    df: Any,
    value_col: str | None,
    group_col: str | None,
    group_a: Any,
    group_b: Any,
    p_value: float | None,
    alpha: float,
    equal_var: bool,
) -> Any:
    """Dispatch a statistics route to the selected deterministic stats tool."""

    from dsa.agents.stats_agent import interpret_p_value, run_independent_ttest

    if selected_tool == "interpret_p_value":
        return interpret_p_value(float(p_value), alpha=alpha)

    return run_independent_ttest(
        df,
        value_col or "",
        group_col or "",
        group_a,
        group_b,
        alpha=alpha,
        equal_var=equal_var,
    )


def _dispatch_ml(
    selected_tool: str,
    *,
    df: Any,
    target_col: str | None,
    feature_cols: list[str] | None,
    test_size: float,
    seed: int,
) -> dict[str, Any]:
    """Dispatch an ML route to deterministic binary classification tools."""

    from dsa.agents.ml_agent import (
        evaluate_binary_classifier,
        prepare_binary_classification_data,
        train_logistic_regression,
        train_random_forest,
    )

    data = prepare_binary_classification_data(
        df,
        target_col or "",
        feature_cols,
        test_size=test_size,
        seed=seed,
    )

    if selected_tool == "train_logistic_regression":
        model = train_logistic_regression(data, seed=seed)
        model_type = "logistic_regression"
    else:
        model = train_random_forest(data, seed=seed)
        model_type = "random_forest"

    return {
        "model_type": model_type,
        "metrics": evaluate_binary_classifier(model, data),
    }


def _dispatch_sql(
    selected_tool: str,
    *,
    database_path: str | Path,
    sql: str | None,
    table_name: str | None,
    max_rows: int,
) -> Any:
    """Dispatch a SQL route through the existing read-only SQL guard."""

    from dsa.agents.sql_agent import (
        create_sqlite_engine,
        execute_read_only_query,
        get_table_schema,
        list_tables,
        validate_read_only_sql,
    )

    engine = create_sqlite_engine(database_path)

    try:
        if selected_tool == "list_tables":
            return list_tables(engine)

        if selected_tool == "get_table_schema":
            return get_table_schema(engine, table_name or "")

        safe_sql = validate_read_only_sql(sql or "")
        return execute_read_only_query(engine, safe_sql, max_rows=max_rows)
    finally:
        engine.dispose()


def _dispatch_by_route(
    route: RouteResult,
    *,
    df: Any,
    database_path: str | Path | None,
    sql: str | None,
    table_name: str | None,
    target_col: str | None,
    value_col: str | None,
    group_col: str | None,
    group_a: Any,
    group_b: Any,
    p_value: float | None,
    feature_cols: list[str] | None,
    output_path: str | Path | None,
    max_rows: int,
    max_categories: int,
    bins: int,
    method: str,
    alpha: float,
    equal_var: bool,
    test_size: float,
    seed: int,
) -> Any:
    """Dispatch a classified route to a deterministic tool."""

    selected_tool = route.selected_tool

    if selected_tool is None:
        return None

    if route.category == "eda":
        return _dispatch_eda(
            selected_tool,
            df=df,
            target_col=target_col,
            max_categories=max_categories,
        )

    if route.category == "plot":
        return _dispatch_plot(
            selected_tool,
            df=df,
            value_col=value_col,
            target_col=target_col,
            output_path=output_path,
            bins=bins,
            method=method,
        )

    if route.category == "stats":
        return _dispatch_stats(
            selected_tool,
            df=df,
            value_col=value_col,
            group_col=group_col,
            group_a=group_a,
            group_b=group_b,
            p_value=p_value,
            alpha=alpha,
            equal_var=equal_var,
        )

    if route.category == "ml":
        return _dispatch_ml(
            selected_tool,
            df=df,
            target_col=target_col,
            feature_cols=feature_cols,
            test_size=test_size,
            seed=seed,
        )

    if route.category == "sql":
        return _dispatch_sql(
            selected_tool,
            database_path=database_path or "",
            sql=sql,
            table_name=table_name,
            max_rows=max_rows,
        )

    return None


def run_agent(
    question: str,
    *,
    dispatch: bool = True,
    df: Any = None,
    database_path: str | Path | None = None,
    sql: str | None = None,
    table_name: str | None = None,
    target_col: str | None = None,
    value_col: str | None = None,
    group_col: str | None = None,
    group_a: Any = None,
    group_b: Any = None,
    p_value: float | None = None,
    feature_cols: list[str] | None = None,
    output_path: str | Path | None = None,
    max_rows: int = 100,
    max_categories: int = 10,
    bins: int = 20,
    method: str = "pearson",
    alpha: float = 0.05,
    equal_var: bool = False,
    test_size: float = 0.25,
    seed: int = 42,
) -> AgentResponse:
    """Classify a question and dispatch only when required inputs exist."""

    route = route_question(question)

    if route.category in {"help", "unknown"}:
        return AgentResponse(
            route=route,
            dispatched=False,
            tool_name=None,
            missing_inputs=(),
            result=None,
            error=None,
            message=(
                "Ask about one available category: "
                + ", ".join(route.available_categories)
                + "."
            ),
        )

    if route.category == "rag":
        return AgentResponse(
            route=route,
            dispatched=False,
            tool_name=None,
            missing_inputs=(),
            result=None,
            error=None,
            message="RAG dispatch is classify-only in v1 to keep CI API-key free.",
        )

    if not dispatch:
        return AgentResponse(
            route=route,
            dispatched=False,
            tool_name=route.selected_tool,
            missing_inputs=(),
            result=None,
            error=None,
            message="Classified only because dispatch was disabled.",
        )

    if route.category == "sql" and sql is None:
        sql = _extract_sql_candidate(question)

    input_values = {
        "df": df,
        "database_path": database_path,
        "sql": sql,
        "table_name": table_name,
        "target_col": target_col,
        "value_col": value_col,
        "group_col": group_col,
        "group_a": group_a,
        "group_b": group_b,
        "p_value": p_value,
    }
    missing_inputs = _collect_missing_inputs(route.required_inputs, input_values)

    if missing_inputs:
        return AgentResponse(
            route=route,
            dispatched=False,
            tool_name=route.selected_tool,
            missing_inputs=missing_inputs,
            result=None,
            error=None,
            message="Missing required input(s): " + ", ".join(missing_inputs) + ".",
        )

    try:
        result = _dispatch_by_route(
            route,
            df=df,
            database_path=database_path,
            sql=sql,
            table_name=table_name,
            target_col=target_col,
            value_col=value_col,
            group_col=group_col,
            group_a=group_a,
            group_b=group_b,
            p_value=p_value,
            feature_cols=feature_cols,
            output_path=output_path,
            max_rows=max_rows,
            max_categories=max_categories,
            bins=bins,
            method=method,
            alpha=alpha,
            equal_var=equal_var,
            test_size=test_size,
            seed=seed,
        )
    except Exception as exc:
        return AgentResponse(
            route=route,
            dispatched=False,
            tool_name=route.selected_tool,
            missing_inputs=(),
            result=None,
            error=str(exc),
            message=f"Dispatch failed for {route.selected_tool}: {exc}",
        )

    return AgentResponse(
        route=route,
        dispatched=True,
        tool_name=route.selected_tool,
        missing_inputs=(),
        result=result,
        error=None,
        message=f"Dispatched to {route.selected_tool}.",
    )


def format_route_response(route: RouteResult) -> str:
    """Format a route result for command-line display."""

    lines = [
        f"Category: {route.category}",
        f"Label: {route.label}",
        f"Explanation: {route.explanation}",
    ]

    if route.selected_tool:
        lines.append(f"Selected tool: {route.selected_tool}")

    if route.required_inputs:
        lines.append("Required inputs: " + ", ".join(route.required_inputs))

    if route.matched_keywords:
        lines.append("Matched keywords: " + ", ".join(route.matched_keywords))

    if route.category in {"help", "unknown"}:
        lines.append(
            "Available categories: " + ", ".join(route.available_categories)
        )

    return "\n".join(lines)


def format_agent_response(agent_response: AgentResponse) -> str:
    """Format an agent response for command-line display."""

    route = agent_response.route
    lines = [
        format_route_response(route),
        f"Dispatched: {'yes' if agent_response.dispatched else 'no'}",
        f"Message: {agent_response.message}",
    ]

    if agent_response.missing_inputs:
        lines.append("Missing inputs: " + ", ".join(agent_response.missing_inputs))

    if agent_response.error:
        lines.append(f"Error: {agent_response.error}")

    if agent_response.result is not None:
        safe_result = _json_safe(agent_response.result)
        lines.append("Result:")
        lines.append(json.dumps(safe_result, indent=2, sort_keys=True))

    return "\n".join(lines)


__all__ = [
    "AgentResponse",
    "RouteResult",
    "RouteRule",
    "SUPPORTED_CATEGORIES",
    "format_agent_response",
    "format_route_response",
    "route_question",
    "run_agent",
    "supported_routes",
]