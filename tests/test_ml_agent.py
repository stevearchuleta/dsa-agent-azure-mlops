"""Tests for deterministic ML agent tools."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from dsa.agents.ml_agent import (
    evaluate_binary_classifier,
    prepare_binary_classification_data,
    save_model_metrics,
    train_logistic_regression,
    train_random_forest,
)


def make_binary_classification_dataframe() -> pd.DataFrame:
    """Create a deterministic binary-classification DataFrame."""
    return pd.DataFrame(
        {
            "signal": [
                0.10,
                0.20,
                0.30,
                0.40,
                0.50,
                0.60,
                0.70,
                0.80,
                0.90,
                1.00,
                2.10,
                2.20,
                2.30,
                2.40,
                2.50,
                2.60,
                2.70,
                2.80,
                2.90,
                3.00,
            ],
            "noise": [
                5.0,
                4.0,
                6.0,
                5.5,
                4.5,
                6.5,
                5.2,
                4.8,
                6.2,
                5.8,
                3.0,
                2.0,
                4.0,
                3.5,
                2.5,
                4.5,
                3.2,
                2.8,
                4.2,
                3.8,
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
            "target": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        }
    )


def assert_metric_range(metrics: dict[str, float | int | None], metric_name: str) -> None:
    """Assert that a model metric is between zero and one."""
    value = metrics[metric_name]
    assert value is not None
    assert 0.0 <= float(value) <= 1.0


def test_prepare_binary_classification_data_returns_stable_split_sizes():
    df = make_binary_classification_dataframe()

    data = prepare_binary_classification_data(
        df,
        target_col="target",
        feature_cols=["signal", "noise", "segment"],
        test_size=0.25,
        seed=42,
    )

    assert data.x_train.shape[0] == 15
    assert data.x_test.shape[0] == 5
    assert data.y_train.shape[0] == 15
    assert data.y_test.shape[0] == 5
    assert data.feature_columns == ["signal", "noise", "segment"]
    assert data.target_mapping == {0: 0, 1: 1}


def test_train_logistic_regression_evaluates_metrics_in_range():
    df = make_binary_classification_dataframe()
    data = prepare_binary_classification_data(df, target_col="target", seed=42)

    model = train_logistic_regression(data, seed=42)
    metrics = evaluate_binary_classifier(model, data)

    assert metrics["train_row_count"] == 15
    assert metrics["test_row_count"] == 5
    assert_metric_range(metrics, "accuracy")
    assert_metric_range(metrics, "precision")
    assert_metric_range(metrics, "recall")
    assert_metric_range(metrics, "f1")
    assert_metric_range(metrics, "roc_auc")


def test_train_random_forest_evaluates_metrics_in_range():
    df = make_binary_classification_dataframe()
    data = prepare_binary_classification_data(df, target_col="target", seed=42)

    model = train_random_forest(data, seed=42, n_estimators=20)
    metrics = evaluate_binary_classifier(model, data)

    assert metrics["train_row_count"] == 15
    assert metrics["test_row_count"] == 5
    assert_metric_range(metrics, "accuracy")
    assert_metric_range(metrics, "precision")
    assert_metric_range(metrics, "recall")
    assert_metric_range(metrics, "f1")
    assert_metric_range(metrics, "roc_auc")


def test_save_model_metrics_writes_json(tmp_path):
    metrics = {
        "accuracy": 0.9,
        "precision": 0.8,
        "recall": 0.7,
        "f1": 0.75,
        "roc_auc": 0.95,
    }

    output_path = tmp_path / "metrics.json"
    saved_path = save_model_metrics(metrics, output_path)

    assert saved_path == output_path
    assert saved_path.exists()

    saved_metrics = json.loads(saved_path.read_text(encoding="utf-8"))

    assert saved_metrics["accuracy"] == 0.9
    assert saved_metrics["roc_auc"] == 0.95


def test_prepare_binary_classification_data_rejects_multiclass_target():
    df = make_binary_classification_dataframe()
    df.loc[0, "target"] = 2

    with pytest.raises(ValueError, match="exactly two"):
        prepare_binary_classification_data(df, target_col="target")


def test_prepare_binary_classification_data_rejects_missing_feature():
    df = make_binary_classification_dataframe()

    with pytest.raises(KeyError, match="Column not found"):
        prepare_binary_classification_data(
            df,
            target_col="target",
            feature_cols=["missing_feature"],
        )


def test_prepare_binary_classification_data_rejects_non_dataframe():
    with pytest.raises(TypeError, match="Expected a pandas DataFrame"):
        prepare_binary_classification_data(  # type: ignore[arg-type]
            {"not": "a dataframe"},
            target_col="target",
        )