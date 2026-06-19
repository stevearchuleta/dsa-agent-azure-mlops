"""
Deterministic machine learning tools for pandas DataFrames.

These functions provide safe, repeatable binary-classification workflows using
scikit-learn. The module avoids arbitrary LLM-generated modeling code and is
intended for local tests, Azure ML jobs, and future agent tool routing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class BinaryClassificationData:
    """Train/test data returned by the deterministic data-prep tool."""

    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    target_column: str
    feature_columns: list[str]
    target_mapping: dict[Any, int]
    seed: int


def _validate_dataframe(df: pd.DataFrame) -> None:
    """Raise a helpful error when the supplied object is not a DataFrame."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")


def _validate_column(df: pd.DataFrame, column: str) -> None:
    """Raise a helpful error when a required column is missing."""
    if column not in df.columns:
        raise KeyError(f"Column not found: {column}")


def _validate_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """Validate a list of DataFrame columns."""
    for column in columns:
        _validate_column(df, column)


def _validate_binary_target(target: pd.Series) -> dict[Any, int]:
    """Return a deterministic binary target mapping."""
    unique_values = target.dropna().unique().tolist()
    sorted_values = sorted(unique_values, key=lambda value: str(value))

    if len(sorted_values) != 2:
        raise ValueError("Target column must contain exactly two non-missing classes.")

    return {
        sorted_values[0]: 0,
        sorted_values[1]: 1,
    }


def _build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    """Build a deterministic preprocessing transformer."""
    numeric_columns = x.select_dtypes(include="number").columns.to_list()
    categorical_columns = x.select_dtypes(exclude="number").columns.to_list()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )


def prepare_binary_classification_data(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str] | None = None,
    *,
    test_size: float = 0.25,
    seed: int = 42,
    stratify: bool = True,
) -> BinaryClassificationData:
    """Prepare deterministic train/test splits for binary classification."""
    _validate_dataframe(df)
    _validate_column(df, target_col)

    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be greater than 0 and less than 1.")

    selected_feature_columns = (
        list(feature_cols)
        if feature_cols is not None
        else [str(column) for column in df.columns if column != target_col]
    )

    if not selected_feature_columns:
        raise ValueError("At least one feature column is required.")

    _validate_columns(df, selected_feature_columns)

    working_df = df[selected_feature_columns + [target_col]].dropna(
        subset=[target_col]
    )

    target_mapping = _validate_binary_target(working_df[target_col])
    y = working_df[target_col].map(target_mapping).astype(int)
    x = working_df[selected_feature_columns].copy()

    class_counts = y.value_counts()
    stratify_values = y if stratify and class_counts.min() >= 2 else None

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=stratify_values,
    )

    return BinaryClassificationData(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        target_column=target_col,
        feature_columns=selected_feature_columns,
        target_mapping=target_mapping,
        seed=seed,
    )


def train_logistic_regression(
    data: BinaryClassificationData,
    *,
    seed: int = 42,
    max_iter: int = 1000,
) -> Pipeline:
    """Train a deterministic logistic-regression classifier."""
    model = Pipeline(
        steps=[
            ("preprocessor", _build_preprocessor(data.x_train)),
            (
                "classifier",
                LogisticRegression(
                    max_iter=max_iter,
                    random_state=seed,
                ),
            ),
        ]
    )

    model.fit(data.x_train, data.y_train)
    return model


def train_random_forest(
    data: BinaryClassificationData,
    *,
    seed: int = 42,
    n_estimators: int = 100,
    max_depth: int | None = None,
) -> Pipeline:
    """Train a deterministic random-forest classifier."""
    model = Pipeline(
        steps=[
            ("preprocessor", _build_preprocessor(data.x_train)),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=seed,
                    n_jobs=1,
                ),
            ),
        ]
    )

    model.fit(data.x_train, data.y_train)
    return model


def evaluate_binary_classifier(
    model: BaseEstimator,
    data: BinaryClassificationData,
) -> dict[str, float | int | None]:
    """Evaluate a binary classifier with deterministic metric names."""
    y_pred = model.predict(data.x_test)

    roc_auc: float | None = None

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(data.x_test)[:, 1]

        if data.y_test.nunique() == 2:
            roc_auc = float(roc_auc_score(data.y_test, y_score))

    return {
        "accuracy": float(accuracy_score(data.y_test, y_pred)),
        "precision": float(precision_score(data.y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(data.y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(data.y_test, y_pred, zero_division=0)),
        "roc_auc": roc_auc,
        "train_row_count": int(data.x_train.shape[0]),
        "test_row_count": int(data.x_test.shape[0]),
    }


def _json_safe_value(value: Any) -> Any:
    """Convert NumPy scalar values to JSON-safe Python values."""
    if isinstance(value, np.generic):
        return value.item()

    return value


def save_model_metrics(
    metrics: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """Save model metrics as deterministic JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    safe_metrics = {
        str(key): _json_safe_value(value)
        for key, value in metrics.items()
    }

    path.write_text(
        json.dumps(safe_metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return path