"""Compare model sensitivity to feature normalization on a tabular dataset.

The script loads a user supplied dataset, splits it into training and test
subsets (80/20), and evaluates several machine learning models with and
without numeric feature normalization.  Cross validation is executed on the
training data and the model whose score changes the most when normalization is
applied is reported.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import get_scorer
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def _safe_import(name: str, from_: Optional[str] = None):
    """Attempt to import ``name`` from ``from_`` returning ``None`` on failure."""

    try:
        if from_:
            module = __import__(from_, fromlist=[name])
            return getattr(module, name)
        return __import__(name)
    except ImportError:
        return None


LGBMClassifier = _safe_import("LGBMClassifier", "lightgbm.sklearn")
LGBMRegressor = _safe_import("LGBMRegressor", "lightgbm.sklearn")
XGBClassifier = _safe_import("XGBClassifier", "xgboost.sklearn")
XGBRegressor = _safe_import("XGBRegressor", "xgboost.sklearn")


@dataclass
class ModelComparison:
    """Holds evaluation metrics for a single model."""

    name: str
    cv_mean_raw: float
    cv_std_raw: float
    cv_mean_norm: float
    cv_std_norm: float
    delta: float
    abs_delta: float
    test_score_raw: float
    test_score_norm: float


def load_dataset(path: str, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Load a dataset from ``path`` and separate features and target column."""

    df = pd.read_csv(path)
    if target not in df.columns:
        available = ", ".join(df.columns)
        raise ValueError(f"Target column '{target}' not found. Available: {available}")

    y = df[target]
    X = df.drop(columns=[target])
    return X, y


def build_preprocessors(X: pd.DataFrame) -> Tuple[ColumnTransformer, ColumnTransformer]:
    """Create preprocessors for raw and normalized numeric features."""

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_imputer = SimpleImputer(strategy="median")
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    raw_numeric_transformer = Pipeline(steps=[("imputer", numeric_imputer)])
    normalized_numeric_transformer = Pipeline(
        steps=[
            ("imputer", numeric_imputer),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor_raw = ColumnTransformer(
        transformers=[
            ("num", raw_numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    preprocessor_norm = ColumnTransformer(
        transformers=[
            ("num", normalized_numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    return preprocessor_raw, preprocessor_norm


def build_models(task: str) -> Dict[str, BaseEstimator]:
    """Create a dictionary of candidate models for the requested task."""

    task = task.lower()
    if task not in {"classification", "regression"}:
        raise ValueError("Task must be either 'classification' or 'regression'.")

    models: Dict[str, BaseEstimator]
    if task == "classification":
        models = {
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "RandomForestClassifier": RandomForestClassifier(random_state=42),
        }
        if XGBClassifier is not None:
            models["XGBoostClassifier"] = XGBClassifier(
                eval_metric="logloss", use_label_encoder=False, random_state=42
            )
        if LGBMClassifier is not None:
            models["LightGBMClassifier"] = LGBMClassifier(random_state=42)
    else:
        models = {
            "LinearRegression": LinearRegression(),
            "RandomForestRegressor": RandomForestRegressor(random_state=42),
        }
        if XGBRegressor is not None:
            models["XGBoostRegressor"] = XGBRegressor(random_state=42)
        if LGBMRegressor is not None:
            models["LightGBMRegressor"] = LGBMRegressor(random_state=42)

    return models


def choose_cv(task: str, n_splits: int):
    if task == "classification":
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    return KFold(n_splits=n_splits, shuffle=True, random_state=42)


def evaluate_models(
    X: pd.DataFrame,
    y: pd.Series,
    task: str,
    n_splits: int,
) -> Tuple[List[ModelComparison], pd.DataFrame]:
    """Run cross validation with and without normalization for all models."""

    preprocessor_raw, preprocessor_norm = build_preprocessors(X)
    models = build_models(task)

    scorer = get_scorer("accuracy" if task == "classification" else "r2")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if task == "classification" else None
    )

    cv = choose_cv(task, n_splits)
    cv_splits = list(cv.split(X_train, y_train))

    rows: List[ModelComparison] = []
    score_rows: List[Dict[str, float]] = []

    for name, estimator in models.items():
        pipeline_raw = Pipeline(
            steps=[("preprocessor", preprocessor_raw), ("model", clone(estimator))]
        )
        pipeline_norm = Pipeline(
            steps=[("preprocessor", preprocessor_norm), ("model", clone(estimator))]
        )

        cv_raw = cross_validate(
            pipeline_raw,
            X_train,
            y_train,
            cv=cv_splits,
            scoring=scorer,
            n_jobs=-1,
            return_train_score=False,
        )
        cv_norm = cross_validate(
            pipeline_norm,
            X_train,
            y_train,
            cv=cv_splits,
            scoring=scorer,
            n_jobs=-1,
            return_train_score=False,
        )

        mean_raw = float(np.mean(cv_raw["test_score"]))
        std_raw = float(np.std(cv_raw["test_score"]))
        mean_norm = float(np.mean(cv_norm["test_score"]))
        std_norm = float(np.std(cv_norm["test_score"]))
        delta = mean_norm - mean_raw
        abs_delta = abs(delta)

        test_pipeline_raw = Pipeline(
            steps=[("preprocessor", preprocessor_raw), ("model", clone(estimator))]
        )
        test_pipeline_norm = Pipeline(
            steps=[("preprocessor", preprocessor_norm), ("model", clone(estimator))]
        )

        test_pipeline_raw.fit(X_train, y_train)
        test_pipeline_norm.fit(X_train, y_train)
        test_raw = float(test_pipeline_raw.score(X_test, y_test))
        test_norm = float(test_pipeline_norm.score(X_test, y_test))

        rows.append(
            ModelComparison(
                name=name,
                cv_mean_raw=mean_raw,
                cv_std_raw=std_raw,
                cv_mean_norm=mean_norm,
                cv_std_norm=std_norm,
                delta=delta,
                abs_delta=abs_delta,
                test_score_raw=test_raw,
                test_score_norm=test_norm,
            )
        )

        score_rows.append(
            {
                "model": name,
                "cv_mean_raw": mean_raw,
                "cv_std_raw": std_raw,
                "cv_mean_norm": mean_norm,
                "cv_std_norm": std_norm,
                "delta": delta,
                "abs_delta": abs_delta,
                "test_score_raw": test_raw,
                "test_score_norm": test_norm,
            }
        )

    table = pd.DataFrame(score_rows).sort_values(by="abs_delta", ascending=False)
    return rows, table


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to the CSV dataset")
    parser.add_argument("--target", required=True, help="Name of the target column")
    parser.add_argument(
        "--task",
        default="classification",
        choices=["classification", "regression"],
        help="Type of machine learning task",
    )
    parser.add_argument(
        "--folds", type=int, default=5, help="Number of cross validation folds"
    )
    parser.add_argument(
        "--output",
        help="Optional path to write the summary table as a CSV",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    X, y = load_dataset(args.data, args.target)
    comparisons, table = evaluate_models(X, y, args.task, args.folds)

    most_sensitive = max(comparisons, key=lambda row: row.abs_delta)

    print("模型在是否归一化条件下的交叉验证表现对比 (按差异排序):")
    print(table.to_string(index=False))
    print()
    print(
        f"对归一化最敏感的模型: {most_sensitive.name} (归一化后得分变化 {most_sensitive.delta:.4f})"
    )

    if args.output:
        table.to_csv(args.output, index=False)
        print(f"结果已写入 {args.output}")

    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    sys.exit(main())

