"""Leakage-conscious training and evaluation for both quotation regressors."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .features import (
    CATEGORICAL_ENHANCED, CATEGORICAL_SIMPLE, NUMERIC_ENHANCED,
    NUMERIC_SIMPLE, build_quotation_table, load_orders,
)


def build_model(numeric: list[str], categorical: list[str], *, enhanced: bool) -> Pipeline:
    transformer = ColumnTransformer([
        ("num", "passthrough", numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    regressor = (GradientBoostingRegressor(n_estimators=200, learning_rate=0.08, max_depth=5, random_state=42)
                 if enhanced else GradientBoostingRegressor(n_estimators=100, random_state=42))
    return Pipeline([("features", transformer), ("regressor", regressor)])


def metrics(y_true, y_pred) -> dict:
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "r2": round(float(r2_score(y_true, y_pred)), 6),
    }


def train_quotes(db_path: str | Path, output_dir: str | Path = "artifacts") -> dict:
    features, counts = build_quotation_table(load_orders(db_path))
    if len(features) < 12:
        raise ValueError(f"Need at least 12 valid orders; found {len(features)}. Add more representative records.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_idx, test_idx = train_test_split(np.arange(len(features)), test_size=0.20, random_state=42)
    y_train = features.iloc[train_idx]["total_amount"]
    y_test = features.iloc[test_idx]["total_amount"]
    baseline = DummyRegressor(strategy="median").fit(np.zeros((len(y_train), 1)), y_train)
    results = {
        "dataset": counts,
        "evaluation": {"split": "80/20 fixed-seed random holdout", "seed": 42,
                       "train_records": len(train_idx), "test_records": len(test_idx),
                       "baseline_median": metrics(y_test, baseline.predict(np.zeros((len(y_test), 1))))},
        "models": {},
        "limitations": ["Historical orders may not reflect future prices", "Random holdout is not a forward-in-time validation"],
    }
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, nums, cats in [
        ("simplified", NUMERIC_SIMPLE, CATEGORICAL_SIMPLE),
        ("enhanced", NUMERIC_ENHANCED, CATEGORICAL_ENHANCED),
    ]:
        X = features[nums + cats]
        model = build_model(nums, cats, enhanced=name == "enhanced")
        model.fit(X.iloc[train_idx], y_train)
        pred = np.maximum(0, model.predict(X.iloc[test_idx]))
        results["models"][name] = metrics(y_test, pred)
        # Evaluate on the holdout BEFORE training the deployed model on all observations.
        if name == "enhanced":
            ax.scatter(y_test, pred, alpha=0.65, label="Enhanced model")
            errors = pd.DataFrame({"actual": y_test.to_numpy(), "predicted": pred,
                                   "absolute_error": np.abs(y_test.to_numpy() - pred)},
                                  index=features.index[test_idx])
            errors.to_csv(output_dir / "enhanced_holdout_predictions.csv", index=False)
        model.fit(X, features["total_amount"])
        joblib.dump(model, output_dir / f"{name}_quote_model.joblib")

    lower = 0
    upper = float(max(y_test.max(), errors["predicted"].max())) * 1.02
    ax.plot([lower, upper], [lower, upper], linestyle="--", color="#ef4444", label="Perfect prediction")
    ax.set(xlabel="Actual quotation amount", ylabel="Predicted amount", title="Enhanced quotation model: held-out test set")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "quotation_actual_vs_predicted.png", dpi=170)
    plt.close(fig)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return results


def predict_quote(spec: dict, output_dir: str | Path = "artifacts", mode: str = "enhanced") -> float:
    """Only load models you trained yourself; joblib is not safe for untrusted files."""
    from .features import features_from_form
    if mode not in {"simplified", "enhanced"}:
        raise ValueError("mode must be 'simplified' or 'enhanced'")
    path = Path(output_dir) / f"{mode}_quote_model.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}. Run the training command first.")
    pipeline = joblib.load(path)
    features = features_from_form(spec)
    return float(max(0, pipeline.predict(pd.DataFrame([features]))[0]))
