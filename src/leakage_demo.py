"""Demonstrate the invalid performance gain from label-leaking features."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate

from src.config import (
    CLEAN_DATA_PATH,
    FIGURES_DIR,
    LEAKY_COLUMNS,
    LEAKY_DATA_PATH,
    MODEL_FEATURES,
    SEED,
    TABLES_DIR,
    TARGET,
)
from src.eda import load_data
from src.modeling import SCORING, _pipeline
from xgboost import XGBClassifier


def _xgboost_pipeline(feature_columns: list[str]):
    """Build the same XGBoost estimator used for the leak-free benchmark."""
    numeric = [column for column in feature_columns if column != "orbit_class"]
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder

    preprocessor = ColumnTransformer(
        [
            ("numeric", "passthrough", numeric),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), ["orbit_class"]),
        ]
    )
    model = XGBClassifier(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=SEED,
        n_jobs=2,
    )
    from imblearn.pipeline import Pipeline

    return Pipeline([("preprocess", preprocessor), ("model", model)])


def _evaluate(df: pd.DataFrame, feature_columns: list[str], label: str) -> pd.DataFrame:
    X = df[feature_columns]
    y = df[TARGET].astype(int)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    results = cross_validate(
        _xgboost_pipeline(feature_columns),
        X,
        y,
        scoring=SCORING,
        cv=cv,
        n_jobs=1,
    )
    rows = []
    for metric in SCORING:
        values = results[f"test_{metric}"]
        if metric == "brier":
            values = -values
        rows.append(
            {
                "analysis": label,
                "feature_set": "leak-free" if label == "Leak-free XGBoost" else "leaky",
                "metric": metric,
                "mean": values.mean(),
                "std": values.std(ddof=1),
                "leaky_columns_used": label == "Leaky XGBoost",
            }
        )
    return pd.DataFrame(rows)


def run_leakage_demo() -> pd.DataFrame:
    """Compare leak-free and invalid leaky XGBoost models under identical CV."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    clean = load_data(CLEAN_DATA_PATH)
    leaky = load_data(LEAKY_DATA_PATH)
    leaky_features = MODEL_FEATURES + ["moid", "abs_magnitude"]
    clean_results = _evaluate(clean, MODEL_FEATURES, "Leak-free XGBoost")
    leaky_results = _evaluate(leaky, leaky_features, "Leaky XGBoost")
    comparison = pd.concat([clean_results, leaky_results], ignore_index=True)
    comparison["interpretation"] = comparison["leaky_columns_used"].map(
        {
            True: "Invalid demonstration: MOID and absolute magnitude encode the label definition",
            False: "Valid benchmark: only leak-free orbital and approach features",
        }
    )
    comparison.to_csv(TABLES_DIR / "leakage_comparison.csv", index=False)

    roc = comparison[comparison["metric"].isin(["roc_auc", "pr_auc"])]
    pivot = roc.pivot(index="metric", columns="analysis", values="mean")
    ax = pivot.plot(kind="bar", figsize=(8, 5), ylim=(0, 1), rot=0)
    ax.set_ylabel("Repeated-CV mean score")
    ax.set_title("Invalid leakage inflates apparent model performance")
    ax.legend(title="")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "leakage_comparison.png", dpi=300)
    plt.close()
    return comparison


if __name__ == "__main__":
    result = run_leakage_demo()
    print(result[result["metric"].isin(["roc_auc", "pr_auc"])].to_string(index=False))
