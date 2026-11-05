"""Explainability outputs for the best leak-free NEO classifier."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.model_selection import train_test_split

from src.config import (
    CLEAN_DATA_PATH,
    FIGURES_DIR,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    SEED,
    TABLES_DIR,
    TARGET,
)
from src.eda import load_data
from src.modeling import _pipeline
from xgboost import XGBClassifier


def _best_model():
    """Build the leak-free XGBoost estimator selected in Step 5."""
    return _pipeline(
        XGBClassifier(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=SEED,
            n_jobs=2,
        ),
        scale_numeric=False,
    )


def run_explainability(path: Path = CLEAN_DATA_PATH) -> pd.DataFrame:
    """Fit the best model and save permutation, SHAP, and PDP outputs."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data(path)
    X = df[MODEL_FEATURES]
    X = X.copy()
    X[NUMERIC_FEATURES] = X[NUMERIC_FEATURES].astype(float)
    y = df[TARGET].astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=SEED
    )
    model = _best_model()
    model.fit(x_train, y_train)

    permutation = permutation_importance(
        model,
        x_test,
        y_test,
        scoring="roc_auc",
        n_repeats=20,
        random_state=SEED,
        n_jobs=1,
    )
    importance = pd.DataFrame(
        {
            "feature": MODEL_FEATURES,
            "importance_mean": permutation.importances_mean,
            "importance_std": permutation.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    importance.to_csv(TABLES_DIR / "explainability_permutation_importance.csv", index=False)
    top_features = importance.head(3)["feature"].tolist()

    transformed_test = model.named_steps["preprocess"].transform(x_test)
    transformed_train = model.named_steps["preprocess"].transform(x_train)
    feature_names = model.named_steps["preprocess"].get_feature_names_out()
    transformed_test = transformed_test.toarray() if hasattr(transformed_test, "toarray") else transformed_test
    transformed_train = transformed_train.toarray() if hasattr(transformed_train, "toarray") else transformed_train
    tree_model = model.named_steps["model"]
    explainer = shap.TreeExplainer(tree_model)
    shap_values = explainer.shap_values(transformed_test)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    shap.summary_plot(
        shap_values,
        transformed_test,
        feature_names=feature_names,
        show=False,
        max_display=15,
    )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "explainability_shap_summary.png", dpi=300, bbox_inches="tight")
    plt.close()

    shap_importance = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    shap_importance.to_csv(TABLES_DIR / "explainability_shap_importance.csv", index=False)

    for feature in top_features:
        transformed_feature = [
            name for name in feature_names if name.endswith(f"__{feature}")
        ]
        if not transformed_feature:
            transformed_feature = [f"num__{feature}"]
        index = list(feature_names).index(transformed_feature[0])
        shap.dependence_plot(
            index,
            shap_values,
            transformed_test,
            feature_names=feature_names,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(
            FIGURES_DIR / f"explainability_shap_dependence_{feature}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    display = PartialDependenceDisplay.from_estimator(
        model,
        x_test,
        features=top_features,
        response_method="predict_proba",
        kind="average",
        n_jobs=1,
    )
    display.figure_.suptitle("Partial dependence for top three features")
    display.figure_.tight_layout()
    display.figure_.savefig(
        FIGURES_DIR / "explainability_partial_dependence_top3.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(display.figure_)

    interpretations = []
    for _, row in importance.head(3).iterrows():
        direction = (
            "higher values tend to increase discrimination toward hazard"
            if row["importance_mean"] > 0
            else "has weak or unstable marginal contribution"
        )
        interpretations.append(
            {
                "feature": row["feature"],
                "permutation_importance_mean": row["importance_mean"],
                "interpretation": (
                    f"{row['feature']} is among the three most influential raw features; "
                    f"{direction}. SHAP dependence should be read as association, not causation."
                ),
            }
        )
    interpretation_table = pd.DataFrame(interpretations)
    interpretation_table.to_csv(TABLES_DIR / "explainability_interpretations.csv", index=False)
    return importance


if __name__ == "__main__":
    print(run_explainability().head(10).to_string(index=False))
