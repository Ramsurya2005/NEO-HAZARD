"""Leak-free predictive modelling workflow for NEO hazard classification."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.stats import randint, uniform, wilcoxon
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

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


SCORING = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "balanced_accuracy": "balanced_accuracy",
    "mcc": "matthews_corrcoef",
    "brier": "neg_brier_score",
}


def _preprocessor(scale_numeric: bool = True) -> ColumnTransformer:
    numeric = Pipeline(
        [("scale", StandardScaler())] if scale_numeric else [("identity", "passthrough")]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), ["orbit_class"]),
        ]
    )


def _pipeline(model, smote: bool = False, scale_numeric: bool = True):
    steps = [("preprocess", _preprocessor(scale_numeric))]
    if smote:
        steps.append(("smote", SMOTE(random_state=SEED)))
    steps.append(("model", model))
    return ImbPipeline(steps)


def _models() -> dict:
    return {
        "DummyClassifier": _pipeline(
            DummyClassifier(strategy="prior"), scale_numeric=False
        ),
        "LogisticRegression_weighted": _pipeline(
            LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=SEED
            )
        ),
        "LogisticRegression_SMOTE": _pipeline(
            LogisticRegression(max_iter=2000, random_state=SEED), smote=True
        ),
        "DecisionTree_weighted": _pipeline(
            __import__("sklearn.tree", fromlist=["DecisionTreeClassifier"]).DecisionTreeClassifier(
                class_weight="balanced", random_state=SEED, max_depth=5
            ),
            scale_numeric=False,
        ),
        "RandomForest_weighted": _pipeline(
            RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=SEED,
                n_jobs=-1,
                max_features="sqrt",
            ),
            scale_numeric=False,
        ),
        "RandomForest_SMOTE": _pipeline(
            RandomForestClassifier(
                n_estimators=300, random_state=SEED, n_jobs=-1, max_features="sqrt"
            ),
            smote=True,
            scale_numeric=False,
        ),
        "XGBoost": _pipeline(
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
        ),
        "GradientBoosting": _pipeline(
            GradientBoostingClassifier(random_state=SEED, n_estimators=150),
            scale_numeric=False,
        ),
    }


def _metric_frame(cv_results: dict, model_name: str) -> pd.DataFrame:
    rows = []
    for metric in SCORING:
        values = cv_results[f"test_{metric}"]
        if metric == "brier":
            values = -values
        rows.append(
            {
                "model": model_name,
                "metric": metric,
                "mean": np.mean(values),
                "std": np.std(values, ddof=1),
            }
        )
    return pd.DataFrame(rows)


def _evaluate_models(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, dict]:
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    all_results = []
    fold_scores = {}
    for name, model in _models().items():
        results = cross_validate(
            model,
            X,
            y,
            scoring=SCORING,
            cv=cv,
            n_jobs=1,
            return_train_score=False,
        )
        all_results.append(_metric_frame(results, name))
        fold_scores[name] = results["test_roc_auc"]
    return pd.concat(all_results, ignore_index=True), fold_scores


def _tune_models(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    x_tune, _, y_tune, _ = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    searches = {
        "RandomForest_tuned": (
            _pipeline(
                RandomForestClassifier(
                    class_weight="balanced", random_state=SEED, n_jobs=-1
                ),
                scale_numeric=False,
            ),
            {
                "model__n_estimators": randint(150, 400),
                "model__max_depth": randint(3, 12),
                "model__min_samples_leaf": randint(1, 8),
                "model__max_features": ["sqrt", "log2", None],
            },
        ),
        "XGBoost_tuned": (
            _pipeline(
                XGBClassifier(
                    eval_metric="logloss", random_state=SEED, n_jobs=2
                ),
                scale_numeric=False,
            ),
            {
                "model__n_estimators": randint(150, 400),
                "model__max_depth": randint(2, 8),
                "model__learning_rate": uniform(0.02, 0.18),
                "model__subsample": uniform(0.7, 0.3),
                "model__colsample_bytree": uniform(0.7, 0.3),
            },
        ),
    }
    rows = []
    for name, (estimator, params) in searches.items():
        search = RandomizedSearchCV(
            estimator,
            params,
            n_iter=8,
            scoring="roc_auc",
            cv=StratifiedKFold(3, shuffle=True, random_state=SEED),
            random_state=SEED,
            n_jobs=1,
        )
        search.fit(x_tune, y_tune)
        rows.append(
            {
                "model": name,
                "best_cv_roc_auc": search.best_score_,
                "best_params": str(search.best_params_),
                "tuning_split": "80% training split; 20% held out before tuning",
            }
        )
    return pd.DataFrame(rows)


def _fit_best_and_plots(X: pd.DataFrame, y: pd.Series, comparison: pd.DataFrame) -> str:
    roc_summary = comparison[comparison["metric"] == "roc_auc"].sort_values(
        "mean", ascending=False
    )
    best_name = roc_summary.iloc[0]["model"]
    model = _models()[best_name]
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=SEED
    )
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    fpr, tpr, _ = roc_curve(y_test, probabilities)
    precision, recall, _ = precision_recall_curve(y_test, probabilities)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f"{best_name} (AUC={roc_auc_score(y_test, probabilities):.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, label=f"{best_name} (AP={average_precision_score(y_test, probabilities):.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-recall curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "modeling_roc_pr_curves.png", dpi=300)
    plt.close()

    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.xticks([0, 1], ["Non-hazardous", "Hazardous"])
    plt.yticks([0, 1], ["Non-hazardous", "Hazardous"])
    plt.xlabel("Predicted")
    plt.ylabel("Observed")
    plt.title(f"Confusion matrix: {best_name}")
    for row in range(2):
        for column in range(2):
            plt.text(column, row, cm[row, column], ha="center", va="center")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "modeling_best_confusion_matrix.png", dpi=300)
    plt.close()

    frac_pos, mean_pred = calibration_curve(y_test, probabilities, n_bins=10)
    plt.figure(figsize=(6, 6))
    plt.plot(mean_pred, frac_pos, "o-", label=best_name)
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed hazard rate")
    plt.title("Best-model calibration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "modeling_best_calibration.png", dpi=300)
    plt.close()

    thresholds = np.linspace(0.05, 0.95, 19)
    threshold_rows = []
    for threshold in thresholds:
        threshold_predictions = (probabilities >= threshold).astype(int)
        threshold_rows.append(
            {
                "threshold": threshold,
                "precision": precision_score(y_test, threshold_predictions, zero_division=0),
                "recall": recall_score(y_test, threshold_predictions, zero_division=0),
                "f1": f1_score(y_test, threshold_predictions, zero_division=0),
                "balanced_accuracy": balanced_accuracy_score(y_test, threshold_predictions),
                "mcc": matthews_corrcoef(y_test, threshold_predictions),
            }
        )
    threshold_table = pd.DataFrame(threshold_rows)
    threshold_table.to_csv(TABLES_DIR / "modeling_threshold_analysis.csv", index=False)
    plt.figure(figsize=(8, 5))
    for metric in ["precision", "recall", "f1", "balanced_accuracy"]:
        plt.plot(threshold_table["threshold"], threshold_table[metric], label=metric)
    plt.xlabel("Decision threshold")
    plt.ylabel("Metric")
    plt.title("Threshold tuning for best model")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "modeling_threshold_tuning.png", dpi=300)
    plt.close()

    point_metrics = {
        "model": best_name,
        "roc_auc": roc_auc_score(y_test, probabilities),
        "pr_auc": average_precision_score(y_test, probabilities),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
        "mcc": matthews_corrcoef(y_test, predictions),
        "brier": brier_score_loss(y_test, probabilities),
    }
    pd.DataFrame([point_metrics]).to_csv(TABLES_DIR / "modeling_best_holdout_metrics.csv", index=False)
    return best_name


def run_modeling(path: Path = CLEAN_DATA_PATH) -> str:
    """Run repeated CV, tuning, and diagnostic plots for Step 5."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data(path)
    X = df[MODEL_FEATURES]
    y = df[TARGET].astype(int)
    comparison, fold_scores = _evaluate_models(X, y)
    comparison.to_csv(TABLES_DIR / "modeling_comparison.csv", index=False)
    tuning = _tune_models(X, y)
    tuning.to_csv(TABLES_DIR / "modeling_tuning.csv", index=False)
    best_name = _fit_best_and_plots(X, y, comparison)
    baseline = fold_scores["DummyClassifier"]
    best_scores = fold_scores[best_name]
    paired = wilcoxon(best_scores, baseline, alternative="greater")
    pd.DataFrame(
        [
            {
                "best_model": best_name,
                "baseline": "DummyClassifier",
                "test": "Wilcoxon signed-rank on repeated-CV ROC-AUC",
                "statistic": paired.statistic,
                "p": paired.pvalue,
                "n_paired_folds": len(best_scores),
            }
        ]
    ).to_csv(TABLES_DIR / "modeling_best_vs_baseline_test.csv", index=False)
    pd.DataFrame(
        [
            {
                "best_model": best_name,
                "model_features": ", ".join(MODEL_FEATURES),
                "leaky_columns_used": False,
                "cv": "RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)",
            }
        ]
    ).to_csv(TABLES_DIR / "modeling_feature_audit.csv", index=False)
    return best_name


if __name__ == "__main__":
    print(f"Best model: {run_modeling()}")
