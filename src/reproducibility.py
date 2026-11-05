"""Final reproducibility and leakage checks for the NEO project."""

from pathlib import Path

import pandas as pd

from src.config import (
    CLEAN_DATA_PATH,
    LEAKY_COLUMNS,
    MODEL_FEATURES,
    OUTPUTS_DIR,
    TABLES_DIR,
)


def run_reproducibility_checks() -> pd.DataFrame:
    """Verify required inputs, outputs, and absence of leaky model features."""
    checks = []
    checks.append(
        {
            "check": "clean_dataset_exists",
            "passed": CLEAN_DATA_PATH.is_file(),
            "details": str(CLEAN_DATA_PATH),
        }
    )
    checks.append(
        {
            "check": "model_features_exclude_all_leaky_columns",
            "passed": not set(MODEL_FEATURES).intersection(LEAKY_COLUMNS),
            "details": ", ".join(sorted(set(MODEL_FEATURES).intersection(LEAKY_COLUMNS)))
            or "none",
        }
    )
    required_outputs = [
        OUTPUTS_DIR / "tables" / "modeling_feature_audit.csv",
        OUTPUTS_DIR / "tables" / "leakage_comparison.csv",
        OUTPUTS_DIR / "figures" / "modeling_roc_pr_curves.png",
        OUTPUTS_DIR / "figures" / "explainability_shap_summary.png",
    ]
    for output in required_outputs:
        checks.append(
            {
                "check": f"required_output_{output.name}",
                "passed": output.is_file(),
                "details": str(output),
            }
        )
    audit_path = TABLES_DIR / "modeling_feature_audit.csv"
    if audit_path.is_file():
        audit = pd.read_csv(audit_path)
        audit_features = set(
            str(audit.iloc[0]["model_features"]).split(", ")
        )
        checks.append(
            {
                "check": "saved_model_audit_excludes_all_leaky_columns",
                "passed": not audit_features.intersection(LEAKY_COLUMNS),
                "details": ", ".join(sorted(audit_features.intersection(LEAKY_COLUMNS)))
                or "none",
            }
        )
    result = pd.DataFrame(checks)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(TABLES_DIR / "reproducibility_checklist.csv", index=False)
    if not result["passed"].all():
        failed = result.loc[~result["passed"], "check"].tolist()
        raise RuntimeError(f"Reproducibility checks failed: {failed}")
    return result


if __name__ == "__main__":
    print(run_reproducibility_checks().to_string(index=False))
