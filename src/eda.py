"""Exploratory data analysis for the leak-free NEO dataset."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import (
    CLEAN_DATA_PATH,
    FIGURES_DIR,
    NUMERIC_FEATURES,
    OUTPUTS_DIR,
    TABLES_DIR,
)


def load_data(path: Path = CLEAN_DATA_PATH) -> pd.DataFrame:
    """Load the clean, one-row-per-asteroid dataset and merge Atira into Aten."""
    df = pd.read_csv(path)
    df["orbit_class"] = df["orbit_class"].replace({"Atira": "Aten+Atira"})
    return df


def _ensure_output_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def _save_audit_tables(df: pd.DataFrame) -> None:
    audit = pd.DataFrame(
        {
            "metric": ["rows", "columns", "duplicate_rows", "duplicate_neo_ids"],
            "value": [
                len(df),
                len(df.columns),
                int(df.duplicated().sum()),
                int(df["neo_id"].duplicated().sum()),
            ],
        }
    )
    audit.to_csv(TABLES_DIR / "eda_audit.csv", index=False)

    missing = df.isna().sum().rename("missing_count").to_frame()
    missing["missing_percent"] = missing["missing_count"] / len(df) * 100
    missing.to_csv(TABLES_DIR / "eda_missing_values.csv")

    dtypes = df.dtypes.astype(str).rename("dtype").to_frame()
    dtypes.to_csv(TABLES_DIR / "eda_dtypes.csv")

    df.groupby("hazardous")[NUMERIC_FEATURES].describe().T.to_csv(
        TABLES_DIR / "eda_describe_by_hazard.csv"
    )
    df["hazardous"].value_counts(dropna=False).rename("count").to_frame().to_csv(
        TABLES_DIR / "eda_hazard_balance.csv"
    )
    df["orbit_class"].value_counts(dropna=False).rename("count").to_frame().to_csv(
        TABLES_DIR / "eda_orbit_class_counts.csv"
    )

    numeric = df[NUMERIC_FEATURES]
    numeric.corr(method="pearson").to_csv(TABLES_DIR / "eda_correlations.csv")
    skewness = numeric.skew().sort_values(ascending=False).rename("skewness").to_frame()
    skewness["flag_skewed_abs_gt_1"] = skewness["skewness"].abs() > 1
    skewness.to_csv(TABLES_DIR / "eda_skewness.csv")

    # One-hot encode the merged orbit class for a complete feature VIF table.
    vif_input = pd.concat(
        [numeric, pd.get_dummies(df["orbit_class"], prefix="orbit", drop_first=True, dtype=float)],
        axis=1,
    ).astype(float)
    vif_input.insert(0, "const", 1.0)
    vif_rows = []
    for index, column in enumerate(vif_input.columns):
        if column == "const":
            continue
        y = vif_input[column].to_numpy()
        predictors = vif_input.drop(columns=column).to_numpy()
        coefficients, *_ = np.linalg.lstsq(predictors, y, rcond=None)
        residuals = y - predictors @ coefficients
        total_ss = np.sum((y - y.mean()) ** 2)
        r_squared = 1 - np.sum(residuals**2) / total_ss if total_ss else 0.0
        vif_value = np.inf if r_squared >= 1 - 1e-12 else 1 / (1 - r_squared)
        vif_rows.append(
            {
                "feature": column,
                "vif": vif_value,
            }
        )
    vif = pd.DataFrame(vif_rows).sort_values("vif", ascending=False)
    vif["flag_high_vif_gt_5"] = vif["vif"] > 5
    vif.to_csv(TABLES_DIR / "eda_vif.csv", index=False)


def _save_figures(df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    long_df = df.melt(
        id_vars="hazardous",
        value_vars=NUMERIC_FEATURES,
        var_name="feature",
        value_name="value",
    )
    g = sns.FacetGrid(
        long_df,
        col="feature",
        col_wrap=4,
        hue="hazardous",
        sharex=False,
        sharey=False,
        height=2.7,
    )
    g.map_dataframe(sns.histplot, x="value", stat="density", common_norm=False, bins=25)
    g.add_legend(title="Hazardous")
    g.set_titles("{col_name}")
    g.savefig(FIGURES_DIR / "eda_histograms.png", dpi=300, bbox_inches="tight")
    plt.close(g.figure)

    plt.figure(figsize=(14, 8))
    sns.boxplot(data=long_df, x="feature", y="value", hue="hazardous", showfliers=False)
    plt.xticks(rotation=55, ha="right")
    plt.title("Numeric features by hazard status")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "eda_boxplots_by_hazard.png", dpi=300)
    plt.close()

    plt.figure(figsize=(11, 9))
    sns.heatmap(df[NUMERIC_FEATURES].corr(), cmap="vlag", center=0, square=True)
    plt.title("Pearson correlation heatmap")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "eda_correlation_heatmap.png", dpi=300)
    plt.close()


def run_eda() -> pd.DataFrame:
    """Run Step 2 EDA and save all requested tables and figures."""
    _ensure_output_dirs()
    df = load_data()
    _save_audit_tables(df)
    _save_figures(df)
    return df


if __name__ == "__main__":
    result = run_eda()
    print(f"Generated EDA outputs for {len(result):,} asteroids.")
