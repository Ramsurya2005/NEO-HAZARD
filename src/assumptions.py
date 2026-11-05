"""Distribution and variance checks for the NEO inference workflow."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

from src.config import CLEAN_DATA_PATH, FIGURES_DIR, NUMERIC_FEATURES, TABLES_DIR, SEED
from src.eda import load_data


def _subsample(values: pd.Series, max_n: int = 5000) -> pd.Series:
    """Return a reproducible finite subsample suitable for normality testing."""
    values = values.dropna()
    if len(values) <= max_n:
        return values
    return values.sample(max_n, random_state=SEED)


def run_assumption_checks(path: Path = CLEAN_DATA_PATH) -> pd.DataFrame:
    """Run Shapiro-Wilk normality and Levene variance tests by hazard status."""
    df = load_data(path)
    rows = []
    for feature in NUMERIC_FEATURES:
        groups = [
            _subsample(df.loc[df["hazardous"] == status, feature])
            for status in [False, True]
        ]
        shapiro_results = [stats.shapiro(group) for group in groups]
        levene_result = stats.levene(*groups, center="median")
        both_normal = all(result.pvalue >= 0.05 for result in shapiro_results)
        recommendation = (
            "Welch t-test"
            if both_normal and levene_result.pvalue >= 0.05
            else "Mann-Whitney U"
        )
        rows.append(
            {
                "feature": feature,
                "non_hazardous_shapiro_stat": shapiro_results[0].statistic,
                "non_hazardous_shapiro_p": shapiro_results[0].pvalue,
                "hazardous_shapiro_stat": shapiro_results[1].statistic,
                "hazardous_shapiro_p": shapiro_results[1].pvalue,
                "levene_stat": levene_result.statistic,
                "levene_p": levene_result.pvalue,
                "both_groups_approximately_normal": both_normal,
                "equal_variance_at_alpha_0.05": levene_result.pvalue >= 0.05,
                "recommended_test": recommendation,
            }
        )
    results = pd.DataFrame(rows)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(TABLES_DIR / "assumption_checks.csv", index=False)
    _save_qq_plots(df)
    return results


def _save_qq_plots(df: pd.DataFrame) -> None:
    """Save Q-Q plots for each numeric feature, split by hazard status."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(7, 4, figsize=(16, 25))
    axes = axes.ravel()
    for index, feature in enumerate(NUMERIC_FEATURES):
        ax = axes[index]
        for status, color, label in [
            (False, "tab:blue", "Non-hazardous"),
            (True, "tab:red", "Hazardous"),
        ]:
            values = df.loc[df["hazardous"] == status, feature].dropna()
            stats.probplot(values, dist="norm", plot=ax)
            lines = ax.get_lines()
            lines[-2].set_color(color)
            lines[-1].set_color(color)
            lines[-2].set_label(label)
        ax.set_title(feature)
        ax.legend(fontsize=7)
    for ax in axes[len(NUMERIC_FEATURES) :]:
        ax.remove()
    fig.suptitle("Normal Q-Q plots by hazard status", y=0.995)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "assumption_qq_plots.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    checks = run_assumption_checks()
    print(checks[["feature", "recommended_test"]].to_string(index=False))
