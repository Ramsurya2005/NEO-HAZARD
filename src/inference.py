"""Inferential statistics for the leak-free NEO hazard dataset."""

from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from src.config import (
    ALPHA,
    CLEAN_DATA_PATH,
    FIGURES_DIR,
    NUMERIC_FEATURES,
    SEED,
    TABLES_DIR,
)
from src.eda import load_data


def _holm_adjust(p_values: list[float]) -> np.ndarray:
    """Return Holm-adjusted p-values in the original order."""
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running_max = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (len(p_values) - rank) * p_values[index])
        running_max = max(running_max, value)
        adjusted[index] = running_max
    return adjusted


def _bootstrap_difference(
    first: np.ndarray,
    second: np.ndarray,
    statistic: str,
    n_boot: int = 2000,
) -> tuple[float, float, float]:
    """Bootstrap the first-minus-second difference in means or medians."""
    rng = np.random.default_rng(SEED)
    observed = (
        np.median(first) - np.median(second)
        if statistic == "median"
        else np.mean(first) - np.mean(second)
    )
    estimates = np.empty(n_boot)
    for index in range(n_boot):
        first_sample = rng.choice(first, len(first), replace=True)
        second_sample = rng.choice(second, len(second), replace=True)
        estimates[index] = (
            np.median(first_sample) - np.median(second_sample)
            if statistic == "median"
            else np.mean(first_sample) - np.mean(second_sample)
        )
    low, high = np.quantile(estimates, [ALPHA / 2, 1 - ALPHA / 2])
    return observed, low, high


def _rank_biserial(u_statistic: float, n_first: int, n_second: int) -> float:
    """Calculate rank-biserial correlation for a Mann-Whitney test."""
    return 2 * u_statistic / (n_first * n_second) - 1


def _numeric_inference(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    p_values = []
    for feature in NUMERIC_FEATURES:
        non_hazardous = df.loc[~df["hazardous"], feature].dropna().to_numpy()
        hazardous = df.loc[df["hazardous"], feature].dropna().to_numpy()
        test = stats.mannwhitneyu(
            hazardous, non_hazardous, alternative="two-sided", method="asymptotic"
        )
        observed, ci_low, ci_high = _bootstrap_difference(
            hazardous, non_hazardous, statistic="median"
        )
        effect = _rank_biserial(test.statistic, len(hazardous), len(non_hazardous))
        p_values.append(test.pvalue)
        rows.append(
            {
                "analysis": "hazardous_vs_non_hazardous",
                "feature": feature,
                "test": "Mann-Whitney U",
                "statistic": test.statistic,
                "p": test.pvalue,
                "effect_size": effect,
                "effect_size_name": "rank-biserial correlation",
                "estimate_hazardous_minus_non_hazardous": observed,
                "ci_95_low": ci_low,
                "ci_95_high": ci_high,
            }
        )
    results = pd.DataFrame(rows)
    results["adjusted_p"] = _holm_adjust(p_values)
    results["interpretation"] = np.where(
        results["adjusted_p"] < ALPHA,
        "Evidence of a distributional difference after Holm correction",
        "No evidence of a distributional difference after Holm correction",
    )
    return results


def _orbit_hazard_inference(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    table = pd.crosstab(df["orbit_class"], df["hazardous"])
    chi2, p_value, dof, expected = stats.chi2_contingency(table)
    n = table.to_numpy().sum()
    rows = pd.DataFrame(
        [
            {
                "analysis": "orbit_class_vs_hazard",
                "test": "Chi-square independence",
                "statistic": chi2,
                "degrees_of_freedom": dof,
                "p": p_value,
                "effect_size": np.sqrt(chi2 / (n * (min(table.shape) - 1))),
                "effect_size_name": "Cramer's V",
                "minimum_expected_count": expected.min(),
                "all_expected_counts_at_least_5": bool((expected >= 5).all()),
            }
        ]
    )
    rows["adjusted_p"] = rows["p"]
    rows["interpretation"] = np.where(
        rows["p"] < ALPHA,
        "Evidence of association between orbit class and hazard",
        "No evidence of association between orbit class and hazard",
    )
    expected_df = pd.DataFrame(expected, index=table.index, columns=table.columns)
    residuals = (table - expected_df) / np.sqrt(expected_df)
    residuals.index.name = "orbit_class"
    residuals.to_csv(TABLES_DIR / "inference_standardized_residuals.csv")
    return rows, table


def _kruskal_dunn_inference(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in NUMERIC_FEATURES:
        groups = [
            group[feature].dropna().to_numpy()
            for _, group in df.groupby("orbit_class", sort=True)
        ]
        classes = sorted(df["orbit_class"].unique())
        kruskal = stats.kruskal(*groups)
        n = sum(len(group) for group in groups)
        epsilon_squared = max(0.0, (kruskal.statistic - len(groups) + 1) / (n - len(groups)))
        rows.append(
            {
                "analysis": "feature_across_orbit_classes",
                "feature": feature,
                "test": "Kruskal-Wallis",
                "comparison": "all orbit classes",
                "statistic": kruskal.statistic,
                "p": kruskal.pvalue,
                "effect_size": epsilon_squared,
                "effect_size_name": "epsilon-squared",
            }
        )
        pair_rows = []
        pair_p = []
        for first, second in combinations(classes, 2):
            first_values = df.loc[df["orbit_class"] == first, feature].dropna()
            second_values = df.loc[df["orbit_class"] == second, feature].dropna()
            test = stats.mannwhitneyu(first_values, second_values, alternative="two-sided")
            pair_p.append(test.pvalue)
            pair_rows.append(
                {
                    "analysis": "feature_across_orbit_classes",
                    "feature": feature,
                    "test": "Dunn pairwise rank test",
                    "comparison": f"{first} vs {second}",
                    "statistic": test.statistic,
                    "p": test.pvalue,
                    "effect_size": _rank_biserial(
                        test.statistic, len(first_values), len(second_values)
                    ),
                    "effect_size_name": "rank-biserial correlation",
                }
            )
        pair_adjusted = _holm_adjust(pair_p)
        for row, adjusted in zip(pair_rows, pair_adjusted):
            row["adjusted_p"] = adjusted
            row["interpretation"] = (
                "Significant pairwise difference after Holm correction"
                if adjusted < ALPHA
                else "No significant pairwise difference after Holm correction"
            )
            rows.append(row)
    result = pd.DataFrame(rows)
    result["adjusted_p"] = result["adjusted_p"].fillna(result["p"])
    missing_interpretation = result["interpretation"].isna()
    result.loc[missing_interpretation, "interpretation"] = np.where(
        result.loc[missing_interpretation, "adjusted_p"] < ALPHA,
        "Evidence of differences across orbit classes",
        "No evidence of differences across orbit classes",
    )
    return result


def _vif_prune(df: pd.DataFrame, threshold: float = 10.0) -> tuple[pd.DataFrame, list[str]]:
    """Iteratively remove the highest-VIF feature until all VIFs are acceptable."""
    features = list(NUMERIC_FEATURES) + ["orbit_Apollo", "orbit_Amor", "orbit_Aten+Atira"]
    design = pd.concat(
        [df[NUMERIC_FEATURES], pd.get_dummies(df["orbit_class"], prefix="orbit", dtype=float)],
        axis=1,
    )
    design = design.reindex(columns=features, fill_value=0.0).astype(float)
    removed = []
    while len(features) > 2:
        x = sm.add_constant(design[features], has_constant="add")
        vif_values = {}
        for feature in features:
            target = x[feature].to_numpy()
            predictors = x.drop(columns=feature).to_numpy()
            coefficients, *_ = np.linalg.lstsq(predictors, target, rcond=None)
            residuals = target - predictors @ coefficients
            total_ss = np.sum((target - target.mean()) ** 2)
            r_squared = 1 - np.sum(residuals**2) / total_ss if total_ss else 0.0
            vif_values[feature] = np.inf if r_squared >= 1 - 1e-10 else 1 / (1 - r_squared)
        highest = max(vif_values, key=vif_values.get)
        if vif_values[highest] <= threshold:
            break
        removed.append(highest)
        features.remove(highest)
    return design[features], removed


def _logistic_inference(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    design, removed = _vif_prune(df)
    model = sm.Logit(df["hazardous"].astype(int), sm.add_constant(design)).fit(disp=False)
    confidence = model.conf_int()
    coefficients = pd.DataFrame(
        {
            "feature": model.params.index,
            "odds_ratio": np.exp(model.params.values),
            "ci_95_low": np.exp(confidence[0].values),
            "ci_95_high": np.exp(confidence[1].values),
            "p": model.pvalues.values,
        }
    )
    coefficients["adjusted_p"] = _holm_adjust(coefficients["p"].tolist())
    coefficients["interpretation"] = np.where(
        coefficients["adjusted_p"] < ALPHA,
        "Evidence of association after Holm correction",
        "No evidence of association after Holm correction",
    )
    pd.DataFrame({"removed_feature": removed}).to_csv(
        TABLES_DIR / "inference_vif_pruned_features.csv", index=False
    )

    probabilities = model.predict(sm.add_constant(design))
    calibration = pd.DataFrame({"observed": df["hazardous"].astype(int), "predicted": probabilities})
    calibration["decile"] = pd.qcut(calibration["predicted"], q=10, duplicates="drop")
    grouped = calibration.groupby("decile", observed=True)
    observed = grouped["observed"].sum()
    expected = grouped["predicted"].sum()
    total = grouped.size()
    hl_stat = np.sum(
        (observed - expected) ** 2 / np.maximum(expected, 1e-12)
        + ((total - observed) - (total - expected)) ** 2
        / np.maximum(total - expected, 1e-12)
    )
    hl_df = max(1, len(observed) - 2)
    hl_p = stats.chi2.sf(hl_stat, hl_df)
    calibration_result = pd.DataFrame(
        [
            {
                "test": "Hosmer-Lemeshow style decile calibration",
                "statistic": hl_stat,
                "degrees_of_freedom": hl_df,
                "p": hl_p,
                "interpretation": "No evidence of lack of fit"
                if hl_p >= ALPHA
                else "Evidence of calibration lack of fit",
            }
        ]
    )
    return coefficients, calibration_result


def run_inference(path: Path = CLEAN_DATA_PATH) -> None:
    """Run all Step 4 analyses and save tables and diagnostic figures."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data(path)
    numeric = _numeric_inference(df)
    orbit, contingency = _orbit_hazard_inference(df)
    class_tests = _kruskal_dunn_inference(df)
    logistic, calibration = _logistic_inference(df)
    numeric.to_csv(TABLES_DIR / "inference_numeric_tests.csv", index=False)
    orbit.to_csv(TABLES_DIR / "inference_orbit_hazard.csv", index=False)
    contingency.to_csv(TABLES_DIR / "inference_orbit_hazard_contingency.csv")
    class_tests.to_csv(TABLES_DIR / "inference_orbit_class_tests.csv", index=False)
    logistic.to_csv(TABLES_DIR / "inference_logistic_odds_ratios.csv", index=False)
    calibration.to_csv(TABLES_DIR / "inference_logistic_calibration.csv", index=False)
    summary = pd.concat(
        [
            numeric,
            orbit,
            class_tests,
            logistic.assign(
                analysis="logistic_regression",
                test="VIF-pruned statsmodels Logit",
                statistic=np.nan,
                effect_size=np.nan,
                effect_size_name="odds ratio",
            ),
        ],
        ignore_index=True,
        sort=False,
    )
    summary.to_csv(TABLES_DIR / "inference_summary.csv", index=False)
    _save_calibration_plot(df, logistic)


def _save_calibration_plot(df: pd.DataFrame, logistic: pd.DataFrame) -> None:
    """Save a calibration curve for the fitted VIF-pruned logistic model."""
    design, _ = _vif_prune(df)
    model = sm.Logit(df["hazardous"].astype(int), sm.add_constant(design)).fit(disp=False)
    predicted = model.predict(sm.add_constant(design))
    bins = pd.qcut(predicted, q=10, duplicates="drop")
    grouped = pd.DataFrame({"predicted": predicted, "observed": df["hazardous"].astype(int)}).groupby(
        bins, observed=True
    )
    mean_predicted = grouped["predicted"].mean()
    mean_observed = grouped["observed"].mean()
    plt.figure(figsize=(6, 6))
    plt.plot(mean_predicted, mean_observed, "o-", label="VIF-pruned logistic")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed hazard rate")
    plt.title("Logistic calibration by predicted-probability decile")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "inference_logistic_calibration.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    run_inference()
    print("Generated Step 4 inferential outputs.")
