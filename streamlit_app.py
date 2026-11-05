"""Interactive Streamlit dashboard for the NEO hazard analysis."""

from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures"


st.set_page_config(
    page_title="NEO Hazard Analysis",
    page_icon="☄️",
    layout="wide",
)

st.title("☄️ Potentially Hazardous NEO Analysis")
st.caption("Interactive presentation of the reproducible statistical and machine-learning results.")


@st.cache_data
def read_table(filename: str) -> pd.DataFrame:
    """Read a generated project table."""
    return pd.read_csv(TABLES / filename)


def show_figure(filename: str, caption: str) -> None:
    """Display a generated figure when it is available."""
    path = FIGURES / filename
    if path.is_file():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.warning(f"Figure not found: {filename}. Run `python run_all.py` first.")


with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "View",
        [
            "Overview",
            "EDA",
            "Inference",
            "Model comparison",
            "Leakage demonstration",
            "Explainability",
        ],
    )
    st.divider()
    st.markdown(
        "Valid models exclude `moid`, `abs_magnitude`, and diameter columns."
    )


if page == "Overview":
    st.header("Project overview")
    audit = read_table("eda_audit.csv")
    balance = read_table("eda_hazard_balance.csv")
    model_audit = read_table("modeling_feature_audit.csv")
    columns = st.columns(3)
    columns[0].metric("Asteroids", f"{int(audit.loc[audit.metric == 'rows', 'value'].iloc[0]):,}")
    columns[1].metric("Hazardous", f"{int(balance.loc[balance.hazardous == True, 'count'].iloc[0]):,}")
    columns[2].metric("Valid model features", len(model_audit.loc[0, "model_features"].split(", ")))
    st.markdown(
        """
        This dashboard presents the results of the leak-free NEO hazard project.
        The dataset contains one row per asteroid, and `Atira` observations are
        merged into `Aten+Atira` for comparisons and modelling.
        """
    )
    st.subheader("Hazard balance")
    st.bar_chart(balance.set_index("hazardous")["count"])

elif page == "EDA":
    st.header("Exploratory data analysis")
    left, right = st.columns(2)
    with left:
        show_figure("eda_histograms.png", "Numeric feature distributions")
    with right:
        show_figure("eda_boxplots_by_hazard.png", "Features by hazard status")
    show_figure("eda_correlation_heatmap.png", "Numeric correlation heatmap")
    st.subheader("Skewness")
    st.dataframe(read_table("eda_skewness.csv"), use_container_width=True)
    st.subheader("Variance inflation factors")
    st.dataframe(read_table("eda_vif.csv"), use_container_width=True)

elif page == "Inference":
    st.header("Inferential statistics")
    st.dataframe(read_table("inference_numeric_tests.csv"), use_container_width=True)
    st.subheader("Orbit class and hazard")
    st.dataframe(read_table("inference_orbit_hazard.csv"), use_container_width=True)
    st.subheader("Logistic regression odds ratios")
    st.dataframe(read_table("inference_logistic_odds_ratios.csv"), use_container_width=True)
    show_figure("inference_logistic_calibration.png", "Logistic calibration")

elif page == "Model comparison":
    st.header("Predictive modelling")
    comparison = read_table("modeling_comparison.csv")
    metric = st.selectbox("Metric", sorted(comparison["metric"].unique()), index=0)
    selected = comparison[comparison["metric"] == metric].sort_values("mean", ascending=False)
    st.bar_chart(selected.set_index("model")["mean"])
    st.dataframe(selected, use_container_width=True)
    show_figure("modeling_roc_pr_curves.png", "ROC and precision-recall curves")
    show_figure("modeling_best_calibration.png", "Best-model calibration")
    show_figure("modeling_threshold_tuning.png", "Decision-threshold analysis")
    st.download_button(
        "Download model comparison CSV",
        comparison.to_csv(index=False),
        "modeling_comparison.csv",
        "text/csv",
    )

elif page == "Leakage demonstration":
    st.header("Leakage demonstration")
    st.error(
        "MOID and absolute magnitude define the hazardous label and must not be used "
        "as valid model features."
    )
    leakage = read_table("leakage_comparison.csv")
    selected = leakage[leakage["metric"].isin(["roc_auc", "pr_auc"])]
    st.bar_chart(selected.pivot(index="metric", columns="analysis", values="mean"))
    st.dataframe(leakage, use_container_width=True)
    show_figure("leakage_comparison.png", "Leak-free versus invalid leaky performance")

elif page == "Explainability":
    st.header("Model explainability")
    st.dataframe(
        read_table("explainability_interpretations.csv"),
        use_container_width=True,
    )
    show_figure("explainability_shap_summary.png", "SHAP summary")
    show_figure(
        "explainability_partial_dependence_top3.png",
        "Partial dependence for the top three features",
    )
    importance = read_table("explainability_permutation_importance.csv")
    st.subheader("Permutation importance")
    st.bar_chart(importance.head(10).set_index("feature")["importance_mean"])
    for feature in importance.head(3)["feature"]:
        show_figure(
            f"explainability_shap_dependence_{feature}.png",
            f"SHAP dependence: {feature}",
        )
