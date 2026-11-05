# Predicting Potentially Hazardous Near-Earth Objects from Orbital Parameters: A Statistical and Machine Learning Approach

## Abstract

This study evaluates whether potentially hazardous near-Earth objects (NEOs) can be classified from orbital and close-approach parameters without using variables that define the hazardous label. The analysis uses 3,692 asteroid-level observations derived from a 1995–2016 Kaggle snapshot, including 576 hazardous objects (15.6%). After one-row-per-asteroid aggregation, Atira objects were merged into `Aten+Atira`, and MOID, absolute magnitude, and diameter variables were excluded from valid modelling because they leak the target definition. Inferential tests identified distributional differences for selected orbital variables and an association between orbit class and hazard status. Among repeated stratified cross-validation models, XGBoost achieved the highest mean ROC-AUC (0.907 ± 0.011) and PR-AUC (0.631 ± 0.030). An intentionally invalid leakage experiment produced ROC-AUC 0.9997, demonstrating why those variables must not be used. Explainability analyses identified orbit uncertainty and perihelion distance as the most influential raw features. Results support useful statistical discrimination, but not causal interpretation or operational deployment without external validation.

## Introduction

Near-Earth object hazard assessment is an important scientific and operational problem. Statistical models can help characterize which recorded orbital and approach parameters are associated with the historical hazardous label and can provide a reproducible benchmark for predictive analysis.

This project addresses two questions: (1) which leak-free orbital and approach variables differ between hazardous and non-hazardous objects, and (2) how well common statistical and machine-learning classifiers discriminate the label under stratified validation? The study emphasizes inferential effect sizes, confidence intervals, class imbalance, calibration, and explicit leakage control.

## Related Work

*Placeholder for citations on NEO hazard definitions, orbital dynamics, asteroid population modelling, imbalanced classification, and interpretable machine learning.*

## Data and Preprocessing

The source is a Kaggle snapshot of NASA NEO close-approach data covering 1995–2016. The cleaning workflow collapsed repeated close-approach rows to one row per asteroid, producing 3,692 asteroid-level observations. The target contains 576 hazardous and 3,116 non-hazardous objects.

The clean modelling variables were eccentricity, semi-major axis, inclination, ascending-node longitude, orbital period, perihelion distance, perihelion argument, aphelion distance, Tisserand invariant, orbit uncertainty, number of approaches, mean velocity, minimum miss distance, and orbit class. Atira was merged into `Aten+Atira` because it contained only seven objects.

MOID, absolute magnitude, and diameter variables were excluded from valid models. NASA's hazardous definition is rule-derived from MOID and absolute magnitude thresholds, so including those columns would expose the target definition. They were used only in the explicitly invalid leakage demonstration.

## Methods

### Exploratory and assumption checks

The audit assessed dimensions, data types, missing values, duplicate rows and IDs, class balance, groupwise descriptive statistics, orbit-class counts, histograms, hazard-status boxplots, correlations, skewness, and variance inflation factors. Shapiro–Wilk tests were applied to reproducible subsamples, Q–Q plots were generated, and Levene tests assessed variance equality.

### Inferential statistics

Because normality was rejected for the numeric variables, hazardous and non-hazardous groups were compared using two-sided Mann–Whitney U tests. Rank-biserial correlations quantified effect size, and bootstrap 95% confidence intervals quantified median differences. Holm correction controlled multiplicity.

Orbit class and hazard were assessed with a chi-square test, Cramer's V, expected-count checks, and standardized residuals. Numeric differences across orbit classes were assessed with Kruskal–Wallis tests and pairwise rank comparisons with Holm correction. A statsmodels logistic regression used iterative VIF pruning, odds ratios, 95% confidence intervals, and a Hosmer–Lemeshow-style decile calibration check.

### Predictive modelling

Models included a dummy baseline, weighted and SMOTE logistic regression, a weighted decision tree, weighted and SMOTE random forests, XGBoost, and gradient boosting. Preprocessing and SMOTE were performed inside cross-validation pipelines. Evaluation used repeated stratified five-fold cross-validation with three repeats and ROC-AUC, PR-AUC, precision, recall, F1, balanced accuracy, MCC, and Brier score. Randomized tuning used an 80/20 tuning split with three-fold stratified search. A Wilcoxon signed-rank test compared the best model's repeated-CV ROC-AUC against the dummy baseline.

### Explainability

The best leak-free model was examined with permutation importance, SHAP summary and dependence plots, and partial dependence for the three highest-ranked raw features. These outputs are associative and do not establish causal effects.

## Results

### Data and assumptions

The dataset contained 3,692 rows and 16 columns, with no duplicate rows or duplicate asteroid IDs. The hazardous prevalence was 15.6%. Orbit classes were Apollo (2,038), Amor (926), Aten (721), and Aten+Atira (7). Shapiro–Wilk tests rejected normality for all tested numeric variables in at least one group, so non-parametric inference was used.

### Inferential results

After Holm correction, eccentricity, perihelion distance, and aphelion distance showed evidence of distributional differences between hazardous and non-hazardous groups. The orbit-class/hazard association was significant, χ²(3) = 190.22, p < 0.001, with Cramer's V = 0.227. The minimum expected cell count was 1.09, so the chi-square approximation should be interpreted cautiously.

The VIF-pruned logistic model retained significant adjusted associations for inclination (odds ratio 0.973), orbit uncertainty (odds ratio 0.661), mean velocity (odds ratio 1.049), minimum miss distance (odds ratio 0.113), and the Amor indicator (odds ratio 0.099). The Hosmer–Lemeshow-style calibration check was significant, χ²(8) = 29.59, p < 0.001, indicating calibration lack of fit in this uncalibrated model.

### Predictive results

The best repeated-CV model was XGBoost, with ROC-AUC 0.907 ± 0.011, PR-AUC 0.631 ± 0.030, balanced accuracy 0.688 ± 0.016, MCC 0.468 ± 0.027, and Brier score 0.084 ± 0.004. The dummy baseline had ROC-AUC 0.500. The paired Wilcoxon comparison favored XGBoost over the dummy model, p = 3.05 × 10⁻⁵.

On the held-out diagnostic split, XGBoost achieved ROC-AUC 0.905, PR-AUC 0.627, precision 0.722, recall 0.361, F1 0.481, balanced accuracy 0.668, MCC 0.454, and Brier score 0.086 at the default threshold.

### Leakage demonstration

The valid leak-free XGBoost model had repeated-CV ROC-AUC 0.907 and PR-AUC 0.631. Adding MOID and absolute magnitude increased these to 0.9997 and 0.9978. This apparent near-perfect performance is invalid because the added variables encode the rule used to derive the hazardous label.

### Explainability

Permutation importance ranked orbit uncertainty first (mean importance 0.188), followed by perihelion distance (0.130) and perihelion argument (0.035). These rankings describe predictive contribution under the fitted model and dataset; they should not be interpreted as causal effects.

Figures: [EDA histograms](../outputs/figures/eda_histograms.png), [correlation heatmap](../outputs/figures/eda_correlation_heatmap.png), [model ROC/PR curves](../outputs/figures/modeling_roc_pr_curves.png), [leakage comparison](../outputs/figures/leakage_comparison.png), and [SHAP summary](../outputs/figures/explainability_shap_summary.png).

## Discussion

The results show that leak-free orbital and approach variables contain useful predictive signal for the historical hazardous label. XGBoost outperformed the dummy baseline and the other evaluated model families by repeated-CV ROC-AUC, while the precision-recall results reflect the difficulty of identifying a minority class.

The inferential and predictive results are complementary. Significant group differences do not by themselves imply high predictive value, and model importance does not establish a physical or causal mechanism. Calibration diagnostics indicate that probability estimates require additional calibration before they could support decision-making.

Limitations include the 1995–2016 Kaggle snapshot, possible dataset-specific measurement and selection effects, the very small Atira-derived group, and a label that is rule-derived rather than an independent observed outcome. The one-row-per-asteroid aggregation also summarizes repeated approach records and may discard temporal structure.

## Conclusion

Using only leak-free orbital and approach parameters, the best evaluated model achieved repeated-CV ROC-AUC 0.907 and PR-AUC 0.631. The leakage experiment confirms that MOID, absolute magnitude, and diameter variables must not be used for valid prediction of this rule-derived label. The project provides a reproducible statistical and machine-learning benchmark, but not a deployment-ready hazard assessment system.

## Future Work

Future work should replicate the analysis on the NASA JPL Small-Body Database, evaluate temporal and external validation, investigate calibrated probabilities, model repeated close approaches explicitly, and assess robustness across changes in hazard-definition policy and measurement uncertainty.
