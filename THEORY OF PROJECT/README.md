# NEO Hazard Project Theory

## 1. Project purpose

This project studies whether a Near-Earth Object (NEO) can be classified as
potentially hazardous using orbital and close-approach information.

The project has two goals:

1. Understand how hazardous and non-hazardous asteroids differ.
2. Build and evaluate machine-learning models that predict the existing
   `hazardous` label.

This is a binary classification project:

- `True`: potentially hazardous
- `False`: non-hazardous

The label was already present in the original dataset. The project analyses
and predicts it; it does not invent the label.

## 2. Complete workflow

```text
Raw NASA/Kaggle data
        |
        v
Data cleaning and aggregation
        |
        v
One row per asteroid
        |
        v
Exploratory data analysis (EDA)
        |
        v
Statistical assumption checks
        |
        v
Inferential statistics
        |
        v
Machine-learning modelling
        |
        v
Leakage demonstration
        |
        v
Model explainability
        |
        v
Reproducibility checks
        |
        v
Tables, figures, paper, and dashboard
```

The complete workflow is started by `run_all.py`.

## 3. Data cleaning

The cleaning script is `dataset/scripts/clean_neo_data.py`.

It:

1. Renames long source column names to readable Python names.
2. Removes unnecessary or duplicate columns.
3. Creates an orbit class from orbital measurements.
4. Aggregates repeated close-approach records.
5. Produces one row per asteroid.
6. Creates a clean modelling dataset.
7. Creates a separate reference dataset containing leakage columns.

The original file has several close-approach records for the same asteroid.
Keeping all of them could allow the same asteroid to appear in both training
and testing data. Aggregation prevents that problem.

The close-approach values are summarised as:

- `n_approaches`: number of recorded approaches
- `mean_velocity_kms`: average approach velocity
- `min_miss_dist_au`: smallest recorded miss distance

The clean file is:

```text
dataset/data/neo_asteroid_level_clean.csv
```

## 4. Features

The permitted model features are defined in `src/config.py`.

### Orbital features

- `eccentricity`: how stretched the orbit is
- `semi_major_axis`: size of the orbit
- `inclination`: tilt of the orbit
- `asc_node_long`: orbit orientation
- `orbital_period`: time needed to complete one orbit
- `perihelion_dist`: closest distance to the Sun
- `aphelion_dist`: farthest distance from the Sun
- `perihelion_arg`: orientation of the orbit
- `tisserand_j`: orbital relationship with Jupiter
- `orbit_uncertainty`: uncertainty in the calculated orbit
- `orbit_class`: category such as Apollo, Amor, or Aten

### Close-approach features

- `n_approaches`: number of recorded approaches
- `mean_velocity_kms`: average approach speed
- `min_miss_dist_au`: closest recorded distance from Earth

## 5. Data leakage

Data leakage happens when a model receives information that directly reveals
the answer or would not be available in a valid prediction situation.

The hazardous label is rule-related to some source variables, including:

- `moid`
- `abs_magnitude`
- `dia_km_min`
- `dia_km_max`

These are excluded from valid modelling. The valid model only uses the
features in `MODEL_FEATURES` from `src/config.py`.

The file `src/leakage_demo.py` intentionally compares a valid model with an
invalid model. The invalid model appears almost perfect because it receives
variables that encode the label definition. This is a warning, not a valid
performance result.

## 6. Step 2: exploratory data analysis

The EDA code is `src/eda.py`.

It first loads the clean dataset and combines the small `Atira` group into
`Aten+Atira` for more stable comparisons.

EDA checks:

- Number of rows and columns
- Duplicate rows
- Duplicate asteroid IDs
- Missing values
- Data types
- Hazardous/non-hazardous class balance
- Orbit-class balance
- Descriptive statistics
- Correlations
- Skewness
- Variance inflation factor (VIF)

### EDA graphs

#### Histograms

Histograms show where the values of each numeric feature are concentrated.
The colours compare hazardous and non-hazardous asteroids.

#### Boxplots

The x-axis contains feature names. The y-axis contains the actual values of
the selected feature. Colours separate hazardous and non-hazardous groups.

The median and spread can be compared within each feature.

#### Correlation heatmap

The heatmap compares numeric features with one another:

- `+1`: strong positive relationship
- `0`: little or no linear relationship
- `-1`: strong negative relationship

Correlation shows association, not causation.

EDA outputs are saved under:

```text
outputs/tables/
outputs/figures/
```

## 7. Step 3: assumption checks

The assumption-checking code is `src/assumptions.py`.

Before choosing statistical tests, the project checks:

### Normality

The Shapiro-Wilk test checks whether values resemble a normal distribution.
Q-Q plots provide a visual version of the same check.

### Equal variance

The Levene test checks whether hazardous and non-hazardous groups have
similar variance.

The results determine whether a feature is suitable for a Welch t-test or
whether a non-parametric Mann-Whitney U test is more appropriate.

## 8. Step 4: inferential statistics

The code is `src/inference.py`.

This step explains relationships in the existing data rather than primarily
trying to predict new objects.

It uses:

- Mann-Whitney U tests for hazardous versus non-hazardous features
- Rank-biserial correlation for effect size
- Bootstrap confidence intervals
- Holm correction for multiple tests
- Chi-square test for orbit class versus hazard
- Cramer's V for association strength
- Kruskal-Wallis tests across orbit classes
- Pairwise rank tests
- VIF-pruned logistic regression
- Odds ratios and confidence intervals
- A calibration check

An odds ratio above 1 indicates higher estimated odds, while an odds ratio
below 1 indicates lower estimated odds, holding the other model variables
constant.

Statistical association does not prove that a feature causes hazard.

## 9. Step 5: predictive modelling

The code is `src/modeling.py`.

The target is:

```text
hazardous
```

The models receive the permitted orbital and approach features and produce a
probability of hazard.

Models compared include:

- Dummy classifier baseline
- Logistic regression
- Decision tree
- Random forest
- XGBoost
- Gradient boosting

Some models use class weights. Some use SMOTE to create synthetic minority
training examples. These operations are inside the modelling pipeline so
they are applied only to training data during validation.

### Cross-validation

The project uses repeated stratified five-fold cross-validation:

- Five groups are created.
- Four groups train the model.
- One group tests it.
- Every group is used for testing.
- The process is repeated three times.

Stratification keeps the hazardous/non-hazardous ratio similar in each fold.

### Evaluation metrics

- ROC-AUC: ranking quality across thresholds
- PR-AUC: precision-recall performance, useful for imbalance
- Precision: correctness of hazardous predictions
- Recall: hazardous objects successfully found
- F1: balance of precision and recall
- Balanced accuracy: average class-wise accuracy
- MCC: balanced correlation-style classification score
- Brier score: probability quality; lower is better

The project also tests different decision thresholds. A lower threshold
usually finds more hazardous objects but creates more false alarms. A higher
threshold usually gives fewer false alarms but may miss more hazardous
objects.

## 10. Step 6: leakage demonstration

The code is `src/leakage_demo.py`.

It evaluates the same type of model twice:

1. With only valid features.
2. With invalid leakage features added.

The comparison demonstrates why extremely high performance is not always
trustworthy. A model can appear excellent simply because it has been given
part of the answer.

## 11. Step 7: explainability

The code is `src/explainability.py`.

The best leak-free model is examined using:

### Permutation importance

One feature is shuffled at a time. If model performance decreases, that
feature was useful to the model.

### SHAP

SHAP estimates how features push individual predictions toward or away from
the hazardous class.

### Partial dependence

Partial dependence shows how the average prediction changes as a feature
changes.

These methods explain model behaviour. They do not prove that a feature
causes an asteroid to become hazardous.

## 12. Step 10: reproducibility

The code is `src/reproducibility.py`.

It verifies that:

- The clean dataset exists.
- The valid feature list contains no leakage columns.
- Important output files exist.
- The saved model audit also contains no leakage columns.

The project uses a fixed random seed of `42` to make random sampling,
cross-validation, and model results more repeatable.

## 13. Main files

| File | Purpose |
|---|---|
| `dataset/scripts/clean_neo_data.py` | Cleans and aggregates raw data |
| `src/config.py` | Shared paths, features, target, and constants |
| `src/eda.py` | Exploratory analysis and plots |
| `src/assumptions.py` | Normality and variance checks |
| `src/inference.py` | Statistical tests and logistic inference |
| `src/modeling.py` | Model training and evaluation |
| `src/leakage_demo.py` | Invalid-feature demonstration |
| `src/explainability.py` | SHAP and feature explanations |
| `src/reproducibility.py` | Final validation checks |
| `run_all.py` | Runs the complete workflow |
| `streamlit_app.py` | Displays outputs interactively |
| `paper/paper.md` | Research-paper-style report |

## 14. Running the project

From the project root:

```powershell
python -m pip install -r requirements.txt
$env:PYTHONPATH = (Get-Location).Path
python run_all.py
```

To open the dashboard:

```powershell
streamlit run streamlit_app.py
```

## 15. Main conclusion

The project shows that valid orbital and close-approach features contain
useful information for distinguishing hazardous and non-hazardous NEOs.
XGBoost performed best among the tested models, but the result is a research
benchmark rather than a ready-to-deploy safety system.

The most important principles are:

1. Use one row per asteroid.
2. Separate the target from the input features.
3. Remove variables that leak the target definition.
4. Use statistical tests appropriate for the data.
5. Evaluate with stratified cross-validation.
6. Report several metrics, not accuracy alone.
7. Treat model explanations as associations, not causes.
