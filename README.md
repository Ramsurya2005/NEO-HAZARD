# Predicting Potentially Hazardous Near-Earth Objects

Publication-quality inferential statistics and predictive modelling for potentially hazardous near-Earth objects.

## Project structure

- `dataset/data/neo_asteroid_level_clean.csv`: leak-free, one-row-per-asteroid modelling data.
- `dataset/data/neo_asteroid_level_with_leaky_cols.csv`: reference-only data for the Step 6 leakage demonstration.
- `dataset/scripts/clean_neo_data.py`: reproducible raw-data cleaning script.
- `src/`: reusable EDA, assumption-check, inference, modelling, leakage, explainability, and reproducibility code.
- `notebooks/`: notebooks for Steps 2–7 in execution order.
- `outputs/figures/`: generated PNG figures at 300 dpi.
- `outputs/tables/`: generated CSV tables.
- `paper/paper.md`: paper draft using the generated results.

Step 8, the optional regression extension, was intentionally skipped.

## Setup

Use Python 3 and install the declared dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run commands from the project root:

```powershell
$env:PYTHONPATH = (Get-Location).Path
```

## Reproduce the analysis

Run the complete implemented workflow:

```powershell
python run_all.py
```

This runs EDA, assumption checks, inferential statistics, predictive modelling, the leakage demonstration, explainability, and the final reproducibility checks. It does not run the optional Step 8 or rewrite the Step 9 paper draft.

## Interactive Streamlit dashboard

After generating the outputs, launch the interactive dashboard locally:

```powershell
python -m pip install streamlit
streamlit run streamlit_app.py
```

The dashboard presents the existing tables and figures interactively. It is intended for exploration and presentation; the notebooks and saved CSV/PNG outputs remain the reproducible research record.

Alternatively, run notebooks in order:

1. `notebooks/01_eda.ipynb`
2. `notebooks/02_assumptions.ipynb`
3. `notebooks/03_inference.ipynb`
4. `notebooks/04_modeling.ipynb`
5. `notebooks/05_leakage_demo.ipynb`
6. `notebooks/06_explainability.ipynb`

## Leakage checklist

Valid modelling uses only `src.config.MODEL_FEATURES`. The prohibited columns are `moid`, `abs_magnitude`, `dia_km_min`, and `dia_km_max`. The leakage demonstration uses only `moid` and `abs_magnitude` from the reference-only file and is never part of the valid model benchmark.

The final checklist is generated at `outputs/tables/reproducibility_checklist.csv`.
