# NEO Hazard Prediction Project

Inferential statistics and predictive analysis of Potentially Hazardous Asteroids.

## Folder contents
- `data/nasa_raw.csv` - original Kaggle file (4,687 close-approach rows)
- `data/neo_asteroid_level_clean.csv` - cleaned, one row per asteroid, NO leaky columns (use this for modelling)
- `data/neo_asteroid_level_with_leaky_cols.csv` - same, plus MOID, magnitude, diameters (reference only)
- `scripts/clean_neo_data.py` - reproduces the cleaning (run from inside `scripts/`)

## Cleaning steps
1. Dropped constant and duplicate-unit columns
2. Removed label-leaking variables (MOID, absolute magnitude, diameters) from the modelling set
3. Derived orbit class (Aten/Apollo/Amor/Atira) from a, q, Q
4. Collapsed to one row per asteroid (3,692 asteroids; 576 hazardous = 15.6%)

## Notes
- Atira has only 7 asteroids: merge with Aten or exclude from class comparisons
- Baseline (orbital features only, 5-fold CV): LogReg AUC 0.826, Random Forest AUC 0.895
