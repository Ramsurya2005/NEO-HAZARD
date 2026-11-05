"""Shared configuration for the NEO statistical and machine-learning project."""

from pathlib import Path

SEED = 42
ALPHA = 0.05

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "dataset" / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"

CLEAN_DATA_PATH = DATA_DIR / "neo_asteroid_level_clean.csv"
LEAKY_DATA_PATH = DATA_DIR / "neo_asteroid_level_with_leaky_cols.csv"

TARGET = "hazardous"
IDENTIFIER_COLUMNS = ["neo_id"]
CATEGORICAL_FEATURES = ["orbit_class"]

# These are the only model features permitted for the leak-free analysis.
NUMERIC_FEATURES = [
    "eccentricity",
    "semi_major_axis",
    "inclination",
    "asc_node_long",
    "orbital_period",
    "perihelion_dist",
    "aphelion_dist",
    "perihelion_arg",
    "tisserand_j",
    "orbit_uncertainty",
    "n_approaches",
    "mean_velocity_kms",
    "min_miss_dist_au",
]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# These columns are reserved for the explicitly invalid leakage demonstration.
LEAKY_COLUMNS = ["moid", "abs_magnitude", "dia_km_min", "dia_km_max"]
LEAKY_DEMO_FEATURES = MODEL_FEATURES + ["moid", "abs_magnitude"]

ORBIT_CLASS_MERGE = {"Atira": "Aten+Atira"}

REQUIRED_CLEAN_COLUMNS = (
    [TARGET]
    + IDENTIFIER_COLUMNS
    + MODEL_FEATURES
)
