"""Clean the NASA NEO close-approach CSV for the hazard-prediction project.

Fixes applied:
  1. Drop constant and duplicate-unit columns (Name, Orbiting Body, Equinox, unit copies)
  2. Remove label-leaking variables (MOID, Absolute Magnitude, diameters) from the modelling set
  3. Derive orbit class (Aten / Apollo / Amor / Atira / Other) from a, q, Q
  4. Collapse close-approach rows to ONE row per asteroid (so train/test can't share asteroids)
"""
import pandas as pd

IN_PATH = "../data/nasa_raw.csv"
OUT_DIR = "../data"

df = pd.read_csv(IN_PATH)

# --- 1. rename + drop redundant columns -------------------------------------
df = df.rename(columns={
    "Neo Reference ID": "neo_id",
    "Absolute Magnitude": "abs_magnitude",
    "Est Dia in KM(min)": "dia_km_min",
    "Est Dia in KM(max)": "dia_km_max",
    "Relative Velocity km per sec": "velocity_kms",
    "Miss Dist.(Astronomical)": "miss_dist_au",
    "Orbit Uncertainity": "orbit_uncertainty",
    "Minimum Orbit Intersection": "moid",
    "Jupiter Tisserand Invariant": "tisserand_j",
    "Eccentricity": "eccentricity",
    "Semi Major Axis": "semi_major_axis",
    "Inclination": "inclination",
    "Asc Node Longitude": "asc_node_long",
    "Orbital Period": "orbital_period",
    "Perihelion Distance": "perihelion_dist",
    "Perihelion Arg": "perihelion_arg",
    "Aphelion Dist": "aphelion_dist",
    "Mean Anomaly": "mean_anomaly",
    "Mean Motion": "mean_motion",
    "Close Approach Date": "approach_date",
    "Hazardous": "hazardous",
})
keep = ["neo_id", "approach_date", "hazardous", "abs_magnitude", "dia_km_min", "dia_km_max",
        "velocity_kms", "miss_dist_au", "orbit_uncertainty", "moid", "tisserand_j",
        "eccentricity", "semi_major_axis", "inclination", "asc_node_long", "orbital_period",
        "perihelion_dist", "perihelion_arg", "aphelion_dist", "mean_anomaly", "mean_motion"]
df = df[keep]

# --- 3. orbit class (standard NEO definitions) --------------------------------
a, q, Q = df["semi_major_axis"], df["perihelion_dist"], df["aphelion_dist"]
df["orbit_class"] = "Other"
df.loc[(a >= 1.0) & (q > 1.017) & (q < 1.3), "orbit_class"] = "Amor"
df.loc[(a >= 1.0) & (q <= 1.017), "orbit_class"] = "Apollo"
df.loc[(a < 1.0) & (Q > 0.983), "orbit_class"] = "Aten"
df.loc[Q < 0.983, "orbit_class"] = "Atira"

# --- 4. collapse to one row per asteroid ---------------------------------------
orbital = ["eccentricity", "semi_major_axis", "inclination", "asc_node_long", "orbital_period",
           "perihelion_dist", "perihelion_arg", "aphelion_dist", "tisserand_j", "orbit_uncertainty"]
static = df.groupby("neo_id")[orbital + ["moid", "abs_magnitude", "dia_km_min", "dia_km_max",
                                         "hazardous", "orbit_class"]].first()
approach = df.groupby("neo_id").agg(
    n_approaches=("approach_date", "count"),
    mean_velocity_kms=("velocity_kms", "mean"),
    min_miss_dist_au=("miss_dist_au", "min"),
).round(6)
ast = static.join(approach).reset_index()

# --- 2. split into modelling set (no leakage) and reference set -----------------
leaky = ["moid", "abs_magnitude", "dia_km_min", "dia_km_max"]
model_cols = ["neo_id", "hazardous", "orbit_class"] + orbital + ["n_approaches",
              "mean_velocity_kms", "min_miss_dist_au"]
ast[model_cols].to_csv(f"{OUT_DIR}/neo_asteroid_level_clean.csv", index=False)
ast.to_csv(f"{OUT_DIR}/neo_asteroid_level_with_leaky_cols.csv", index=False)

print("Asteroid-level shape:", ast[model_cols].shape)
print(ast["hazardous"].value_counts().to_dict())
print(ast["orbit_class"].value_counts().to_dict())
print(pd.crosstab(ast["orbit_class"], ast["hazardous"]))
