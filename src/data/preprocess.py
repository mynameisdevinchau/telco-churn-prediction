"""
src/data/preprocess.py

Cleaning pipeline for the raw Telco Customer Churn CSV.
Called by notebooks and the training script — never duplicate this logic.

Usage:
    from src.data.preprocess import load_raw, clean
    df = clean(load_raw("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"))
"""

import pandas as pd
import yaml


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_raw(path: str) -> pd.DataFrame:
    """Load the raw CSV without any modifications."""
    return pd.read_csv(path)


def fix_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix the TotalCharges bug.

    Root cause: TotalCharges is stored as a string. 11 rows have blank
    strings (not NaN) representing new customers with tenure=0 who have
    not yet been billed.

    Fix: coerce to float via pd.to_numeric, then fill resulting NaNs
    with 0.0 — these customers genuinely have $0 in total charges.
    """
    df = df.copy()
    n_blank = (df["TotalCharges"].astype(str).str.strip() == "").sum()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
    if n_blank > 0:
        print(f"  [preprocess] Fixed {n_blank} blank TotalCharges → 0.0")
    return df


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Map Churn Yes/No → 1/0."""
    df = df.copy()
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full cleaning pipeline.

    Steps applied in order:
    1. Fix TotalCharges string → float
    2. Encode Churn as binary int
    3. Ensure SeniorCitizen is int
    4. Validate no nulls remain

    Returns the cleaned DataFrame. Does not modify the input.
    """
    df = (
        df
        .pipe(fix_total_charges)
        .pipe(encode_target)
    )
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)

    remaining_nulls = df.isnull().sum().sum()
    if remaining_nulls > 0:
        null_cols = df.columns[df.isnull().any()].tolist()
        print(f"  [preprocess] WARNING: {remaining_nulls} nulls remain in {null_cols}")
    else:
        print(f"  [preprocess] Clean — no nulls remaining. Shape: {df.shape}")

    return df


def run(config_path: str = "config.yaml") -> pd.DataFrame:
    """
    Full pipeline: load raw → clean → save to processed path.
    Run directly: python -m src.data.preprocess
    """
    cfg = load_config(config_path)
    print(f"Loading raw data from {cfg['data']['raw_path']}...")
    df_raw = load_raw(cfg["data"]["raw_path"])

    print("Cleaning...")
    df_clean = clean(df_raw)

    out_path = cfg["data"]["processed_path"]
    df_clean.to_parquet(out_path, index=False)
    print(f"Saved cleaned data to {out_path}")
    return df_clean


if __name__ == "__main__":
    run()