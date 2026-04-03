"""
src/features/engineer.py

All feature engineering logic in one place.
Called by notebooks, the training script, and the dashboard.

Usage:
    from src.features.engineer import build_features
    df_features = build_features(df_clean)
"""

import pandas as pd
import numpy as np


def _binary(series: pd.Series) -> pd.Series:
    """Map Yes→1, No→0, anything else→0."""
    return series.map({"Yes": 1, "No": 0}).fillna(0).astype(int)


def add_financial_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    charge_per_tenure  : average monthly spend over customer lifetime.
                         Falls back to MonthlyCharges for tenure=0.
    monthly_charge_bin : quartile bucket 1–4.
    high_value_flag    : 1 if MonthlyCharges > 75th percentile (~$89.85).
    """
    df = df.copy()
    df["charge_per_tenure"] = (
        df["TotalCharges"] / df["tenure"].replace(0, np.nan)
    ).fillna(df["MonthlyCharges"])

    df["monthly_charge_bin"] = pd.qcut(
        df["MonthlyCharges"], q=4, labels=[1, 2, 3, 4]
    ).astype(int)

    threshold = df["MonthlyCharges"].quantile(0.75)
    df["high_value_flag"] = (df["MonthlyCharges"] > threshold).astype(int)
    return df


def add_tenure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    tenure_bucket      : new / developing / established / loyal.
    is_new_customer    : 1 if tenure <= 12 months (47.7% churn from EDA).
    is_loyal_customer  : 1 if tenure >= 48 months (9.5% churn from EDA).
    """
    df = df.copy()
    df["tenure_bucket"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=["new", "developing", "established", "loyal"],
        include_lowest=True,
    )
    df["is_new_customer"]   = (df["tenure"] <= 12).astype(int)
    df["is_loyal_customer"] = (df["tenure"] >= 48).astype(int)
    return df


def add_contract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    contract_risk_score : month-to-month=2, one year=1, two year=0.
    is_autopay          : 1 if bank transfer or credit card auto-pay.
    paperless_billing   : binary encode.
    """
    df = df.copy()
    df["contract_risk_score"] = df["Contract"].map({
        "Month-to-month": 2,
        "One year":        1,
        "Two year":        0,
    })
    df["is_autopay"] = df["PaymentMethod"].isin(
        {"Bank transfer (automatic)", "Credit card (automatic)"}
    ).astype(int)
    df["paperless_billing"] = _binary(df["PaperlessBilling"])
    return df


def add_service_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    addon_count         : number of active add-on services (0–6).
    has_protection      : 1 if OnlineSecurity or TechSupport active.
    is_streaming_only   : streaming services but no protection/support.
    fiber_no_protection : fiber optic + no protection (55% churn rate).
    """
    df = df.copy()
    addon_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    df["addon_count"] = sum(
        (df[col] == "Yes").astype(int) for col in addon_cols
    )
    df["has_protection"] = (
        (df["OnlineSecurity"] == "Yes") | (df["TechSupport"] == "Yes")
    ).astype(int)
    df["is_streaming_only"] = (
        ((df["StreamingTV"] == "Yes") | (df["StreamingMovies"] == "Yes")) &
        (df["has_protection"] == 0)
    ).astype(int)
    df["fiber_no_protection"] = (
        (df["InternetService"] == "Fiber optic") &
        (df["has_protection"] == 0)
    ).astype(int)
    return df


def add_demographic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    is_senior      : SeniorCitizen as int.
    has_dependents : binary encode Dependents.
    has_partner    : binary encode Partner.
    is_independent : no partner AND no dependents (fewer switching costs).
    """
    df = df.copy()
    df["is_senior"]      = df["SeniorCitizen"].astype(int)
    df["has_dependents"] = _binary(df["Dependents"])
    df["has_partner"]    = _binary(df["Partner"])
    df["is_independent"] = (
        (df["has_partner"] == 0) & (df["has_dependents"] == 0)
    ).astype(int)
    return df


def build_features(df: pd.DataFrame, drop_originals: bool = True) -> pd.DataFrame:
    """
    Run all feature engineering steps.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame from src.data.preprocess.clean().
    drop_originals : bool
        Drop raw columns fully represented by engineered features.
        Set False for debugging.

    Returns
    -------
    pd.DataFrame with 16 engineered features added.
    """
    df = (
        df
        .pipe(add_financial_features)
        .pipe(add_tenure_features)
        .pipe(add_contract_features)
        .pipe(add_service_features)
        .pipe(add_demographic_features)
    )

    if drop_originals:
        cols_to_drop = [
            "customerID",
            "gender",
            "Partner",
            "Dependents",
            "PaperlessBilling",
            "PaymentMethod",
        ]
        bin_cols = [c for c in df.columns if c.endswith("_bin")]
        df = df.drop(
            columns=[c for c in cols_to_drop + bin_cols if c in df.columns]
        )

    return df


def run(config_path: str = "config.yaml") -> pd.DataFrame:
    """
    Full pipeline: load clean data → build features → save.
    Run directly: python -m src.features.engineer
    """
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    df_clean = pd.read_parquet(cfg["data"]["processed_path"])
    print(f"Loaded clean data: {df_clean.shape}")

    df_features = build_features(df_clean)
    print(f"Features built: {df_features.shape}")

    out_path = cfg["data"]["features_path"]
    df_features.to_parquet(out_path, index=False)
    print(f"Saved to {out_path}")
    return df_features


if __name__ == "__main__":
    run()