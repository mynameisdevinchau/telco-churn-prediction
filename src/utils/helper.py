"""
src/utils/helpers.py

Shared utilities used across the pipeline.
"""

import yaml
import pandas as pd


def load_config(path: str = "config.yaml") -> dict:
    """Load and return the project config from config.yaml."""
    with open(path) as f:
        return yaml.safe_load(f)


def assign_intervention(row: pd.Series) -> str:
    """
    Map a customer's engineered features to a recommended retention action.

    Priority order:
    1. Month-to-month contract → offer upgrade (biggest churn driver)
    2. Fiber + no protection   → bundle add-on (55% churn, fiber anomaly)
    3. New customer            → onboarding outreach (47% churn in yr 1)
    4. High value              → account manager escalation
    5. Default                 → standard retention offer
    """
    if row.get("contract_risk_score", 0) == 2:
        return "Offer contract upgrade"
    elif row.get("fiber_no_protection", 0) == 1:
        return "Bundle security/support add-on"
    elif row.get("is_new_customer", 0) == 1:
        return "Trigger onboarding outreach"
    elif row.get("high_value_flag", 0) == 1:
        return "Escalate to account manager"
    else:
        return "Standard retention offer"


def validate_features(df: pd.DataFrame, expected_cols: list) -> None:
    """
    Assert that df contains all expected feature columns.
    Raises ValueError with a clear message if any are missing.
    """
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing {len(missing)} expected feature columns: {missing}"
        )


def print_section(title: str) -> None:
    """Print a formatted section header for script output."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")