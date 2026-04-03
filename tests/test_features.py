"""
tests/test_features.py

Unit tests for src.features.engineer

Run with: pytest tests/ -v
"""

import pytest
import pandas as pd
import numpy as np

from src.data.preprocess import clean
from src.features.engineer import (
    add_financial_features,
    add_tenure_features,
    add_contract_features,
    add_service_features,
    add_demographic_features,
    build_features,
)


# ── Fixture ───────────────────────────────────────────────────────────────────

@pytest.fixture
def clean_df():
    """Clean dataframe ready for feature engineering."""
    raw = pd.DataFrame({
        "customerID":       ["A1", "A2", "A3", "A4"],
        "gender":           ["Male", "Female", "Male", "Female"],
        "SeniorCitizen":    [0, 1, 0, 0],
        "Partner":          ["Yes", "No", "No", "Yes"],
        "Dependents":       ["No", "No", "Yes", "No"],
        "tenure":           [0, 12, 24, 60],
        "PhoneService":     ["No", "Yes", "Yes", "Yes"],
        "MultipleLines":    ["No phone service", "No", "Yes", "Yes"],
        "InternetService":  ["DSL", "Fiber optic", "DSL", "No"],
        "OnlineSecurity":   ["No", "No", "Yes", "No internet service"],
        "OnlineBackup":     ["Yes", "No", "No", "No internet service"],
        "DeviceProtection": ["No", "No", "Yes", "No internet service"],
        "TechSupport":      ["No", "No", "No", "No internet service"],
        "StreamingTV":      ["No", "Yes", "No", "No internet service"],
        "StreamingMovies":  ["No", "Yes", "No", "No internet service"],
        "Contract":         ["Month-to-month", "Month-to-month", "One year", "Two year"],
        "PaperlessBilling": ["Yes", "Yes", "No", "No"],
        "PaymentMethod":    [
            "Electronic check", "Electronic check",
            "Bank transfer (automatic)", "Credit card (automatic)",
        ],
        "MonthlyCharges":   [29.85, 70.70, 53.85, 42.30],
        "TotalCharges":     ["   ", "1889.5", "108.15", "1840.75"],
        "Churn":            ["No", "Yes", "No", "No"],
    })
    return clean(raw)


# ── Financial features ────────────────────────────────────────────────────────

class TestFinancialFeatures:

    def test_charge_per_tenure_fallback_on_zero_tenure(self, clean_df):
        """tenure=0 must not divide by zero — should fall back to MonthlyCharges."""
        result = add_financial_features(clean_df)
        assert result.loc[0, "charge_per_tenure"] == pytest.approx(29.85)

    def test_charge_per_tenure_normal_calculation(self, clean_df):
        """Normal tenure should compute TotalCharges / tenure."""
        result = add_financial_features(clean_df)
        expected = 1889.5 / 12
        assert result.loc[1, "charge_per_tenure"] == pytest.approx(expected, rel=1e-3)

    def test_monthly_charge_bin_is_1_to_4(self, clean_df):
        """Bin values must be integers between 1 and 4 inclusive."""
        result = add_financial_features(clean_df)
        assert result["monthly_charge_bin"].between(1, 4).all()

    def test_high_value_flag_is_binary(self, clean_df):
        """high_value_flag must only contain 0 or 1."""
        result = add_financial_features(clean_df)
        assert set(result["high_value_flag"].unique()).issubset({0, 1})

    def test_no_nulls_introduced(self, clean_df):
        result = add_financial_features(clean_df)
        new_cols = ["charge_per_tenure", "monthly_charge_bin", "high_value_flag"]
        assert result[new_cols].isnull().sum().sum() == 0


# ── Tenure features ───────────────────────────────────────────────────────────

class TestTenureFeatures:

    def test_tenure_zero_is_new_customer(self, clean_df):
        result = add_tenure_features(clean_df)
        assert result.loc[0, "is_new_customer"] == 1

    def test_tenure_60_is_loyal_customer(self, clean_df):
        result = add_tenure_features(clean_df)
        assert result.loc[3, "is_loyal_customer"] == 1

    def test_tenure_24_is_not_new_or_loyal(self, clean_df):
        result = add_tenure_features(clean_df)
        assert result.loc[2, "is_new_customer"]   == 0
        assert result.loc[2, "is_loyal_customer"] == 0

    def test_tenure_bucket_has_four_categories(self, clean_df):
        result = add_tenure_features(clean_df)
        valid = {"new", "developing", "established", "loyal"}
        actual = set(result["tenure_bucket"].astype(str).unique())
        assert actual.issubset(valid)

    def test_flags_are_binary(self, clean_df):
        result = add_tenure_features(clean_df)
        assert set(result["is_new_customer"].unique()).issubset({0, 1})
        assert set(result["is_loyal_customer"].unique()).issubset({0, 1})


# ── Contract features ─────────────────────────────────────────────────────────

class TestContractFeatures:

    def test_month_to_month_risk_score_is_2(self, clean_df):
        result = add_contract_features(clean_df)
        assert result.loc[0, "contract_risk_score"] == 2
        assert result.loc[1, "contract_risk_score"] == 2

    def test_one_year_risk_score_is_1(self, clean_df):
        result = add_contract_features(clean_df)
        assert result.loc[2, "contract_risk_score"] == 1

    def test_two_year_risk_score_is_0(self, clean_df):
        result = add_contract_features(clean_df)
        assert result.loc[3, "contract_risk_score"] == 0

    def test_autopay_methods_flagged(self, clean_df):
        """Bank transfer and credit card should be flagged as autopay."""
        result = add_contract_features(clean_df)
        assert result.loc[2, "is_autopay"] == 1  # bank transfer
        assert result.loc[3, "is_autopay"] == 1  # credit card

    def test_non_autopay_methods_not_flagged(self, clean_df):
        """Electronic check should not be autopay."""
        result = add_contract_features(clean_df)
        assert result.loc[0, "is_autopay"] == 0
        assert result.loc[1, "is_autopay"] == 0

    def test_risk_score_range(self, clean_df):
        result = add_contract_features(clean_df)
        assert result["contract_risk_score"].between(0, 2).all()


# ── Service features ──────────────────────────────────────────────────────────

class TestServiceFeatures:

    def test_fiber_no_protection_flagged(self, clean_df):
        """Fiber optic + no OnlineSecurity + no TechSupport → flag=1."""
        result = add_service_features(clean_df)
        # Row 1: Fiber optic, no security, no tech support
        assert result.loc[1, "fiber_no_protection"] == 1

    def test_non_fiber_not_flagged(self, clean_df):
        """DSL customer should not be flagged for fiber_no_protection."""
        result = add_service_features(clean_df)
        assert result.loc[0, "fiber_no_protection"] == 0

    def test_has_protection_with_security(self, clean_df):
        """Customer with OnlineSecurity=Yes should have has_protection=1."""
        result = add_service_features(clean_df)
        assert result.loc[2, "has_protection"] == 1

    def test_addon_count_max_is_6(self, clean_df):
        """addon_count should never exceed 6."""
        result = add_service_features(clean_df)
        assert result["addon_count"].max() <= 6

    def test_addon_count_min_is_0(self, clean_df):
        """addon_count should never be negative."""
        result = add_service_features(clean_df)
        assert result["addon_count"].min() >= 0

    def test_streaming_only_flag(self, clean_df):
        """Customer with streaming but no protection should be streaming_only."""
        result = add_service_features(clean_df)
        # Row 1: StreamingTV=Yes, StreamingMovies=Yes, no protection
        assert result.loc[1, "is_streaming_only"] == 1

    def test_no_nulls_in_service_features(self, clean_df):
        result = add_service_features(clean_df)
        new_cols = ["addon_count", "has_protection", "is_streaming_only", "fiber_no_protection"]
        assert result[new_cols].isnull().sum().sum() == 0


# ── Demographic features ──────────────────────────────────────────────────────

class TestDemographicFeatures:

    def test_senior_citizen_encoded(self, clean_df):
        result = add_demographic_features(clean_df)
        assert result.loc[1, "is_senior"] == 1
        assert result.loc[0, "is_senior"] == 0

    def test_is_independent_no_partner_no_dependents(self, clean_df):
        """Customer with no partner and no dependents should be independent."""
        result = add_demographic_features(clean_df)
        # Row 1: Partner=No, Dependents=No
        assert result.loc[1, "is_independent"] == 1

    def test_is_independent_false_with_partner(self, clean_df):
        """Customer with a partner should not be independent."""
        result = add_demographic_features(clean_df)
        # Row 0: Partner=Yes
        assert result.loc[0, "is_independent"] == 0

    def test_demographic_flags_are_binary(self, clean_df):
        result = add_demographic_features(clean_df)
        for col in ["is_senior", "has_dependents", "has_partner", "is_independent"]:
            assert set(result[col].unique()).issubset({0, 1}), f"{col} is not binary"


# ── build_features (integration) ──────────────────────────────────────────────

class TestBuildFeatures:

    def test_no_nulls_in_output(self, clean_df):
        result = build_features(clean_df)
        assert result.isnull().sum().sum() == 0

    def test_row_count_preserved(self, clean_df):
        result = build_features(clean_df)
        assert len(result) == len(clean_df)

    def test_churn_column_present(self, clean_df):
        result = build_features(clean_df)
        assert "Churn" in result.columns

    def test_dropped_columns_absent(self, clean_df):
        """customerID and gender should be dropped by default."""
        result = build_features(clean_df, drop_originals=True)
        assert "customerID" not in result.columns
        assert "gender"     not in result.columns

    def test_drop_originals_false_keeps_raw_cols(self, clean_df):
        result = build_features(clean_df, drop_originals=False)
        assert "customerID" in result.columns

    def test_engineered_features_present(self, clean_df):
        result = build_features(clean_df)
        expected = [
            "charge_per_tenure", "is_new_customer", "is_loyal_customer",
            "contract_risk_score", "is_autopay", "fiber_no_protection",
            "addon_count", "has_protection", "is_independent",
        ]
        for col in expected:
            assert col in result.columns, f"Missing expected feature: {col}"

    def test_does_not_modify_input(self, clean_df):
        original_cols = clean_df.columns.tolist()
        build_features(clean_df)
        assert clean_df.columns.tolist() == original_cols