"""
tests/test_preprocess.py

Unit tests for src.data.preprocess

Run with: pytest tests/ -v
"""

import pytest
import pandas as pd
import numpy as np

from src.data.preprocess import fix_total_charges, encode_target, clean


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Minimal realistic dataframe mirroring the raw Telco CSV structure."""
    return pd.DataFrame({
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


# ── fix_total_charges ─────────────────────────────────────────────────────────

class TestFixTotalCharges:

    def test_blank_string_becomes_zero(self, sample_df):
        """Blank string (the real IBM dataset bug) should become 0.0."""
        result = fix_total_charges(sample_df)
        assert result.loc[0, "TotalCharges"] == 0.0

    def test_valid_string_parses_correctly(self, sample_df):
        """Valid numeric strings should parse to float."""
        result = fix_total_charges(sample_df)
        assert result.loc[1, "TotalCharges"] == 1889.5
        assert result.loc[2, "TotalCharges"] == 108.15

    def test_output_dtype_is_float(self, sample_df):
        """TotalCharges column must be float after fix."""
        result = fix_total_charges(sample_df)
        assert result["TotalCharges"].dtype == float

    def test_no_nulls_remain(self, sample_df):
        """No NaN values should remain after fix."""
        result = fix_total_charges(sample_df)
        assert result["TotalCharges"].isnull().sum() == 0

    def test_does_not_modify_input(self, sample_df):
        """Input DataFrame should not be mutated."""
        original_dtype = sample_df["TotalCharges"].dtype
        fix_total_charges(sample_df)
        assert sample_df["TotalCharges"].dtype == original_dtype

    def test_whitespace_only_string_becomes_zero(self):
        """Various whitespace patterns should all become 0.0."""
        df = pd.DataFrame({"TotalCharges": ["  ", "\t", " ", "100.0"]})
        result = fix_total_charges(df)
        assert result.loc[0, "TotalCharges"] == 0.0
        assert result.loc[1, "TotalCharges"] == 0.0
        assert result.loc[2, "TotalCharges"] == 0.0
        assert result.loc[3, "TotalCharges"] == 100.0


# ── encode_target ─────────────────────────────────────────────────────────────

class TestEncodeTarget:

    def test_yes_maps_to_one(self, sample_df):
        result = encode_target(sample_df)
        assert result.loc[1, "Churn"] == 1

    def test_no_maps_to_zero(self, sample_df):
        result = encode_target(sample_df)
        assert result.loc[0, "Churn"] == 0

    def test_output_is_integer(self, sample_df):
        result = encode_target(sample_df)
        assert result["Churn"].dtype in [int, np.int32, np.int64]

    def test_all_values_are_binary(self, sample_df):
        result = encode_target(sample_df)
        assert set(result["Churn"].unique()).issubset({0, 1})

    def test_does_not_modify_input(self, sample_df):
        original = sample_df["Churn"].copy()
        encode_target(sample_df)
        pd.testing.assert_series_equal(sample_df["Churn"], original)


# ── clean (integration) ───────────────────────────────────────────────────────

class TestClean:

    def test_no_nulls_after_clean(self, sample_df):
        result = clean(sample_df)
        assert result.isnull().sum().sum() == 0

    def test_total_charges_is_float(self, sample_df):
        result = clean(sample_df)
        assert result["TotalCharges"].dtype == float

    def test_churn_is_binary_int(self, sample_df):
        result = clean(sample_df)
        assert set(result["Churn"].unique()).issubset({0, 1})

    def test_shape_preserved(self, sample_df):
        """clean() should not drop any rows."""
        result = clean(sample_df)
        assert result.shape[0] == sample_df.shape[0]

    def test_senior_citizen_is_int(self, sample_df):
        result = clean(sample_df)
        assert result["SeniorCitizen"].dtype in [int, np.int32, np.int64]