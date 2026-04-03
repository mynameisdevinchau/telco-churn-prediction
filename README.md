# Telco Customer Churn Prediction

A production-style churn prediction pipeline built on the IBM Telco Customer Churn dataset. The project covers the full lifecycle — data cleaning, feature engineering, model training, and a Streamlit dashboard that surfaces actionable retention insights for business stakeholders.

**[Live Dashboard →](https://your-app.streamlit.app)** _(update after deploying to Streamlit Cloud)_

![CI](https://github.com/your-username/telco-churn-prediction/actions/workflows/ci.yml/badge.svg)

---

## The business problem

A telecom company loses roughly **26.5% of its customers** to churn each year. Churned customers have higher average monthly charges ($74) than retained ones ($61), meaning the revenue at risk is disproportionately large relative to the raw churn rate.

The goal: identify which customers are most likely to churn _before they leave_, and surface them to the retention team with a specific recommended action — not just a risk score.

---

## Results

| Model               | ROC-AUC   | PR-AUC    | Precision | Recall    | F1        |
| ------------------- | --------- | --------- | --------- | --------- | --------- |
| Logistic Regression | 0.821     | 0.594     | 0.516     | 0.770     | 0.618     |
| XGBoost             | 0.835     | 0.639     | 0.531     | 0.735     | 0.617     |
| **Random Forest**   | **0.835** | **0.624** | **0.506** | **0.826** | **0.627** |

**Random Forest selected as production model** — highest recall (0.826) means the most churners are correctly flagged for intervention. In a retention use case, missing a churner (false negative) is more costly than a wasted offer (false positive).

Key findings from the model:

- `contract_risk_score` is the dominant predictor — month-to-month customers churn at **42.7%** vs **2.8%** for two-year contracts
- Customers in their **first 12 months** churn at nearly 1 in 2 (47.7%)
- Fiber optic customers without protection services churn at **55%** — the single highest-risk segment identified
- Auto-pay customers churn at **16%** vs **35%** for non-auto-pay

---

## Dashboard

The Streamlit dashboard is built for a retention team, not a data scientist. Every chart answers a business question.

- **KPI row** — total customers, high-risk count, average churn probability, monthly revenue at risk
- **Segment views** — churn risk by contract type, internet service, and tenure bucket
- **SHAP feature importance** — shows which features drive churn and in which direction
- **Intervention table** — every high-risk customer ranked by churn probability with a specific recommended action, exportable as CSV
- **Model comparison tab** — ROC-AUC, PR-AUC, precision, recall, F1 for all three models at the selected threshold

---

## Project structure

```
telco-churn-prediction/
│
├── data/
│   ├── raw/                          # original CSV — never modified
│   └── processed/                    # cleaned and feature-engineered parquet files
│
├── notebooks/
│   ├── 01_eda_and_cleaning.ipynb     # data profiling, bug discovery, cleaning decisions
│   ├── 02_feature_engineering.ipynb  # 16 features across 5 groups with validation
│   └── 03_modeling.ipynb             # 3-model comparison, SMOTE, threshold tuning, SHAP
│
├── src/
│   ├── data/preprocess.py            # cleaning pipeline as importable functions
│   ├── features/engineer.py          # all feature logic — used by notebooks + dashboard
│   ├── models/train.py               # training script driven by config.yaml
│   └── utils/helpers.py              # shared utilities
│
├── tests/
│   ├── test_preprocess.py            # 16 tests — cleaning edge cases
│   └── test_features.py              # 34 tests — feature engineering edge cases
│
├── models/
│   ├── random_forest_churn.pkl
│   ├── xgboost_churn.pkl
│   ├── logistic_churn.pkl
│   └── feature_columns.csv
│
├── app.py                            # Streamlit dashboard
├── config.yaml                       # all hyperparameters, paths, thresholds
├── requirements.txt
├── conftest.py
└── .github/workflows/ci.yml          # runs pytest on every push
```

---

## Technical approach

### Data cleaning

The raw dataset has a subtle bug: `TotalCharges` is stored as a string, and 11 rows contain blank strings rather than `NaN` for new customers with `tenure=0`. Standard null checks miss this entirely. Fix: `pd.to_numeric(errors='coerce')` followed by `fillna(0)` — documented and tested.

### Feature engineering

16 features engineered across 5 groups from 21 raw columns:

| Group       | Features                                                            | Strongest signal                          |
| ----------- | ------------------------------------------------------------------- | ----------------------------------------- |
| Financial   | charge_per_tenure, monthly_charge_bin, high_value_flag              | high_value_flag: 37% vs 22% churn         |
| Tenure      | tenure_bucket, is_new_customer, is_loyal_customer                   | is_new_customer: 47% vs 17% churn         |
| Contract    | contract_risk_score, is_autopay, paperless_billing                  | contract_risk_score: 43% vs 3% churn      |
| Service     | addon_count, has_protection, is_streaming_only, fiber_no_protection | fiber_no_protection: **55% vs 17% churn** |
| Demographic | is_senior, has_dependents, has_partner, is_independent              | is_senior: 42% vs 24% churn               |

`fiber_no_protection` is an interaction feature created by combining the fiber optic anomaly identified in EDA with service adoption data. It has the highest churn rate of any engineered feature at 55%.

All feature logic lives in `src/features/engineer.py` — called by notebooks, the training script, and the dashboard with no duplication.

### Class imbalance

SMOTE applied to the training set only after the train/test split. Applying SMOTE before splitting causes data leakage — synthetic samples from the training set could appear in the test set, inflating all metrics. The balanced training set has 8,278 rows at a 50/50 churn rate.

### Threshold tuning

Default 0.5 threshold is wrong for imbalanced problems. Threshold swept from 0.20 to 0.70 in 0.05 steps; optimal threshold selected by maximising F1. In a retention use case, lower thresholds favour recall (catch more churners at the cost of more false alarms).

### Testing

50 unit tests across two test modules covering edge cases including:

- Division by zero for `tenure=0` customers in `charge_per_tenure`
- Blank string detection in `TotalCharges` (the real IBM dataset bug)
- Immutability — functions return new DataFrames, never mutate inputs
- All binary flag features contain only 0 or 1
- `fiber_no_protection` correctly identifies the highest-risk segment

---

## Quickstart

```bash
# Clone and set up
git clone https://github.com/your-username/telco-churn-prediction.git
cd telco-churn-prediction
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements.txt

# Download the dataset
# Place WA_Fn-UseC_-Telco-Customer-Churn.csv in data/raw/

# Run the pipeline
python -m src.data.preprocess     # clean raw data
python -m src.features.engineer   # build features
python -m src.models.train        # train all 3 models

# Run tests
pytest tests/ -v

# Launch dashboard
streamlit run app.py
```

---

## Dataset

IBM Telco Customer Churn — available on [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).
7,043 customers · 21 features · 26.5% churn rate.

---

## Stack

`Python` · `pandas` · `scikit-learn` · `XGBoost` · `imbalanced-learn` · `SHAP` · `Streamlit` · `matplotlib` · `seaborn` · `pytest` · `GitHub Actions`
