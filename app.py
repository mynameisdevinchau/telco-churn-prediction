"""
app.py — Telco Customer Churn Dashboard
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import joblib
import shap

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Telco Churn Dashboard",
    page_icon="📡",
    layout="wide",
)

sns.set_theme(style="whitegrid", font_scale=1.0)
plt.rcParams["figure.dpi"] = 120

# ── Load models & data ────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    xgb = joblib.load("models/xgboost_churn.pkl")
    rf  = joblib.load("models/random_forest_churn.pkl")
    lr  = joblib.load("models/logistic_churn.pkl")
    return {"XGBoost": xgb, "Random Forest": rf, "Logistic Regression": lr}

@st.cache_data
def load_data():
    df = pd.read_parquet("data/processed/telco_features.parquet")

    # Encode — mirrors modeling notebook exactly
    raw_service_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies", "PhoneService"
    ]
    df = df.drop(columns=[c for c in raw_service_cols if c in df.columns])
    cat_cols = ["MultipleLines", "InternetService", "Contract", "tenure_bucket"]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)
    return df

@st.cache_data
def load_feature_cols():
    raw = pd.read_csv("models/feature_columns.csv")
    # CSV was saved with an auto-index header — feature names are in the first column
    return raw.iloc[:, 0].tolist()

models      = load_models()
df          = load_data()
feat_cols   = load_feature_cols()

X = df[feat_cols]
y = df["Churn"]

THRESHOLD = 0.40   # tuned in modeling notebook — update if yours differs

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📡 Churn Dashboard")
    st.markdown("---")

    selected_model = st.selectbox(
        "Model",
        options=list(models.keys()),
        index=0,
    )

    st.markdown("---")
    st.markdown("**Filters**")

    contract_filter = st.multiselect(
        "Contract type",
        options=["Month-to-month", "One year", "Two year"],
        default=["Month-to-month", "One year", "Two year"],
    )

    internet_filter = st.multiselect(
        "Internet service",
        options=["DSL", "Fiber optic", "No"],
        default=["DSL", "Fiber optic", "No"],
    )

    risk_threshold = st.slider(
        "Churn risk threshold",
        min_value=0.20,
        max_value=0.70,
        value=THRESHOLD,
        step=0.05,
        help="Customers above this probability are flagged as high-risk",
    )

    st.markdown("---")
    st.caption("Data: IBM Telco Customer Churn · Model: XGBoost · SMOTE balanced")

# ── Predictions ───────────────────────────────────────────────────────────────
model = models[selected_model]
probs = model.predict_proba(X)[:, 1]
preds = (probs >= risk_threshold).astype(int)

df_display = df.copy()
df_display["churn_probability"] = probs
df_display["predicted_churn"]   = preds

# Reconstruct readable columns from encoded df for display
# Re-load raw features for display purposes
@st.cache_data
def load_raw():
    raw = pd.read_csv("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    raw["TotalCharges"] = pd.to_numeric(raw["TotalCharges"], errors="coerce").fillna(0)
    return raw

df_raw = load_raw()
df_display["Contract"]        = df_raw["Contract"].values
df_display["InternetService"] = df_raw["InternetService"].values
df_display["customerID"]      = df_raw["customerID"].values

# Apply sidebar filters
mask = (
    df_display["Contract"].isin(contract_filter) &
    df_display["InternetService"].isin(internet_filter)
)
df_filtered = df_display[mask].copy()

# ── Intervention logic ────────────────────────────────────────────────────────
def assign_intervention(row):
    if row["contract_risk_score"] == 2:
        return "Offer contract upgrade"
    elif row.get("fiber_no_protection", 0) == 1:
        return "Bundle security/support add-on"
    elif row.get("is_new_customer", 0) == 1:
        return "Trigger onboarding outreach"
    elif row.get("high_value_flag", 0) == 1:
        return "Escalate to account manager"
    else:
        return "Standard retention offer"

df_filtered["recommended_action"] = df_filtered.apply(assign_intervention, axis=1)

high_risk = df_filtered[df_filtered["predicted_churn"] == 1]

# ── KPI metrics ───────────────────────────────────────────────────────────────
total          = len(df_filtered)
n_high_risk    = len(high_risk)
churn_rate     = df_filtered["churn_probability"].mean()
avg_monthly    = df_raw.loc[mask, "MonthlyCharges"].mean()
revenue_at_risk = (
    df_raw.loc[mask & (preds[mask.values] == 1), "MonthlyCharges"].sum()
)

# ── Layout ────────────────────────────────────────────────────────────────────
st.title("Customer Churn Risk Dashboard")
st.caption(f"Model: **{selected_model}** · Risk threshold: **{risk_threshold:.0%}** · "
           f"Showing **{total:,}** customers after filters")
st.markdown("---")

# ── Row 1: KPI cards ─────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)

k1.metric(
    label="Total customers",
    value=f"{total:,}",
)
k2.metric(
    label="High-risk customers",
    value=f"{n_high_risk:,}",
    delta=f"{n_high_risk/total*100:.1f}% of segment",
    delta_color="inverse",
)
k3.metric(
    label="Avg predicted churn probability",
    value=f"{churn_rate:.1%}",
)
k4.metric(
    label="Monthly revenue at risk",
    value=f"${revenue_at_risk:,.0f}",
    delta="from high-risk customers",
    delta_color="inverse",
)

st.markdown("---")

# ── Row 2: Churn by segment charts ───────────────────────────────────────────
st.subheader("Churn risk by segment")
col1, col2, col3 = st.columns(3)

with col1:
    contract_churn = (
        df_filtered.groupby("Contract")["churn_probability"]
        .mean()
        .sort_values(ascending=True)
    )
    fig, ax = plt.subplots(figsize=(4, 3))
    bars = ax.barh(contract_churn.index, contract_churn.values,
                   color=["#1D9E75", "#EF9F27", "#D85A30"][:len(contract_churn)],
                   edgecolor="none")
    for bar, val in zip(bars, contract_churn.values):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                f"{val:.0%}", va="center", fontsize=9)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.set_title("By contract type", fontsize=11)
    ax.set_xlim(0, contract_churn.max() * 1.25)
    sns.despine()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col2:
    internet_churn = (
        df_filtered.groupby("InternetService")["churn_probability"]
        .mean()
        .sort_values(ascending=True)
    )
    fig, ax = plt.subplots(figsize=(4, 3))
    bars = ax.barh(internet_churn.index, internet_churn.values,
                   color=["#1D9E75", "#EF9F27", "#D85A30"][:len(internet_churn)],
                   edgecolor="none")
    for bar, val in zip(bars, internet_churn.values):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                f"{val:.0%}", va="center", fontsize=9)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.set_title("By internet service", fontsize=11)
    ax.set_xlim(0, internet_churn.max() * 1.25)
    sns.despine()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col3:
    # Tenure bucket churn — reconstruct from engineered flags
    def tenure_label(row):
        if row["is_new_customer"] == 1:
            return "0–12 months"
        elif row["is_loyal_customer"] == 1:
            return "49–72 months"
        elif row["tenure"] <= 24:
            return "13–24 months"
        else:
            return "25–48 months"

    df_filtered["tenure_label"] = df_filtered.apply(tenure_label, axis=1)
    tenure_order = ["0–12 months", "13–24 months", "25–48 months", "49–72 months"]
    tenure_churn = (
        df_filtered.groupby("tenure_label")["churn_probability"]
        .mean()
        .reindex(tenure_order)
        .dropna()
    )
    fig, ax = plt.subplots(figsize=(4, 3))
    colors = ["#D85A30", "#EF9F27", "#1D9E75", "#085041"][:len(tenure_churn)]
    bars = ax.barh(tenure_churn.index, tenure_churn.values,
                   color=colors, edgecolor="none")
    for bar, val in zip(bars, tenure_churn.values):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                f"{val:.0%}", va="center", fontsize=9)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.set_title("By tenure bucket", fontsize=11)
    ax.set_xlim(0, tenure_churn.max() * 1.25)
    sns.despine()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

st.markdown("---")

# ── Row 3: Feature importance & SHAP ─────────────────────────────────────────
st.subheader("Top churn drivers")
col_imp, col_shap = st.columns(2)

with col_imp:
    st.markdown("**Feature importance (XGBoost gain)**")
    xgb_model = models["XGBoost"]
    importances = pd.Series(
        xgb_model.feature_importances_,
        index=feat_cols
    ).sort_values(ascending=True).tail(12)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.barh(importances.index, importances.values,
            color="#534AB7", alpha=0.85, edgecolor="none")
    ax.set_xlabel("Importance (gain)")
    ax.set_title("Top 12 features", fontsize=11)
    sns.despine()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col_shap:
    st.markdown("**SHAP summary — direction of impact**")
    with st.spinner("Computing SHAP values..."):
        explainer   = shap.TreeExplainer(models["XGBoost"])
        sample      = X[feat_cols].sample(min(300, len(X)), random_state=42)
        shap_values = explainer.shap_values(sample)

    fig, ax = plt.subplots(figsize=(5, 5))
    shap.summary_plot(
        shap_values, sample,
        max_display=12,
        show=False,
        plot_size=None,
    )
    plt.title("SHAP values (sample of 300)", fontsize=11)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

st.markdown("---")

# ── Row 4: Intervention table ─────────────────────────────────────────────────
st.subheader(f"High-risk customers — {n_high_risk:,} flagged for intervention")

intervention_cols = [
    "customerID", "churn_probability", "Contract", "InternetService",
    "tenure", "MonthlyCharges", "contract_risk_score",
    "is_new_customer", "high_value_flag", "fiber_no_protection",
    "recommended_action",
]
# Add MonthlyCharges from raw
df_filtered["MonthlyCharges"] = df_raw.loc[mask, "MonthlyCharges"].values
df_filtered["tenure"]         = df_raw.loc[mask, "tenure"].values

action_filter = st.multiselect(
    "Filter by recommended action",
    options=df_filtered["recommended_action"].unique().tolist(),
    default=df_filtered["recommended_action"].unique().tolist(),
)

table_df = (
    high_risk[high_risk["recommended_action"].isin(action_filter)]
    [intervention_cols]
    .sort_values("churn_probability", ascending=False)
    .reset_index(drop=True)
)
table_df["churn_probability"] = table_df["churn_probability"].map("{:.1%}".format)

st.dataframe(
    table_df,
    use_container_width=True,
    height=400,
    column_config={
        "churn_probability": st.column_config.TextColumn("Churn risk"),
        "recommended_action": st.column_config.TextColumn("Recommended action"),
        "contract_risk_score": st.column_config.NumberColumn("Contract risk (0–2)"),
        "is_new_customer": st.column_config.CheckboxColumn("New customer"),
        "high_value_flag": st.column_config.CheckboxColumn("High value"),
        "fiber_no_protection": st.column_config.CheckboxColumn("Fiber/no protection"),
    }
)

col_dl1, col_dl2 = st.columns([1, 5])
with col_dl1:
    csv = table_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Export CSV",
        data=csv,
        file_name="high_risk_customers.csv",
        mime="text/csv",
    )

st.markdown("---")

# ── Row 5: Model performance summary ─────────────────────────────────────────
with st.expander("Model performance summary"):
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score

    perf_rows = []
    for name, m in models.items():
        p = m.predict_proba(X)[:, 1]
        pred = (p >= risk_threshold).astype(int)
        perf_rows.append({
            "Model":     name,
            "ROC-AUC":   round(roc_auc_score(y, p), 3),
            "PR-AUC":    round(average_precision_score(y, p), 3),
            "Precision": round(precision_score(y, pred, zero_division=0), 3),
            "Recall":    round(recall_score(y, pred), 3),
            "F1":        round(f1_score(y, pred), 3),
        })

    perf_df = pd.DataFrame(perf_rows).set_index("Model")
    st.dataframe(perf_df, use_container_width=True)
    st.caption(f"Evaluated on full dataset · threshold = {risk_threshold:.0%}")