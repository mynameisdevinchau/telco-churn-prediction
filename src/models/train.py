"""
src/models/train.py

Training script — reads config.yaml, loads features, trains all three
models, evaluates, and saves artifacts to models/.

Run directly: python -m src.models.train
"""

import os
import yaml
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score,
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def encode(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Drop redundant raw columns and one-hot encode remaining categoricals."""
    raw_service_cols = cfg["features"]["raw_service_cols"]
    cat_cols         = cfg["features"]["categorical_cols"]

    df = df.drop(columns=[c for c in raw_service_cols if c in df.columns])
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)
    return df


def get_models(cfg: dict) -> dict:
    lr_cfg  = cfg["model"]["logistic_regression"]
    rf_cfg  = cfg["model"]["random_forest"]
    xgb_cfg = cfg["model"]["xgboost"]

    return {
        "logistic_churn": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                C=lr_cfg["C"],
                max_iter=lr_cfg["max_iter"],
                random_state=cfg["model"]["random_state"],
            )),
        ]),
        "random_forest_churn": RandomForestClassifier(
            n_estimators=rf_cfg["n_estimators"],
            max_depth=rf_cfg["max_depth"],
            min_samples_leaf=rf_cfg["min_samples_leaf"],
            random_state=cfg["model"]["random_state"],
            n_jobs=-1,
        ),
        "xgboost_churn": XGBClassifier(
            n_estimators=xgb_cfg["n_estimators"],
            max_depth=xgb_cfg["max_depth"],
            learning_rate=xgb_cfg["learning_rate"],
            subsample=xgb_cfg["subsample"],
            colsample_bytree=xgb_cfg["colsample_bytree"],
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=cfg["model"]["random_state"],
            n_jobs=-1,
        ),
    }


def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series,
             threshold: float) -> dict:
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc":   round(roc_auc_score(y_test, y_prob), 4),
        "pr_auc":    round(average_precision_score(y_test, y_prob), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred), 4),
        "f1":        round(f1_score(y_test, y_pred), 4),
    }


def run(config_path: str = "config.yaml") -> None:
    cfg = load_config(config_path)
    mcfg = cfg["model"]

    # ── Load features ─────────────────────────────────────────────────────────
    print("Loading features...")
    df = pd.read_parquet(cfg["data"]["features_path"])
    df = encode(df, cfg)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # ── Split ─────────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=mcfg["test_size"],
        random_state=mcfg["random_state"],
        stratify=y,
    )
    print(f"Train: {X_train.shape[0]:,}  Test: {X_test.shape[0]:,}")

    # ── SMOTE ─────────────────────────────────────────────────────────────────
    if mcfg["smote"]:
        smote = SMOTE(random_state=mcfg["random_state"])
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print(f"After SMOTE: {X_train.shape[0]:,} rows  "
              f"churn rate: {y_train.mean()*100:.1f}%")

    # ── Train & evaluate ──────────────────────────────────────────────────────
    threshold = mcfg["threshold"]
    models    = get_models(cfg)
    results   = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test, threshold)
        results[name] = metrics
        print(f"  ROC-AUC={metrics['roc_auc']}  "
              f"Precision={metrics['precision']}  "
              f"Recall={metrics['recall']}  "
              f"F1={metrics['f1']}")

    # ── Save artifacts ────────────────────────────────────────────────────────
    out_dir = mcfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    for name, model in models.items():
        path = os.path.join(out_dir, f"{name}.pkl")
        joblib.dump(model, path)
        print(f"Saved {path}")

    # Save feature column order — required by dashboard
    pd.Series(X_train.columns.tolist()).to_csv(
        os.path.join(out_dir, "feature_columns.csv"), index=False
    )

    # Save results summary
    results_df = pd.DataFrame(results).T
    results_path = os.path.join(out_dir, "model_results.csv")
    results_df.to_csv(results_path)
    print(f"\nResults saved to {results_path}")
    print(results_df.to_string())


if __name__ == "__main__":
    run()