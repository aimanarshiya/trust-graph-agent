"""
ml_scorer.py
----------------
Wraps the trained IEEE-CIS XGBoost fraud model so it can plug into
risk_scorer.py's final_risk_score. Reports real precision/recall from
training instead of a hardcoded placeholder.

Honest design note: the model was trained on IEEE-CIS transaction
columns. If your live transactions.csv doesn't share those columns,
this module still exposes MODEL_METRICS (the real, validated
precision/recall/AUC) for your writeup, and returns a neutral
per-seller ML score of 0 rather than faking a prediction.
"""

import os
import pandas as pd
import numpy as np
import joblib

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "ml", "artifacts")

_model = None
_feature_columns = None
_metrics = None


def _load_artifacts():
    global _model, _feature_columns, _metrics
    if _model is None:
        _model = joblib.load(os.path.join(ARTIFACTS_DIR, "fraud_model.joblib"))
        _feature_columns = joblib.load(os.path.join(ARTIFACTS_DIR, "feature_columns.joblib"))
        _metrics = joblib.load(os.path.join(ARTIFACTS_DIR, "model_metrics.joblib"))
    return _model, _feature_columns, _metrics


def get_model_metrics() -> dict:
    """Real precision/recall/AUC from the IEEE-CIS holdout set."""
    _, _, metrics = _load_artifacts()
    return metrics


def compute_ml_signals(transactions: pd.DataFrame) -> pd.DataFrame:
    """
    Scores each seller's transactions with the trained model, if the
    transaction schema overlaps enough with the IEEE-CIS training
    columns to be meaningful. Otherwise returns 0 (no fabricated score).

    Returns a DataFrame: seller_id, ml_fraud_score (mean predicted
    fraud probability across that seller's transactions, 0-1).
    """
    model, feature_columns, _ = _load_artifacts()

    available_cols = [c for c in feature_columns if c in transactions.columns]
    overlap_ratio = len(available_cols) / len(feature_columns)

    if overlap_ratio < 0.3 or "seller_id" not in transactions.columns:
        # Not enough real signal to trust a prediction -- return neutral scores
        sellers = transactions["seller_id"].unique() if "seller_id" in transactions.columns else []
        return pd.DataFrame({"seller_id": sellers, "ml_fraud_score": 0.0})

    # Build a frame matching the model's expected columns, filling gaps with 0
    X = pd.DataFrame(0, index=transactions.index, columns=feature_columns)
    X[available_cols] = transactions[available_cols]

    # Encode any leftover categoricals the same way training did
    for col in X.select_dtypes(include=["object", "str"]).columns:
        X[col] = X[col].astype("category").cat.codes

    proba = model.predict_proba(X)[:, 1]
    transactions = transactions.copy()
    transactions["_ml_fraud_proba"] = proba

    per_seller = (
        transactions.groupby("seller_id")["_ml_fraud_proba"]
        .mean()
        .reset_index()
        .rename(columns={"_ml_fraud_proba": "ml_fraud_score"})
    )
    return per_seller