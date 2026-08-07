"""
risk_scorer.py
----------------
Combines graph_engine.py's collusion score, classical RULE-BASED
signals, and the trained IEEE-CIS ML model's fraud probability into a
single final risk score per seller.

This is also where the "save expensive reasoning for high-stakes
decisions" principle is actually implemented: needs_agent_review()
decides whether a case is worth an LLM call at all.
"""

import pandas as pd
from graph_engine import graph_anomaly_scores
from ml_scorer import compute_ml_signals, get_model_metrics


def compute_rule_signals(transactions: pd.DataFrame, deliveries: pd.DataFrame) -> pd.DataFrame:
    """
    Classical, explainable rule-based signals per seller -- each one
    maps directly to a real fraud pattern mentioned in the problem
    statement (self-ordering, delivery fraud, return abuse).
    """
    results = []

    merged = transactions.merge(deliveries, on="transaction_id", how="left")

    for seller_id, group in merged.groupby("seller_id"):
        num_txns = len(group)

        # Return abuse: unusually high return rate vs. the overall average
        return_rate = group["is_return"].astype(bool).mean()

        # Delivery fraud signal: proof-of-delivery missing despite "delivered" status
        delivered = group[group["status"] == "delivered"]
        missing_proof_rate = (
            (~delivered["proof_photo_provided"].astype(bool)).mean()
            if len(delivered) > 0 else 0
        )

        # Velocity signal: many transactions in a very short seller lifespan
        txn_velocity = num_txns  # simplified for hackathon scope

        results.append({
            "seller_id": seller_id,
            "num_txns": num_txns,
            "return_rate": round(return_rate, 3),
            "missing_proof_rate": round(missing_proof_rate, 3),
            "txn_velocity": txn_velocity,
        })

    return pd.DataFrame(results)


def compute_final_risk(transactions: pd.DataFrame, deliveries: pd.DataFrame) -> pd.DataFrame:
    """
    Main entry point: blends graph_risk_score (collusion), rule
    signals (return abuse, delivery fraud), and the trained ML
    model's fraud probability into one explainable score, then
    classifies each case into an action tier.
    """
    graph_scores = graph_anomaly_scores(transactions)
    rule_scores = compute_rule_signals(transactions, deliveries)
    ml_scores = compute_ml_signals(transactions)

    combined = graph_scores.merge(rule_scores, on="seller_id", how="left")
    combined = combined.merge(ml_scores, on="seller_id", how="left")
    combined = combined.fillna(0)

    # Weights re-balanced to fold in the ML signal. Graph stays the
    # primary signal (this track's named core focus); ML and rules
    # act as corroborating evidence.
    combined["final_risk_score"] = (
        0.45 * combined["graph_risk_score"] +
        0.25 * combined["ml_fraud_score"] +
        0.15 * combined["return_rate"] +
        0.15 * combined["missing_proof_rate"]
    ).round(3)

    combined["tier"] = combined["final_risk_score"].apply(classify_tier)
    combined["needs_agent_review"] = combined["final_risk_score"] >= 0.4

    return combined.sort_values("final_risk_score", ascending=False)


def classify_tier(score: float) -> str:
    """
    Graduated response tiers, matching the problem statement's
    'soft interventions before hard ones' requirement.
    """
    if score >= 0.8:
        return "hard_action_candidate"
    elif score >= 0.4:
        return "soft_intervention"
    else:
        return "no_action"


if __name__ == "__main__":
    txns = pd.read_csv("data/transactions.csv")
    dels = pd.read_csv("data/deliveries.csv")

    metrics = get_model_metrics()
    print(f"ML model (IEEE-CIS validated) -- Precision: {metrics['precision']:.3f}, "
          f"Recall: {metrics['recall']:.3f}, AUC: {metrics['auc']:.3f}\n")

    result = compute_final_risk(txns, dels)
    print(result[["seller_id", "final_risk_score", "tier", "needs_agent_review"]].head(10).to_string(index=False))