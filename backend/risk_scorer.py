"""
Risk Scorer
-----------
Combines:
1. Graph-based collusion/anomaly score
2. Classical rule-based fraud signals
3. Trained IEEE-CIS ML model fraud probability

into a single final risk score per seller.

The final score is then classified into:
- no_action
- soft_intervention
- hard_action_candidate

The `needs_agent_review` flag determines whether a case
is important enough to be passed to the AI/LLM agent for
more expensive reasoning.
"""

import pandas as pd
from pathlib import Path

from graph_engine import graph_anomaly_scores
from ml_scorer import compute_ml_signals, get_model_metrics


# ============================================================
# PATH CONFIGURATION
# ============================================================
# risk_scorer.py is inside:
#
# trust-graph-agent/
# └── backend/
#     ├── risk_scorer.py
#     └── data/
#
# Using __file__ makes the paths work regardless of where
# the script is executed from.

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


# ============================================================
# RULE-BASED RISK SIGNALS
# ============================================================

def compute_rule_signals(
    transactions: pd.DataFrame,
    deliveries: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute classical, explainable rule-based fraud signals
    for every seller.

    Signals:
    - Return abuse
    - Missing proof-of-delivery
    - Transaction velocity

    These rules represent understandable fraud patterns and
    complement the graph and ML signals.
    """

    results = []

    # --------------------------------------------------------
    # Merge transactions with delivery information
    # --------------------------------------------------------
    merged = transactions.merge(
        deliveries,
        on="transaction_id",
        how="left"
    )

    # --------------------------------------------------------
    # Calculate signals for every seller
    # --------------------------------------------------------
    for seller_id, group in merged.groupby("seller_id"):

        num_txns = len(group)

        # ----------------------------------------------------
        # Return abuse
        # ----------------------------------------------------
        # High return percentage can indicate return abuse.
        return_rate = (
            group["is_return"]
            .astype(bool)
            .mean()
        )

        # ----------------------------------------------------
        # Delivery fraud
        # ----------------------------------------------------
        # Look only at delivered transactions.
        delivered = group[
            group["status"] == "delivered"
        ]

        if len(delivered) > 0:
            missing_proof_rate = (
                ~delivered["proof_photo_provided"]
                .astype(bool)
            ).mean()
        else:
            missing_proof_rate = 0

        # ----------------------------------------------------
        # Transaction velocity
        # ----------------------------------------------------
        # Simplified for hackathon scope.
        txn_velocity = num_txns

        results.append({
            "seller_id": seller_id,
            "num_txns": num_txns,
            "return_rate": round(return_rate, 3),
            "missing_proof_rate": round(
                missing_proof_rate,
                3
            ),
            "txn_velocity": txn_velocity,
        })

    return pd.DataFrame(results)


# ============================================================
# FINAL RISK SCORE
# ============================================================

def compute_final_risk(
    transactions: pd.DataFrame,
    deliveries: pd.DataFrame
) -> pd.DataFrame:
    """
    Main risk-scoring pipeline.

    Combines:

        45% Graph risk
        25% ML fraud probability
        15% Return abuse
        15% Missing delivery proof

    into one final seller-level risk score.

    The result is also classified into an action tier.
    """

    # --------------------------------------------------------
    # 1. Graph-based risk
    # --------------------------------------------------------
    graph_scores = graph_anomaly_scores(
        transactions
    )

    # --------------------------------------------------------
    # 2. Rule-based risk
    # --------------------------------------------------------
    rule_scores = compute_rule_signals(
        transactions,
        deliveries
    )

    # --------------------------------------------------------
    # 3. ML-based fraud probability
    # --------------------------------------------------------
    ml_scores = compute_ml_signals(
        transactions
    )

    # --------------------------------------------------------
    # 4. Combine all signals
    # --------------------------------------------------------
    combined = graph_scores.merge(
        rule_scores,
        on="seller_id",
        how="left"
    )

    combined = combined.merge(
        ml_scores,
        on="seller_id",
        how="left"
    )

    # Missing values become zero.
    combined = combined.fillna(0)

    # --------------------------------------------------------
    # 5. Calculate final risk score
    # --------------------------------------------------------
    combined["final_risk_score"] = (
        0.45 * combined["graph_risk_score"]
        + 0.25 * combined["ml_fraud_score"]
        + 0.15 * combined["return_rate"]
        + 0.15 * combined["missing_proof_rate"]
    ).round(3)

    # --------------------------------------------------------
    # 6. Classify risk tier
    # --------------------------------------------------------
    combined["tier"] = combined[
        "final_risk_score"
    ].apply(classify_tier)

    # --------------------------------------------------------
    # 7. Decide whether AI agent review is required
    # --------------------------------------------------------
    # Expensive LLM reasoning is only triggered for
    # sufficiently high-risk cases.
    combined["needs_agent_review"] = (
        combined["final_risk_score"] >= 0.4
    )

    # --------------------------------------------------------
    # 8. Highest-risk sellers first
    # --------------------------------------------------------
    return combined.sort_values(
        "final_risk_score",
        ascending=False
    )


# ============================================================
# RISK TIER CLASSIFICATION
# ============================================================

def classify_tier(score: float) -> str:
    """
    Convert the numerical risk score into an action tier.

    >= 0.80
        Hard action candidate

    >= 0.40
        Soft intervention

    < 0.40
        No action
    """

    if score >= 0.8:
        return "hard_action_candidate"

    elif score >= 0.4:
        return "soft_intervention"

    else:
        return "no_action"


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("TRUST-GRAPH RISK SCORING PIPELINE")
    print("=" * 70)

    # --------------------------------------------------------
    # Load datasets using absolute paths based on this file.
    #
    # This prevents the previous problem where:
    #
    # data/transactions.csv
    #
    # accidentally loaded:
    #
    # trust-graph-agent/data/transactions.csv
    #
    # instead of:
    #
    # backend/data/transactions.csv
    # --------------------------------------------------------

    transactions_path = DATA_DIR / "transactions.csv"
    deliveries_path = DATA_DIR / "deliveries.csv"

    print(f"\nTransaction dataset: {transactions_path}")
    print(f"Delivery dataset:    {deliveries_path}")

    # --------------------------------------------------------
    # Load transactions
    # --------------------------------------------------------

    txns = pd.read_csv(
        transactions_path
    )

    # --------------------------------------------------------
    # Load deliveries
    # --------------------------------------------------------

    dels = pd.read_csv(
        deliveries_path
    )

    # --------------------------------------------------------
    # Validate required transaction columns
    # --------------------------------------------------------

    required_transaction_columns = {
        "transaction_id",
        "customer_id",
        "seller_id",
        "delivery_partner_id",
        "amount",
        "device_id",
        "ip_address",
        "is_return",
        "is_fraud_label",
    }

    missing_transaction_columns = (
        required_transaction_columns
        - set(txns.columns)
    )

    if missing_transaction_columns:

        raise ValueError(
            "Transaction dataset is missing required "
            f"columns: {sorted(missing_transaction_columns)}"
        )

    # --------------------------------------------------------
    # Validate required delivery columns
    # --------------------------------------------------------

    required_delivery_columns = {
        "transaction_id",
        "status",
        "proof_photo_provided",
    }

    missing_delivery_columns = (
        required_delivery_columns
        - set(dels.columns)
    )

    if missing_delivery_columns:

        raise ValueError(
            "Delivery dataset is missing required "
            f"columns: {sorted(missing_delivery_columns)}"
        )

    # --------------------------------------------------------
    # Print dataset information
    # --------------------------------------------------------

    print("\nDataset validation successful.")

    print(
        f"Transactions loaded: {len(txns):,}"
    )

    print(
        f"Deliveries loaded:   {len(dels):,}"
    )

    print(
        "\nTransaction columns:"
    )

    print(
        txns.columns.tolist()
    )

    # --------------------------------------------------------
    # Get trained ML model metrics
    # --------------------------------------------------------

    metrics = get_model_metrics()

    print(
        "\nML model (IEEE-CIS validated) -- "
        f"Precision: {metrics['precision']:.3f}, "
        f"Recall: {metrics['recall']:.3f}, "
        f"AUC: {metrics['auc']:.3f}"
    )

    # --------------------------------------------------------
    # Compute final risk
    # --------------------------------------------------------

    result = compute_final_risk(
        txns,
        dels
    )

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TOP SELLER RISK RESULTS")
    print("=" * 70)

    columns_to_display = [
        "seller_id",
        "final_risk_score",
        "tier",
        "needs_agent_review",
    ]

    print(
        result[
            columns_to_display
        ]
        .head(10)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("RISK SUMMARY")
    print("=" * 70)

    total_sellers = len(result)

    agent_review_count = int(
        result["needs_agent_review"].sum()
    )

    hard_action_count = int(
        (
            result["tier"]
            == "hard_action_candidate"
        ).sum()
    )

    soft_intervention_count = int(
        (
            result["tier"]
            == "soft_intervention"
        ).sum()
    )

    no_action_count = int(
        (
            result["tier"]
            == "no_action"
        ).sum()
    )

    print(
        f"Total sellers analyzed: {total_sellers}"
    )

    print(
        f"Agent review required:  {agent_review_count}"
    )

    print(
        f"Hard action candidates: {hard_action_count}"
    )

    print(
        f"Soft interventions:     {soft_intervention_count}"
    )

    print(
        f"No action:              {no_action_count}"
    )

    print("\nRisk scoring completed successfully.")