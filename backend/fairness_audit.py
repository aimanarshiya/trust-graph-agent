"""
fairness_audit.py
--------------------
Required guardrail: report whether the system disproportionately
flags small/new sellers vs large/established ones. No LLM involved --
pure pandas aggregation, so this costs zero API quota.
"""

import pandas as pd
from risk_scorer import compute_final_risk


def fairness_report(transactions: pd.DataFrame, deliveries: pd.DataFrame, sellers: pd.DataFrame) -> pd.DataFrame:
    risk = compute_final_risk(transactions, deliveries)
    merged = risk.merge(sellers, on="seller_id", how="left")

    def tenure_bucket(days):
        if pd.isna(days):
            return "unknown"
        return "new (<90d)" if days < 90 else "established (90d+)"

    merged["tenure_bucket"] = merged["tenure_days"].apply(tenure_bucket) if "tenure_days" in merged else "unknown"

    size_col = "size" if "size" in merged.columns else None

    report_rows = []

    for bucket, group in merged.groupby("tenure_bucket"):
        flagged = group[group["needs_agent_review"] == True]
        report_rows.append({
            "cohort_type": "tenure",
            "cohort": bucket,
            "total_sellers": len(group),
            "flagged_sellers": len(flagged),
            "flag_rate": round(len(flagged) / max(len(group), 1), 3),
        })

    if size_col:
        for size_val, group in merged.groupby(size_col):
            flagged = group[group["needs_agent_review"] == True]
            report_rows.append({
                "cohort_type": "seller_size",
                "cohort": size_val,
                "total_sellers": len(group),
                "flagged_sellers": len(flagged),
                "flag_rate": round(len(flagged) / max(len(group), 1), 3),
            })

    return pd.DataFrame(report_rows)


if __name__ == "__main__":
    txns = pd.read_csv("data/transactions.csv")
    dels = pd.read_csv("data/deliveries.csv")
    sellers = pd.read_csv("data/sellers.csv")

    report = fairness_report(txns, dels, sellers)
    print(report.to_string(index=False))

    max_rate = report["flag_rate"].max()
    min_rate = report["flag_rate"].min()
    print(f"\nFlag-rate spread: {min_rate} to {max_rate}")
    if max_rate - min_rate > 0.3:
        print("WARNING: significant disparity between cohorts -- worth investigating before demo.")
    else:
        print("Flag rates are reasonably balanced across cohorts.")