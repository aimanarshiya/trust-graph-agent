"""
main.py
---------
Runs the full Trust Graph pipeline end-to-end, in order:
graph -> risk scoring -> DB logging -> explain -> remediate -> self-check
Then demonstrates filing + deciding one appeal.

This is your local test harness AND your demo script skeleton.
"""

import pandas as pd
from database import init_all, log_case, get_cases_needing_review, file_appeal, decide_appeal, get_audit_trail
from risk_scorer import compute_final_risk
from agents.explainer_agent import run_for_all_pending as run_explainer
from agents.remediation_agent import run_for_all_pending as run_remediation
from agents.selfcheck_agent import run_for_all_pending as run_selfcheck


def run_pipeline():
    print("=== 1. Initializing DB ===")
    init_all()

    print("\n=== 2. Computing risk scores ===")
    txns = pd.read_csv("data/transactions.csv")
    dels = pd.read_csv("data/deliveries.csv")
    result = compute_final_risk(txns, dels)

    for _, row in result.iterrows():
        log_case(row["seller_id"], row["final_risk_score"], row["tier"], bool(row["needs_agent_review"]))
    print(f"Logged {len(result)} cases.")

    print("\n=== 3. Running explainer agent ===")
    run_explainer()

    print("\n=== 4. Running remediation agent ===")
    run_remediation()

    print("\n=== 5. Running self-check agent ===")
    run_selfcheck()

    print("\n=== 6. Demo: filing an appeal on the first flagged case ===")
    flagged = get_cases_needing_review()
    if flagged:
        case_id = flagged[0]["case_id"]
        seller_id = flagged[0]["seller_id"]
        appeal_id = file_appeal(case_id, seller_id, "I believe this flag is a mistake -- my account is legitimate.")
        print(f"Appeal {appeal_id} filed for case {case_id}.")

        decide_appeal(appeal_id, decision="overturned", decided_by="demo_admin")
        print(f"Appeal {appeal_id} decided: overturned.")

        print(f"\n=== Full audit trail for case {case_id} ===")
        for entry in get_audit_trail(case_id):
            print(f"[{entry['timestamp']}] {entry['event_type']}: {entry['details']}")


if __name__ == "__main__":
    run_pipeline()