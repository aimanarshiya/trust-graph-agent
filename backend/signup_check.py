"""
signup_check.py
------------------
Simulates the signup-time identity check: flag disposable/throwaway
emails at registration as an identity-risk signal.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "external"))
import pandas as pd
from external.debounce_client import check_email


def check_seller_signups(sellers: pd.DataFrame, email_col: str = "signup_email") -> pd.DataFrame:
    if email_col not in sellers.columns:
        print(f"No '{email_col}' column found -- skipping disposable-email check.")
        sellers["disposable_email_flag"] = False
        return sellers

    results = []
    for email in sellers[email_col]:
        result = check_email(email)
        results.append(result["disposable"])

    sellers["disposable_email_flag"] = results
    return sellers


if __name__ == "__main__":
    sellers = pd.read_csv("data/sellers.csv")
    sellers = check_seller_signups(sellers)
    print(sellers[["seller_id", "disposable_email_flag"]].to_string(index=False))
    print(f"\nFlagged {sellers['disposable_email_flag'].sum()} disposable-email signups out of {len(sellers)}.")