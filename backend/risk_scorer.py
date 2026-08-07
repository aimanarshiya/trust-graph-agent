"""
risk_scorer.py
---------------
Turns the raw evidence dictionary from graph_engine.py into:
  1. A risk_score (0-100) -- a simple, explainable weighted sum, NOT a
     trained black-box model. Every point is traceable to a specific piece
     of evidence, which is what the audit-trail requirement demands.
  2. A risk_tier -- low / medium / high -- which decides what happens next.

Why weighted rules instead of a trained classifier here:
The problem statement's guardrail is "hard automated actions need >=95%
precision, otherwise a human reviews." A transparent rule-based score makes
it trivial to prove precision on paper (you can point to the exact rule
that fired). A trained black-box model would need much more labeled data
and validation than a hackathon timeline allows, and would be harder to
explain to a judge or an accused seller. This is also why the engine below
routes almost nothing straight to a "hard action" -- it prefers pushing
borderline/high cases to a human-reviewable queue.

This file has NO LLM calls either -- scoring is still cheap/classical.
The LLM only gets involved after this, and only for medium/high tier cases,
via agents/explainer_agent.py and agents/remediation_agent.py.
"""

from graph_engine import load_data, compute_risk_signals


# Weights are intentionally simple integers so every score is easy to
# hand-verify: "this case scored 55 because ring_membership (+40) and
# gps_mismatch (+15)." No hidden math.
WEIGHTS = {
    "ring_membership": 40,       # part of a repeated seller-customer-delivery triad
    "gps_mismatch": 15,          # delivery GPS didn't match the drop-off
    "no_proof_photo": 15,        # no proof-of-delivery photo captured
    "high_return_rate": 20,      # seller's return rate is unusually high
    "ring_repeat_bonus": 5,      # extra point per repeat beyond the 3rd, capped
}

RETURN_RATE_THRESHOLD = 0.3   # 30%+ returns is flagged as unusually high
MAX_RING_BONUS = 20            # cap so one extreme ring doesn't dominate the score

# Tier boundaries
LOW_MAX = 30
MEDIUM_MAX = 65
# anything above MEDIUM_MAX is "high"

# Guardrail from the problem statement: hard automated actions (suspension,
# payout freeze) require >=95% precision. We don't have a validated
# precision number for a rule-based score built in one night, so we NEVER
# let this engine trigger a hard action by itself -- "high" tier means
# "send to remediation_agent, which can only suggest a hard action, and
# even then it must go through a human-reviewable appeal step."
ALLOW_AUTOMATED_HARD_ACTION = False


def score_transaction(signal: dict) -> dict:
    """Compute a risk score + tier for one transaction's signal dict."""
    score = 0
    reasons = []

    if signal["ring_membership"]:
        score += WEIGHTS["ring_membership"]
        reasons.append(
            f"Part of a repeated seller-customer-delivery pattern "
            f"(seen {signal['ring_repeat_count']} times)"
        )
        bonus = min((signal["ring_repeat_count"] - 3) * WEIGHTS["ring_repeat_bonus"], MAX_RING_BONUS)
        if bonus > 0:
            score += bonus
            reasons.append(f"Ring repeats unusually often (+{bonus} bonus)")

    if signal["gps_mismatch"]:
        score += WEIGHTS["gps_mismatch"]
        reasons.append("Delivery GPS location did not match the claimed drop-off")

    if signal["no_proof_photo"]:
        score += WEIGHTS["no_proof_photo"]
        reasons.append("No proof-of-delivery photo was captured")

    if signal["seller_return_rate"] >= RETURN_RATE_THRESHOLD:
        score += WEIGHTS["high_return_rate"]
        reasons.append(f"Seller's return rate is unusually high ({signal['seller_return_rate']*100:.0f}%)")

    score = min(score, 100)

    if score <= LOW_MAX:
        tier = "low"
    elif score <= MEDIUM_MAX:
        tier = "medium"
    else:
        tier = "high"

    return {
        **signal,
        "risk_score": score,
        "risk_tier": tier,
        "reasons": reasons,
        # Guardrail flag: even a "high" tier case can only get a soft action
        # automatically. Hard actions always require human review.
        "eligible_for_automated_hard_action": False if not ALLOW_AUTOMATED_HARD_ACTION else (score > MEDIUM_MAX),
    }


def score_all(data_dir="data"):
    """Score every transaction in the dataset. Returns a list of scored dicts."""
    transactions, sellers, deliveries = load_data(data_dir)
    signals = compute_risk_signals(transactions, sellers, deliveries)
    return [score_transaction(s) for s in signals]


if __name__ == "__main__":
    # Run with: python backend/risk_scorer.py  (from project root)
    scored = score_all()

    tier_counts = {"low": 0, "medium": 0, "high": 0}
    for s in scored:
        tier_counts[s["risk_tier"]] += 1

    print(f"Scored {len(scored)} transactions")
    print(f"Tier breakdown: {tier_counts}\n")

    print("Cases needing attention (medium/high only):")
    for s in scored:
        if s["risk_tier"] != "low":
            print(f"\n  {s['transaction_id']} | seller {s['seller_id']} | score {s['risk_score']} | tier {s['risk_tier']}")
            for r in s["reasons"]:
                print(f"    - {r}")