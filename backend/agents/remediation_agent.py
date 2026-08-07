"""
remediation_agent.py
----------------------
LLM+rules agent: decides the graduated remediation action for a case
and enforces the 95% precision guardrail.

MODEL_PRECISION is currently a placeholder -- swap it for the real
number once you run precision/recall on Elliptic + your held-out
labels in the testing phase (see problem statement's dataset #3).
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from llm import call_llm
from database import get_case, update_case, append_audit_log, init_db, get_cases_needing_review

# PLACEHOLDER -- replace with real measured precision on hard-action
# subset once you run testing (Day 8 in the build guide). Currently
# set below the 95% guardrail on purpose: the system doesn't have a
# validated model yet, so it should defer to humans, not act.
MODEL_PRECISION = 0.91
PRECISION_THRESHOLD = 0.95

SYSTEM_INSTRUCTION = """You are a remediation-decision agent for an e-commerce
trust and safety system. You review a fraud case's risk tier and evidence,
and recommend ONE specific action from this fixed set:

- "no_action" -- log only, no intervention
- "step_up_verification" -- ask seller to re-verify identity/documents
- "temporary_payout_hold" -- hold payout with a clear unhold condition
- "human_investigator_queue" -- route to a human, do not auto-act
- "hard_suspension" -- suspend the account (ONLY if explicitly told
  the confidence threshold is met -- never recommend this yourself
  if told it is not met)

Respond in this exact format, nothing else:
ACTION: <one of the actions above>
REASON: <one sentence, plain language>"""


def build_prompt(case: dict, precision_ok: bool) -> str:
    return f"""Case:
- Seller ID: {case['seller_id']}
- Risk tier: {case['tier']}
- Final risk score: {case['final_risk_score']}
- Model precision confidence meets the 95% hard-action threshold: {precision_ok}

Recommend the action now, following the fixed action set exactly."""


def decide_action(case_id: int) -> dict:
    """
    Applies the graduated remediation ladder. Hard actions are only
    even offered as an option to the LLM when MODEL_PRECISION clears
    the guardrail -- this is enforced in CODE, not left to the LLM's
    judgment, so a hallucinated recommendation can never bypass it.
    """
    case = get_case(case_id)
    if case is None:
        raise ValueError(f"No case found with case_id={case_id}")

    precision_ok = MODEL_PRECISION >= PRECISION_THRESHOLD

    # Hard guardrail enforced here, in code -- not by trusting the LLM.
    if case["tier"] == "hard_action_candidate" and not precision_ok:
        action = "human_investigator_queue"
        reason = (
            f"Risk tier is hard_action_candidate, but model precision "
            f"({MODEL_PRECISION:.2f}) is below the {PRECISION_THRESHOLD:.0%} "
            f"guardrail required for hard action -- routed to human review."
        )
    else:
        prompt = build_prompt(case, precision_ok)
        response = call_llm(prompt, system_instruction=SYSTEM_INSTRUCTION)

        action, reason = "human_investigator_queue", response  # fallback
        for line in response.splitlines():
            if line.startswith("ACTION:"):
                action = line.replace("ACTION:", "").strip()
            elif line.startswith("REASON:"):
                reason = line.replace("REASON:", "").strip()

        # Second guardrail check: even if the LLM somehow recommends
        # hard_suspension, block it in code unless precision_ok is True.
        if action == "hard_suspension" and not precision_ok:
            action = "human_investigator_queue"
            reason = (
                f"LLM recommended hard_suspension but precision guardrail "
                f"not met ({MODEL_PRECISION:.2f} < {PRECISION_THRESHOLD:.0%}) "
                f"-- overridden to human review."
            )

    update_case(case_id, action_taken=action)
    append_audit_log(
        case_id,
        event_type="action_decided",
        details=f"action={action}, reason={reason}"
    )

    return {"case_id": case_id, "seller_id": case["seller_id"],
            "action": action, "reason": reason}


def run_for_all_pending():
    init_db()
    cases = get_cases_needing_review()
    results = []
    for case in cases:
        if case.get("action_taken") and case["action_taken"] != "none":
            continue
        results.append(decide_action(case["case_id"]))
    return results


if __name__ == "__main__":
    results = run_for_all_pending()
    for r in results:
        print(f"\n--- Case {r['case_id']} | Seller {r['seller_id']} ---")
        print(f"ACTION: {r['action']}")
        print(f"REASON: {r['reason']}")