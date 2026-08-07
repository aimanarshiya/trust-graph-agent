"""
remediation_agent.py
----------------------
LLM agent: given a case's evidence + explanation, recommends a
GRADUATED action -- and enforces the problem statement's explicit
guardrail: hard actions (suspension, payout freeze) are NEVER
auto-applied. They only ever become a RECOMMENDATION for human
review, never an automatic action. Only soft actions can be
auto-applied, and even those are logged and reversible via appeal.
"""

import sys
import os
import time
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from llm import call_llm
from database import get_case, update_case, append_audit_log, get_cases_needing_review, init_db

SYSTEM_INSTRUCTION = """You are a remediation-recommendation agent for an
e-commerce trust and safety system.

Given a fraud case's evidence and tier, recommend ONE specific action
from this exact list only:
- "step_up_verification" (soft): ask seller to verify identity/documents
- "payout_hold" (soft): temporarily hold pending payouts pending review
- "reduced_visibility" (soft): lower search ranking pending review
- "recommend_suspension" (hard, REQUIRES human approval, never auto-applied)

Rules you MUST follow:
- If tier is "no_action", recommend "no_action_needed"
- If tier is "soft_intervention", you may ONLY recommend a soft action
- If tier is "hard_action_candidate", you may recommend
  "recommend_suspension" but MUST note it requires human review before
  taking effect -- you are recommending, not deciding
- Give a one-sentence justification tied to the actual evidence given
- Respond with ONLY valid JSON, nothing else, no markdown, no backticks:
{"action": "<action_name>", "justification": "<one sentence>", "requires_human_approval": true or false}"""


def build_prompt(case: dict) -> str:
    return f"""Case for Seller {case['seller_id']}:
- Tier: {case['tier']}
- Final risk score: {case['final_risk_score']}
- Graph collusion score: {case.get('graph_risk_score', 'N/A')}
- Return rate: {case.get('return_rate', 'N/A')}
- Missing delivery-proof rate: {case.get('missing_proof_rate', 'N/A')}
- Existing explanation on file: {case.get('explanation', 'none yet')}

Recommend the action now, in the exact required format."""

import json
import re

def parse_response(text: str) -> dict:
    """Parses the LLM's JSON reply. Falls back to safe defaults
    (human review) if parsing fails for any reason."""
    default = {"action": "recommend_manual_review", "justification": text.strip()[:200],
               "requires_human_approval": True}
    try:
        # strip markdown code fences if the model added them anyway
        cleaned = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
        parsed = json.loads(cleaned)
        return {
            "action": parsed.get("action", default["action"]),
            "justification": parsed.get("justification", default["justification"]),
            "requires_human_approval": bool(parsed.get("requires_human_approval", True)),
        }
    except Exception:
        return default

# Guardrail enforced IN CODE, not just by prompting -- this is what
# makes it a real guardrail rather than a suggestion the LLM could ignore
HARD_ACTIONS = {"recommend_suspension"}


def remediate_case(case_id: int) -> dict:
    case = get_case(case_id)
    if case is None:
        raise ValueError(f"No case found with case_id={case_id}")

    prompt = build_prompt(case)
    raw_response = call_llm(prompt, system_instruction=SYSTEM_INSTRUCTION)
    parsed = parse_response(raw_response)

    # HARD GUARDRAIL: code-level enforcement, independent of what the
    # LLM said. No hard action is EVER auto-applied, no matter what.
    if parsed["action"] in HARD_ACTIONS:
        parsed["requires_human_approval"] = True
        action_taken = "pending_human_review"
    else:
        action_taken = parsed["action"]

    update_case(case_id, action_taken=action_taken)
    append_audit_log(
        case_id,
        event_type="remediation_recommended",
        details=f"action={parsed['action']}, requires_approval={parsed['requires_human_approval']}, "
                f"justification={parsed['justification']}"
    )

    return {"case_id": case_id, "seller_id": case["seller_id"], **parsed, "action_taken": action_taken}


def run_for_all_pending():
    init_db()
    cases = get_cases_needing_review()
    results = []

    for case in cases:
        if case.get("action_taken") not in (None, "none"):
            continue  # already remediated, skip
        result = remediate_case(case["case_id"])
        results.append(result)
        time.sleep(13)  # stay under the 5 RPM ceiling

    return results


if __name__ == "__main__":
    results = run_for_all_pending()
    for r in results:
        print(f"\n--- Case {r['case_id']} | Seller {r['seller_id']} ---")
        print(f"Action: {r['action']}")
        print(f"Requires human approval: {r['requires_human_approval']}")
        print(f"Justification: {r['justification']}")