"""
selfcheck_agent.py
---------------------
Self-check step: before a remediation action is considered final,
verify it against business-goal guardrails. This is rules-based
(cheap, fast, no LLM required for the check itself) per the build
principle "use classical solvers where practical." An LLM is only
used to write a short plain-language summary of the check result
for the audit trail -- not to make the actual decision.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from llm import call_llm
from database import get_case, update_case, append_audit_log, init_db, get_connection

PRECISION_THRESHOLD = 0.95
MODEL_PRECISION = 0.91  # keep in sync with remediation_agent.py


def check_confidence_guardrail(case: dict) -> dict:
    """
    Rule 1: hard actions must never fire below the precision threshold.
    """
    is_hard_action = case.get("action_taken") == "hard_suspension"
    guardrail_respected = not is_hard_action or MODEL_PRECISION >= PRECISION_THRESHOLD

    return {
        "check": "confidence_guardrail",
        "passed": guardrail_respected,
        "detail": (
            f"Action='{case.get('action_taken')}', model_precision="
            f"{MODEL_PRECISION:.2f}, threshold={PRECISION_THRESHOLD:.0%}. "
            f"{'OK -- guardrail respected.' if guardrail_respected else 'VIOLATION -- hard action taken below threshold!'}"
        ),
    }


def check_fairness_signal(case: dict, all_cases: list) -> dict:
    """
    Rule 2: lightweight proxy for the fairness guardrail -- flags if
    a disproportionate share of ALL non-'no_action' cases so far are
    concentrated on a tiny handful of sellers (could indicate the
    system is fixating rather than distributing scrutiny fairly).
    This is a coarse hackathon-scope proxy; the real fairness report
    (by seller tenure/size cohort) belongs in the admin dashboard
    reports module, not here.
    """
    acted_cases = [c for c in all_cases if c.get("action_taken") not in (None, "none")]
    total_acted = len(acted_cases)

    if total_acted == 0:
        return {"check": "fairness_signal", "passed": True,
                "detail": "No actioned cases yet -- nothing to assess."}

    # crude concentration check: does any single seller account for
    # more than 40% of all actioned cases? (placeholder heuristic)
    from collections import Counter
    counts = Counter(c["seller_id"] for c in acted_cases)
    top_seller, top_count = counts.most_common(1)[0]
    concentration = top_count / total_acted

    passed = concentration <= 0.4
    return {
        "check": "fairness_signal",
        "passed": passed,
        "detail": (
            f"Top-flagged seller '{top_seller}' accounts for "
            f"{concentration:.0%} of all actioned cases. "
            f"{'OK -- reasonably distributed.' if passed else 'FLAG -- concentration may indicate unfair targeting, review recommended.'}"
        ),
    }


def run_selfcheck(case_id: int) -> dict:
    """
    Runs both rules-based checks for a case, asks the LLM for a short
    plain-language summary (for the audit trail only -- not decision-
    making), and logs the result.
    """
    case = get_case(case_id)
    if case is None:
        raise ValueError(f"No case found with case_id={case_id}")

    conn = get_connection()
    all_cases = [dict(r) for r in conn.execute("SELECT * FROM fraud_cases").fetchall()]
    conn.close()

    checks = [
        check_confidence_guardrail(case),
        check_fairness_signal(case, all_cases),
    ]

    all_passed = all(c["passed"] for c in checks)
    checks_text = "\n".join(f"- {c['check']}: {'PASS' if c['passed'] else 'FAIL'} -- {c['detail']}" for c in checks)

    summary_prompt = f"""Summarize this self-check result in ONE plain-language
sentence for an audit log. Be factual, no speculation:

{checks_text}"""

    summary = call_llm(summary_prompt, system_instruction=(
        "You write single-sentence factual audit log summaries. "
        "No preamble, no markdown, just the sentence."
    ))

    status = "final" if all_passed else "flagged_for_human_review"
    if not all_passed:
        update_case(case_id, status="flagged_for_human_review")

    append_audit_log(
        case_id,
        event_type="self_check_completed",
        details=f"status={status} | {summary} | raw_checks: {checks_text}"
    )

    return {
        "case_id": case_id,
        "seller_id": case["seller_id"],
        "all_passed": all_passed,
        "status": status,
        "summary": summary,
        "checks": checks,
    }


def run_for_all_pending():
    init_db()
    conn = get_connection()
    cases = [dict(r) for r in conn.execute(
        "SELECT * FROM fraud_cases WHERE action_taken != 'none' AND status = 'open'"
    ).fetchall()]
    conn.close()

    results = []
    for case in cases:
        results.append(run_selfcheck(case["case_id"]))
    return results


if __name__ == "__main__":
    results = run_for_all_pending()
    for r in results:
        print(f"\n--- Case {r['case_id']} | Seller {r['seller_id']} ---")
        print(f"STATUS: {r['status']}")
        print(f"SUMMARY: {r['summary']}")
        for c in r["checks"]:
            print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {c['check']}: {c['detail']}")