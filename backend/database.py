"""
database.py
-------------
SQLite persistence layer for Trust Graph.

Two tables:
  - fraud_cases : one row per seller case (latest score/tier snapshot)
  - audit_logs  : append-only trail of every state change on a case
                  (score computed, action taken, appeal filed, appeal
                  decided) -- never overwritten, only appended. This
                  is what satisfies the Auditability guardrail.
"""

import sqlite3
import os
import json
from datetime import datetime, timezone

DB_PATH = os.path.join(os.path.dirname(__file__), "trust_graph.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't already exist. Safe to call every run."""
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS fraud_cases (
            case_id             INTEGER PRIMARY KEY AUTOINCREMENT,
            seller_id           TEXT NOT NULL,
            final_risk_score    REAL NOT NULL,
            tier                TEXT NOT NULL,
            needs_agent_review  INTEGER NOT NULL,
            evidence_json       TEXT,
            explanation         TEXT,
            action_taken        TEXT DEFAULT 'none',
            status              TEXT DEFAULT 'open',
            created_at          TEXT NOT NULL,
            updated_at          TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS audit_logs (
            log_id      INTEGER PRIMARY KEY AUTOINCREMENT,
            case_id     INTEGER NOT NULL,
            event_type  TEXT NOT NULL,
            details     TEXT,
            timestamp   TEXT NOT NULL,
            FOREIGN KEY (case_id) REFERENCES fraud_cases (case_id)
        )
    """)

    conn.commit()
    conn.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_case(seller_id: str, final_risk_score: float, tier: str,
             needs_agent_review: bool, evidence: dict = None) -> int:
    """
    Insert a new fraud case (called after risk_scorer.py runs).
    evidence: the full row of signals from risk_scorer.py (graph_risk_score,
    return_rate, missing_proof_rate, etc.) -- stored as JSON so agents
    can access the real evidence later, not just the final score.
    Returns the new case_id. Also writes the first audit_logs entry.
    """
    conn = get_connection()
    cur = conn.cursor()
    now = _now()
    evidence_json = json.dumps(evidence or {})

    cur.execute("""
        INSERT INTO fraud_cases
            (seller_id, final_risk_score, tier, needs_agent_review,
             evidence_json, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (seller_id, final_risk_score, tier, int(needs_agent_review), evidence_json, now, now))

    case_id = cur.lastrowid

    cur.execute("""
        INSERT INTO audit_logs (case_id, event_type, details, timestamp)
        VALUES (?, ?, ?, ?)
    """, (case_id, "score_computed",
          f"final_risk_score={final_risk_score}, tier={tier}", now))

    conn.commit()
    conn.close()
    return case_id


def append_audit_log(case_id: int, event_type: str, details: str = ""):
    """
    Append-only write to audit_logs. Use this for every later state
    change: explanation_generated, action_taken, appeal_filed,
    appeal_decided.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO audit_logs (case_id, event_type, details, timestamp)
        VALUES (?, ?, ?, ?)
    """, (case_id, event_type, details, _now()))
    conn.commit()
    conn.close()


def update_case(case_id: int, explanation: str = None,
                 action_taken: str = None, status: str = None):
    """
    Update the mutable snapshot fields on fraud_cases (NOT the audit
    trail itself -- that stays append-only). Only pass the fields
    you want to change.
    """
    conn = get_connection()
    cur = conn.cursor()

    fields, values = [], []
    if explanation is not None:
        fields.append("explanation = ?")
        values.append(explanation)
    if action_taken is not None:
        fields.append("action_taken = ?")
        values.append(action_taken)
    if status is not None:
        fields.append("status = ?")
        values.append(status)

    if not fields:
        conn.close()
        return

    fields.append("updated_at = ?")
    values.append(_now())
    values.append(case_id)

    cur.execute(f"UPDATE fraud_cases SET {', '.join(fields)} WHERE case_id = ?", values)
    conn.commit()
    conn.close()


def get_case(case_id: int) -> dict:
    """
    Returns the case as a dict, with the stored evidence_json unpacked
    and merged in -- so callers can access case['graph_risk_score'],
    case['return_rate'], etc. directly, not just the final score.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM fraud_cases WHERE case_id = ?", (case_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    case = dict(row)
    evidence = json.loads(case.get("evidence_json") or "{}")
    case.update(evidence)
    return case


def get_audit_trail(case_id: int) -> list:
    """Full immutable history for a case, oldest first -- what an
    accused party would read on appeal."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT * FROM audit_logs WHERE case_id = ?
        ORDER BY timestamp ASC
    """, (case_id,))
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_cases_needing_review() -> list:
    """Cases with needs_agent_review = True and still open -- this is
    the queue the LLM agents will pull from next."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT * FROM fraud_cases
        WHERE needs_agent_review = 1 AND status = 'open'
        ORDER BY final_risk_score DESC
    """)
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def init_appeals_table():
    """Call once alongside init_db() -- separate table for appeals."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS appeals (
            appeal_id     INTEGER PRIMARY KEY AUTOINCREMENT,
            case_id       INTEGER NOT NULL,
            seller_id     TEXT NOT NULL,
            statement     TEXT NOT NULL,
            status        TEXT DEFAULT 'pending',
            decision      TEXT,
            decided_by    TEXT,
            created_at    TEXT NOT NULL,
            decided_at    TEXT,
            FOREIGN KEY (case_id) REFERENCES fraud_cases (case_id)
        )
    """)
    conn.commit()
    conn.close()


def file_appeal(case_id: int, seller_id: str, statement: str) -> int:
    """Seller files an appeal against an actioned case. Routes to a
    human investigator dashboard, NOT back through the AI agents."""
    conn = get_connection()
    cur = conn.cursor()
    now = _now()

    cur.execute("""
        INSERT INTO appeals (case_id, seller_id, statement, created_at)
        VALUES (?, ?, ?, ?)
    """, (case_id, seller_id, statement, now))
    appeal_id = cur.lastrowid
    conn.commit()
    conn.close()

    append_audit_log(case_id, "appeal_filed", f"appeal_id={appeal_id}: {statement[:100]}")
    update_case(case_id, status="under_appeal")
    return appeal_id


def decide_appeal(appeal_id: int, decision: str, decided_by: str = "admin"):
    """Human investigator decides an appeal: 'upheld' (action stands)
    or 'overturned' (action reversed)."""
    conn = get_connection()
    cur = conn.cursor()
    now = _now()

    cur.execute("SELECT case_id FROM appeals WHERE appeal_id = ?", (appeal_id,))
    row = cur.fetchone()
    if not row:
        conn.close()
        raise ValueError(f"No appeal found with appeal_id={appeal_id}")
    case_id = row["case_id"]

    cur.execute("""
        UPDATE appeals SET status = 'decided', decision = ?, decided_by = ?, decided_at = ?
        WHERE appeal_id = ?
    """, (decision, decided_by, now, appeal_id))
    conn.commit()
    conn.close()

    append_audit_log(case_id, "appeal_decided", f"appeal_id={appeal_id}: {decision} by {decided_by}")
    new_status = "closed_overturned" if decision == "overturned" else "closed_upheld"
    update_case(case_id, status=new_status)


def get_appeals_for_case(case_id: int) -> list:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM appeals WHERE case_id = ? ORDER BY created_at ASC", (case_id,))
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_pending_appeals() -> list:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM appeals WHERE status = 'pending' ORDER BY created_at ASC")
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def init_all():
    init_db()
    init_appeals_table()


if __name__ == "__main__":
    # Quick smoke test: wire risk_scorer.py output into the DB
    import pandas as pd
    from risk_scorer import compute_final_risk

    init_all()

    txns = pd.read_csv("data/transactions.csv")
    dels = pd.read_csv("data/deliveries.csv")
    result = compute_final_risk(txns, dels)

    for _, row in result.iterrows():
        case_id = log_case(
            seller_id=row["seller_id"],
            final_risk_score=row["final_risk_score"],
            tier=row["tier"],
            needs_agent_review=bool(row["needs_agent_review"]),
            evidence=row.to_dict(),
        )

    print(f"Logged {len(result)} cases into {DB_PATH}")
    print("\nCases needing agent review:")
    for c in get_cases_needing_review():
        print(f"  case_id={c['case_id']} seller={c['seller_id']} "
              f"score={c['final_risk_score']} tier={c['tier']}")