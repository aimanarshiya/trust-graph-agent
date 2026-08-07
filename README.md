# Trust Graph — Fraud Detection & Trust Analysis System

AI-powered fraud detection for e-commerce, built for [hackathon name] against the
"Trust Graph" problem statement: detect seller/customer/delivery-partner collusion,
explain *why* a case is risky in plain language, and route decisions through a
graduated, human-reviewable response — never straight to a hard action.

---

## 1. Problem Statement (summary)

E-commerce fraud isn't just dishonest customers — sellers, delivery partners, and
customers can all be dishonest, sometimes together (collusion). The system must:

- Score transactions/sellers for fraud risk using real data
- Use **graph-based anomaly detection** to catch network-level collusion, not just
  single-transaction rules
- **Explain** why something was flagged, in plain language
- Use a **graduated response**: soft actions (extra verification, hold payout)
  before hard actions (suspension) — and hard actions require ≥95% precision or a
  human review
- Give every flagged seller a **human-reviewable appeal path**
- Respect fairness (no disproportionate flagging of small/new sellers) and data
  residency requirements

---

## 2. Current Status

| Component | Status | Notes |
|---|---|---|
| Sample/demo transaction data | ✅ Done | `backend/data/` — includes a deliberate collusion ring for demo purposes |
| Classical graph collusion detection | ✅ Done | `backend/graph_engine.py` — no LLM, no ML, pure graph algorithm |
| Trained ML fraud model | ✅ Done | `backend/ml/train_model.py` — XGBoost on IEEE-CIS Kaggle dataset, real validated precision/recall/AUC |
| Graph technique cross-validation | ✅ Done | Same graph technique validated separately on the Elliptic dataset |
| Combined risk scoring | ✅ Done | `backend/risk_scorer.py` — blends graph + ML + rule signals into one explainable score per seller |
| REST API | ✅ Done | `backend/api.py` — FastAPI, serves live risk data as JSON |
| Localhost dashboard | ✅ Done | `frontend/index.html` — working, tested end to end |
| Explainer agent (LLM) | ✅ Done | `backend/agents/explainer_agent.py` — verified: all 7 pending cases have real, evidence-grounded explanations stored |
| Remediation agent (LLM) | 🔶 In progress | `backend/agents/remediation_agent.py` — JSON-based parsing implemented (robust against truncation); blocked on final verification run by rate limit, code is complete |
| Self-check agent (LLM) | ⬜ Not started | Validates the analysis before final output |
| Database / persistence | ⬜ Not started | Currently recomputes live from CSV on every API call |
| Appeal workflow | ⬜ Not started | Required deliverable |
| Fairness audit | ⬜ Not started | Seller size/tenure parity report — required, graded |
| Tech write-up | ⬜ Not started | Required deliverable |
| Business pitch | ⬜ Not started | Required deliverable |

**Honest scope note:** this is an ambitious, enterprise-scale problem statement.
The goal isn't full production coverage — it's a small, real, explainable slice of
the system, built end to end, that we can demo and defend live rather than fake.

---

## 3. Architecture

```
backend/data/transactions.csv + deliveries.csv
              │
              ├──────────────────┬──────────────────────┐
              ▼                  ▼                       ▼
      graph_engine.py      risk_scorer.py           ml_scorer.py
   graph_anomaly_scores() compute_rule_signals()  compute_ml_signals()
              │                  │                       │
     graph_risk_score      return_rate,             ml_fraud_score
      (0–1 per seller)     missing_proof_rate             │
              │                  │                       │
              └──────────┬───────┴───────────────────────┘
                         ▼
              risk_scorer.py: compute_final_risk()
                         │
       final_risk_score = 0.45×graph + 0.25×ML
                        + 0.15×return_rate + 0.15×missing_proof
                         │
                         ▼
              classify_tier() → no_action / soft_intervention
                                 / hard_action_candidate
                         │
                         ▼
             needs_agent_review = score ≥ 0.4
        (ONLY these cases are worth an LLM call — cheap/
         classical layer decides if expensive reasoning is needed)
                         │
                         ▼
                    backend/api.py (FastAPI)
                         │
                         ▼
                 frontend/index.html (dashboard)
```

### Why this design (for judges / evaluators)

- **Graph collusion detection is classical, not ML** — `networkx` community
  detection on a seller↔customer↔device↔IP graph. Cheap, fast, and fully
  explainable: every flag traces back to a real shared device/IP or repeated
  transaction triad, not a black-box score.
- **The ML model is real and validated**, not a placeholder — trained on the
  IEEE-CIS Kaggle fraud dataset, with actual precision/recall/AUC reported (not
  hardcoded).
- **Expensive LLM reasoning is reserved for borderline/high-risk cases only**
  (`needs_agent_review`), matching the "cheap models for routine work, expensive
  reasoning for high-stakes decisions" principle.
- **No automated hard actions.** The system is deliberately designed so nothing
  can auto-suspend a seller — every high-risk case still requires human review,
  satisfying the appeal/fairness requirements from the problem statement.

---

## 4. Project Structure

```
trust-graph-agent/
├── backend/
│   ├── data/
│   │   ├── transactions.csv
│   │   └── deliveries.csv
│   ├── ml/
│   │   ├── train_model.py        # IEEE-CIS XGBoost training pipeline
│   │   ├── train_graph_model.py  # Elliptic graph-technique validation
│   │   └── artifacts/            # fraud_model.joblib, feature_columns.joblib, model_metrics.joblib
│   ├── agents/                   # (planned) explainer, remediation, selfcheck
│   ├── graph_engine.py
│   ├── risk_scorer.py
│   ├── ml_scorer.py
│   ├── api.py
│   └── .env                      # Kaggle + LLM API keys (never committed)
├── frontend/
│   └── index.html
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 5. Running It Locally

```powershell
# from the project root
cd trust-graph-agent
venv\Scripts\activate

# from backend/
cd backend
pip install -r requirements.txt
uvicorn api:app --reload --port 8000
```

Open **http://localhost:8000** — the dashboard loads live risk scores for every
seller in `backend/data/transactions.csv`, along with the real ML model's
validated precision/recall/AUC.

API docs (auto-generated): **http://localhost:8000/docs**

---

## 6. What's Next

1. `agents/explainer_agent.py` — turn each seller's evidence (`reasons`,
   `graph_risk_score`, `return_rate`, etc.) into a plain-language explanation
2. `agents/remediation_agent.py` — recommend soft vs. hard action, enforcing the
   95%-precision guardrail before ever suggesting a hard action
3. `database.py` — persist cases so history/appeals survive a restart
4. Appeal endpoint — minimal human-review workflow
5. `fairness_audit.py` — parity report across seller size/tenure
6. Tech write-up + business pitch

---

*This README reflects project status as of the current build session. See git
commit history for the detailed development log.*