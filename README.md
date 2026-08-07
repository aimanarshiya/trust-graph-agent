---

## 5. Running It Locally

```powershell
# from the project root
cd trust-graph-agent
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# generate the synthetic dataset
python generate_sample_data.py
xcopy data backend\data /E /I /Y

# set up your API key
cd backend
copy .env.example .env
# open .env and paste in your own GEMINI_API_KEY (get one free at
# https://aistudio.google.com/app/apikey)
```

Run the full pipeline end to end:
```powershell
python main.py
```

Or run each stage individually:
```powershell
python graph_engine.py        # graph collusion scores
python risk_scorer.py         # combined risk scores
python database.py            # log all cases into SQLite
python agents/explainer_agent.py
python agents/remediation_agent.py
python agents/selfcheck_agent.py
python fairness_audit.py
```

---

## 6. What's Next

1. `api.py` — FastAPI wrapper to serve live case/risk data as JSON
2. Frontend dashboard — case queue, evidence view, appeal button
3. IEEE-CIS transaction classifier (real ML model, currently graph+rules only)
4. Elliptic benchmark validation for the graph technique
5. Live API integrations (AbuseIPDB, GSTINCheck, DeBounce, SendGrid, Twilio)
6. Tech write-up + business pitch

---

*This README reflects project status as of the current build session. See git
commit history for the detailed development log.*