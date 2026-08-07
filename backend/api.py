"""
api.py
------
Turns the existing risk_scorer.py / graph_engine.py / ml_scorer.py
pipeline into a website-ready backend.

Run with: uvicorn api:app --reload --port 8000
Then visit: http://localhost:8000/        <- the dashboard
            http://localhost:8000/docs    <- auto-generated API test page
"""

import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from risk_scorer import compute_final_risk
from ml_scorer import get_model_metrics

app = FastAPI(title="Trust Graph Fraud Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")


def _load_current_data():
    txn_path = os.path.join(DATA_DIR, "transactions.csv")
    del_path = os.path.join(DATA_DIR, "deliveries.csv")

    if not os.path.exists(txn_path) or not os.path.exists(del_path):
        raise HTTPException(
            status_code=500,
            detail=f"Data files not found. Expected {txn_path} and {del_path}",
        )

    transactions = pd.read_csv(txn_path)
    deliveries = pd.read_csv(del_path)
    return transactions, deliveries


@app.get("/")
def serve_dashboard():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"status": "Trust Graph API is running (no frontend/index.html found yet)"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/api/model-metrics")
def model_metrics():
    return get_model_metrics()


@app.get("/api/cases")
def get_cases():
    transactions, deliveries = _load_current_data()
    result = compute_final_risk(transactions, deliveries)
    return result.to_dict(orient="records")


@app.get("/api/cases/flagged")
def get_flagged_cases():
    transactions, deliveries = _load_current_data()
    result = compute_final_risk(transactions, deliveries)
    flagged = result[result["needs_agent_review"] == True]
    return flagged.to_dict(orient="records")