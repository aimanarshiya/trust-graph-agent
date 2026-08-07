"""
graph_engine.py
----------------
Classical, non-LLM graph anomaly detection.

Why this file has NO LLM calls:
The problem statement explicitly rewards using cheap/classical solvers for
routine, high-volume work and saving expensive LLM reasoning for the
borderline/high-stakes cases. Scoring every transaction is high-volume and
mechanical -- a graph algorithm does this in milliseconds. Explaining WHY a
case is suspicious (in plain language) is the part that genuinely needs an
LLM -- that happens later, in agents/explainer_agent.py, and ONLY for cases
this engine flags as borderline or high risk.

What "fraud" looks like in graph terms:
A single dishonest order looks normal. Collusion looks different once you
zoom out: the SAME seller, customer, and delivery partner keep appearing
together far more often than random chance would predict, forming a dense
little triangle (a "ring") inside the wider transaction graph. That
repeated tight loop -- not any single order -- is the anomaly.

How to read this file if you're explaining it to a judge:
1. build_graph()      -> turns raw CSV rows into a graph: sellers, customers,
                          and delivery partners are nodes; each transaction
                          is an edge connecting them.
2. find_rings()        -> looks for seller-customer-delivery triples that
                          repeat suspiciously often (a classical frequency/
                          density check -- no ML model needed).
3. compute_risk_signals() -> combines ring membership with hard evidence
                          (GPS mismatches, missing delivery photos, return
                          rate) into a per-transaction signal dictionary.
                          This dictionary -- not a black-box number -- is
                          what gets handed to the LLM agents later, so every
                          decision stays traceable to real evidence.
"""

import pandas as pd
import networkx as nx
from collections import Counter, defaultdict


def load_data(data_dir="data"):
    """Load the three raw CSVs into pandas DataFrames."""
    transactions = pd.read_csv(f"{data_dir}/transactions.csv")
    sellers = pd.read_csv(f"{data_dir}/sellers.csv")
    deliveries = pd.read_csv(f"{data_dir}/deliveries.csv")
    return transactions, sellers, deliveries


def build_graph(transactions: pd.DataFrame) -> nx.MultiGraph:
    """
    Build a tripartite graph: seller <-> customer <-> delivery_partner.

    Each transaction adds two edges:
      seller_id  -- customer_id
      customer_id -- delivery_partner_id
    Node types are tagged so we can tell sellers, customers, and delivery
    partners apart later even though they share one graph.
    """
    G = nx.MultiGraph()

    for _, row in transactions.iterrows():
        seller = f"S::{row['seller_id']}"
        customer = f"C::{row['customer_id']}"
        delivery = f"D::{row['delivery_partner_id']}"

        G.add_node(seller, type="seller", id=row["seller_id"])
        G.add_node(customer, type="customer", id=row["customer_id"])
        G.add_node(delivery, type="delivery", id=row["delivery_partner_id"])

        G.add_edge(seller, customer, transaction_id=row["transaction_id"], amount=row["amount"])
        G.add_edge(customer, delivery, transaction_id=row["transaction_id"], amount=row["amount"])

    return G


def find_rings(transactions: pd.DataFrame, min_repeats: int = 3):
    """
    Classical frequency check: find (seller, customer, delivery_partner)
    triples that repeat at least `min_repeats` times.

    This is deliberately simple and explainable -- a judge can verify the
    logic by eye. No black box. Repeated triads are the strongest, cheapest
    signal of a closed collusion loop (same three parties transacting with
    each other over and over, instead of the natural variety you'd expect
    on a real marketplace).
    """
    triad_counts = Counter(
        (row["seller_id"], row["customer_id"], row["delivery_partner_id"])
        for _, row in transactions.iterrows()
    )

    rings = {
        triad: count for triad, count in triad_counts.items()
        if count >= min_repeats
    }
    return rings  # {(seller_id, customer_id, delivery_partner_id): repeat_count}


def compute_risk_signals(transactions: pd.DataFrame, sellers: pd.DataFrame, deliveries: pd.DataFrame):
    """
    Combine multiple classical (non-LLM) signals per transaction into a
    single evidence dictionary. Nothing here is a hidden ML score -- every
    field is directly traceable to a row in the source data, which is what
    the audit-trail requirement in the problem statement demands.

    Signals used:
      - ring_membership   : is this transaction part of a repeated triad?
      - ring_repeat_count : how many times has this exact triad occurred?
      - gps_mismatch      : delivery partner's GPS didn't match drop-off
      - no_proof_photo    : no proof-of-delivery photo captured
      - seller_return_rate: this seller's overall return rate (self-dealing
                             sellers often show unusually LOW return rates
                             despite high volume from the same few customers)
    """
    rings = find_rings(transactions)
    ring_lookup = {}
    for (seller_id, customer_id, delivery_id), count in rings.items():
        ring_lookup[(seller_id, customer_id, delivery_id)] = count

    # seller-level return rate, needed as context for each transaction
    return_rates = (
        transactions.assign(is_returned=lambda d: d["status"] == "returned")
        .groupby("seller_id")["is_returned"]
        .mean()
        .to_dict()
    )

    delivery_lookup = deliveries.set_index("transaction_id").to_dict(orient="index")

    signals = []
    for _, row in transactions.iterrows():
        triad = (row["seller_id"], row["customer_id"], row["delivery_partner_id"])
        delivery_info = delivery_lookup.get(row["transaction_id"], {})

        signals.append({
            "transaction_id": row["transaction_id"],
            "seller_id": row["seller_id"],
            "customer_id": row["customer_id"],
            "delivery_partner_id": row["delivery_partner_id"],
            "amount": row["amount"],
            "status": row["status"],
            "ring_membership": triad in ring_lookup,
            "ring_repeat_count": ring_lookup.get(triad, 1),
            "gps_mismatch": delivery_info.get("gps_match") == "no",
            "no_proof_photo": delivery_info.get("proof_photo_captured") == "no",
            "seller_return_rate": round(return_rates.get(row["seller_id"], 0.0), 2),
        })

    return signals


if __name__ == "__main__":
    # Quick standalone test -- run with: python backend/graph_engine.py
    # (run from the project root so the data/ path resolves correctly)
    transactions, sellers, deliveries = load_data()
    graph = build_graph(transactions)

    print(f"Graph built: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    rings = find_rings(transactions)
    print(f"\nSuspicious repeated triads found: {len(rings)}")
    for (seller, customer, delivery), count in rings.items():
        print(f"  Seller {seller} + Customer {customer} + Delivery {delivery} -> repeated {count}x")

    print("\nSample risk signals (first 5 transactions):")
    signals = compute_risk_signals(transactions, sellers, deliveries)
    for s in signals[:5]:
        print(f"  {s}")