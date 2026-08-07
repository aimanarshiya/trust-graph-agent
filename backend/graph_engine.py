"""
graph_engine.py
-----------------
CLASSICAL graph-based anomaly detection. No LLM calls here -- this is
the "cheap and fast" layer the problem statement explicitly asks for.

WHAT IT DOES:
1. Builds a graph connecting customers <-> sellers <-> shared devices/IPs
2. Finds actors that share a device or IP (a strong collusion signal --
   different "customers" acting from the same device is a classic
   self-ordering fraud pattern)
3. Runs community detection to find tightly-connected clusters that
   look like coordinated rings, not organic shopping behavior
4. Outputs a per-seller and per-customer graph anomaly score

This score feeds into risk_scorer.py, which combines it with rule-based
signals before deciding whether a case is worth an expensive LLM call.
"""

import pandas as pd
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities


def build_actor_graph(transactions: pd.DataFrame) -> nx.Graph:
    """
    Builds an undirected graph where nodes are customers AND sellers,
    and an edge exists between them if a transaction happened. Also
    adds edges between customers who share a device_id or ip_address --
    this is what turns isolated "self-ordering" into a visible cluster.
    """
    G = nx.Graph()

    for _, row in transactions.iterrows():
        cust, seller = row["customer_id"], row["seller_id"]
        G.add_node(cust, type="customer")
        G.add_node(seller, type="seller")
        # weight = number of transactions between this pair (repeat buying is a signal)
        if G.has_edge(cust, seller):
            G[cust][seller]["weight"] += 1
        else:
            G.add_edge(cust, seller, weight=1, relation="transacted")

    # Link customers who share a device or IP -- the core collusion signal
    device_groups = transactions.groupby("device_id")["customer_id"].unique()
    for device, custs in device_groups.items():
        custs = list(set(custs))
        if len(custs) > 1:
            for i in range(len(custs)):
                for j in range(i + 1, len(custs)):
                    G.add_edge(custs[i], custs[j], weight=5, relation="shared_device")

    ip_groups = transactions.groupby("ip_address")["customer_id"].unique()
    for ip, custs in ip_groups.items():
        custs = list(set(custs))
        if len(custs) > 1:
            for i in range(len(custs)):
                for j in range(i + 1, len(custs)):
                    G.add_edge(custs[i], custs[j], weight=5, relation="shared_ip")

    return G


def detect_collusion_rings(G: nx.Graph) -> list[set]:
    """
    Community detection: finds clusters of nodes that are more densely
    connected to each other than to the rest of the graph. A tight
    cluster containing customers linked ONLY by shared device/IP (not
    organic browsing) is a strong collusion signal.
    """
    communities = list(greedy_modularity_communities(G, weight="weight"))
    # Only keep small, dense communities -- large ones are just "popular seller" noise
    suspicious = [c for c in communities if 3 <= len(c) <= 15]
    return suspicious


def graph_anomaly_scores(transactions: pd.DataFrame) -> pd.DataFrame:
    """
    Main entry point. Returns a DataFrame with one row per seller,
    including a graph_risk_score (0-1) based on:
    - how many customers linked to this seller share a device/IP
    - whether the seller sits inside a detected suspicious community
    """
    G = build_actor_graph(transactions)
    rings = detect_collusion_rings(G)

    ring_members = set()
    for ring in rings:
        ring_members.update(ring)

    results = []
    for seller_id in transactions["seller_id"].unique():
        seller_txns = transactions[transactions["seller_id"] == seller_id]
        customers = set(seller_txns["customer_id"])

        # How many of this seller's customers share a device/IP with each other?
        shared_attr_customers = sum(1 for c in customers if c in ring_members)
        shared_ratio = shared_attr_customers / max(len(customers), 1)

        # Is this seller itself inside a suspicious tight community?
        in_ring = seller_id in ring_members

        # Simple, explainable scoring formula (not a black box -- important
        # for the "explain the evidence in plain language" requirement)
        graph_risk_score = min(1.0, 0.6 * shared_ratio + 0.4 * (1 if in_ring else 0))

        results.append({
            "seller_id": seller_id,
            "num_customers": len(customers),
            "shared_attr_customers": shared_attr_customers,
            "shared_attr_ratio": round(shared_ratio, 3),
            "in_suspicious_community": in_ring,
            "graph_risk_score": round(graph_risk_score, 3),
        })

    return pd.DataFrame(results).sort_values("graph_risk_score", ascending=False)


if __name__ == "__main__":
    txns = pd.read_csv("data/transactions.csv")
    scores = graph_anomaly_scores(txns)
    print(scores.head(10).to_string(index=False))