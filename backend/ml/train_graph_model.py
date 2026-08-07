"""
train_graph_model.py
----------------------
Proves the graph-based anomaly detection technique on the Elliptic
Bitcoin dataset (203K-node labeled transaction graph), per the
problem statement's requirement:

    "Catch fraud a single-transaction classifier misses, using
    graph-based anomaly detection proven on real labeled data"

Approach: compute classical GRAPH-STRUCTURAL features (degree,
PageRank, clustering coefficient, k-core, ego-network stats) from the
transaction graph topology -- not the raw per-node features Elliptic
also ships. This is what makes it "graph-based": the model learns
from *network position*, which is exactly the signal a single-
transaction classifier cannot see (collusion rings, hub/spoke
laundering patterns, etc.).

The same feature-extraction function is reused in graph_engine.py to
score your own seller/delivery-partner graph.

Download the dataset first (adjust slug if kagglehub finds a
different exact name in search):
    kagglehub.dataset_download('ellipticco/elliptic-data-set')
"""

import os
import gc
import joblib
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, roc_auc_score, classification_report
import xgboost as xgb

# ---- EDIT THIS to match your downloaded path ----
DATA_PATH = r"C:\Users\DELL-PC\.cache\kagglehub\datasets\ellipticco\elliptic-data-set\versions\1\elliptic_bitcoin_dataset"
# ^ confirmed correct path from your kagglehub download

MODEL_OUT_DIR = os.path.join(os.path.dirname(__file__), "graph_artifacts")
os.makedirs(MODEL_OUT_DIR, exist_ok=True)


def build_graph_structural_features(edges_df: pd.DataFrame, node_ids: pd.Series) -> pd.DataFrame:
    """
    Given an edge list and the set of node ids to score, builds a
    graph and computes structural features per node. This function is
    reused (same shape of output) for your own seller graph later.
    """
    G = nx.DiGraph()
    G.add_nodes_from(node_ids)
    G.add_edges_from(edges_df.itertuples(index=False, name=None))

    print("  computing degree...")
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())

    print("  computing pagerank...")
    pagerank = nx.pagerank(G, alpha=0.85, max_iter=100)

    print("  computing clustering coefficient (on undirected view)...")
    G_undirected = G.to_undirected()
    clustering = nx.clustering(G_undirected)

    print("  computing k-core number...")
    G_undirected.remove_edges_from(nx.selfloop_edges(G_undirected))
    core_number = nx.core_number(G_undirected)

    features = pd.DataFrame({
        "node_id": list(node_ids),
    })
    features["in_degree"] = features["node_id"].map(in_deg).fillna(0)
    features["out_degree"] = features["node_id"].map(out_deg).fillna(0)
    features["total_degree"] = features["in_degree"] + features["out_degree"]
    features["pagerank"] = features["node_id"].map(pagerank).fillna(0)
    features["clustering_coef"] = features["node_id"].map(clustering).fillna(0)
    features["k_core"] = features["node_id"].map(core_number).fillna(0)

    return features


def main():
    print("Loading Elliptic dataset...")
    classes = pd.read_csv(os.path.join(DATA_PATH, "elliptic_txs_classes.csv"))
    edges = pd.read_csv(os.path.join(DATA_PATH, "elliptic_txs_edgelist.csv"))

    # Elliptic labels: '1' = illicit, '2' = licit, 'unknown' = unlabeled
    classes = classes[classes["class"] != "unknown"].copy()
    classes["label"] = (classes["class"] == "1").astype(int)  # 1 = fraud/illicit

    print(f"Labeled nodes: {len(classes)} (illicit: {classes['label'].sum()})")

    print("Building graph and computing structural features...")
    all_node_ids = pd.concat([edges["txId1"], edges["txId2"], classes["txId"]]).unique()
    edges_renamed = edges.rename(columns={"txId1": "src", "txId2": "dst"})[["src", "dst"]]
    feats = build_graph_structural_features(edges_renamed, pd.Series(all_node_ids))

    df = classes.merge(feats, left_on="txId", right_on="node_id", how="left").fillna(0)
    gc.collect()

    feature_cols = ["in_degree", "out_degree", "total_degree", "pagerank", "clustering_coef", "k_core"]
    X = df[feature_cols]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training graph-structural anomaly model...")
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="auc",
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    print(f"\n--- ELLIPTIC GRAPH MODEL RESULTS ---")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"AUC:       {auc:.4f}")
    print(classification_report(y_test, y_pred))

    joblib.dump(model, os.path.join(MODEL_OUT_DIR, "graph_model.joblib"))
    joblib.dump(feature_cols, os.path.join(MODEL_OUT_DIR, "graph_feature_columns.joblib"))
    joblib.dump(
        {"precision": precision, "recall": recall, "auc": auc},
        os.path.join(MODEL_OUT_DIR, "graph_model_metrics.joblib"),
    )
    print(f"\nSaved to {MODEL_OUT_DIR}")


if __name__ == "__main__":
    main()