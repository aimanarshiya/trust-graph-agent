import pandas as pd
import numpy as np
import joblib
import os
import gc
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, roc_auc_score, classification_report
import xgboost as xgb

DATA_PATH = r"C:\Users\DELL-PC\.cache\kagglehub\competitions\ieee-fraud-detection"
MODEL_OUT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
os.makedirs(MODEL_OUT_DIR, exist_ok=True)


def downcast(df):
    """Shrink numeric dtypes to the smallest safe type — cuts memory ~50-70%."""
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object and str(col_type) != "str":
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min >= -32768 and c_max <= 32767:
                    df[col] = df[col].astype(np.int16)
                elif c_min >= -2147483648 and c_max <= 2147483647:
                    df[col] = df[col].astype(np.int32)
            else:  # float
                df[col] = df[col].astype(np.float32)
    return df


print("Loading transaction data...")
train_txn = pd.read_csv(os.path.join(DATA_PATH, "train_transaction.csv"))
train_txn = downcast(train_txn)
gc.collect()

print("Loading identity data...")
train_id = pd.read_csv(os.path.join(DATA_PATH, "train_identity.csv"))
train_id = downcast(train_id)
gc.collect()

print("Merging...")
df = train_txn.merge(train_id, on="TransactionID", how="left")
del train_txn, train_id
gc.collect()
print(f"Merged shape: {df.shape}")

y = df["isFraud"].astype(np.int8)
X = df.drop(columns=["isFraud", "TransactionID"])
del df
gc.collect()

# Label-encode categoricals in place (no extra copy)
cat_cols = X.select_dtypes(include=["object", "str"]).columns
print(f"Encoding {len(cat_cols)} categorical columns...")
for col in cat_cols:
    X[col] = X[col].astype("category").cat.codes.astype(np.int32)
gc.collect()

print(f"Final memory usage: {X.memory_usage(deep=True).sum() / 1e9:.2f} GB")

# Stratified split keeps fraud ratio consistent train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
del X, y
gc.collect()

print("Training XGBoost...")
model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",  # much lower memory footprint than default
    eval_metric="auc",
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

print("Evaluating...")
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba >= 0.5).astype(int)

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print(f"\n--- RESULTS ---")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"AUC:       {auc:.4f}")
print(classification_report(y_test, y_pred))

# Find a threshold that gives a more balanced precision (default 0.5 threshold skews toward recall)
from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
for p, r, t in zip(precisions, recalls, thresholds):
    if p >= 0.5:
        print(f"Threshold {t:.3f} -> Precision {p:.3f}, Recall {r:.3f}")
        break

joblib.dump(model, os.path.join(MODEL_OUT_DIR, "fraud_model.joblib"))
joblib.dump(list(X_train.columns), os.path.join(MODEL_OUT_DIR, "feature_columns.joblib"))

metrics = {"precision": precision, "recall": recall, "auc": auc}
joblib.dump(metrics, os.path.join(MODEL_OUT_DIR, "model_metrics.joblib"))

print(f"\nModel + metrics saved to {MODEL_OUT_DIR}")
