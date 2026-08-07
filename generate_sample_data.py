"""
generate_sample_data.py
--------------------------
Generates synthetic e-commerce transaction/seller/delivery data with
a known, planted subset of collusion rings. Includes device_id and
ip_address columns required by graph_engine.py's collusion detection,
plus signup_email for the DeBounce disposable-email check.
"""

import pandas as pd
import numpy as np
import os
import random

random.seed(42)
np.random.seed(42)

OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_NORMAL_SELLERS = 40
NUM_RING_SELLERS = 3
NUM_CUSTOMERS = 120
NUM_DELIVERY_PARTNERS = 10

transactions = []
deliveries = []
sellers = []

txn_counter = 1000


def new_txn_id():
    global txn_counter
    txn_counter += 1
    return f"T{txn_counter}"


def fake_ip(seed_id):
    """Deterministic fake IP per id, so the same customer usually
    reuses the same IP across orders (realistic-ish)."""
    return f"192.168.{seed_id % 256}.{(seed_id * 7) % 256}"


# ---- Normal sellers ----
for i in range(NUM_NORMAL_SELLERS):
    seller_id = f"S{i:03d}"
    tenure_days = random.choice([random.randint(5, 89), random.randint(90, 900)])
    sellers.append({
        "seller_id": seller_id,
        "tenure_days": tenure_days,
        "size": random.choice(["small", "medium", "large"]),
        "signup_email": f"seller{i}@gmail.com",
    })

    num_txns = random.randint(6, 15)
    used_customers = random.sample(range(NUM_CUSTOMERS), k=min(num_txns, NUM_CUSTOMERS))

    for c_idx in used_customers:
        customer_id = f"C{c_idx:03d}"
        delivery_partner_id = f"D{random.randint(0, NUM_DELIVERY_PARTNERS - 1):02d}"
        txn_id = new_txn_id()
        is_return = random.random() < 0.08
        status = "delivered"

        device_id = f"DEV{c_idx:03d}"
        ip_address = fake_ip(c_idx)

        transactions.append({
            "transaction_id": txn_id,
            "seller_id": seller_id,
            "customer_id": customer_id,
            "delivery_partner_id": delivery_partner_id,
            "device_id": device_id,
            "ip_address": ip_address,
            "amount": random.randint(300, 3000),
            "status": status,
            "is_return": is_return,
        })

        deliveries.append({
            "transaction_id": txn_id,
            "proof_photo_provided": random.random() > 0.1,
            "gps_mismatch": random.random() < 0.05,
        })

# ---- Planted collusion rings ----
for r in range(NUM_RING_SELLERS):
    seller_id = f"S_RING{r:02d}"
    sellers.append({
        "seller_id": seller_id,
        "tenure_days": random.randint(10, 60),
        "size": "small",
        "signup_email": f"ring{r}@mailinator.com",
    })

    ring_customers = [f"C_RING{r}_{k}" for k in range(5)]
    ring_delivery = f"D_RING{r}"
    ring_device = f"DEV_RING{r}"
    ring_ip = f"10.10.10.{r}"

    for _ in range(8):
        customer_id = random.choice(ring_customers)
        txn_id = new_txn_id()
        status = "delivered"

        transactions.append({
            "transaction_id": txn_id,
            "seller_id": seller_id,
            "customer_id": customer_id,
            "delivery_partner_id": ring_delivery,
            "device_id": ring_device,
            "ip_address": ring_ip,
            "amount": random.randint(800, 1500),
            "status": status,
            "is_return": False,
        })

        deliveries.append({
            "transaction_id": txn_id,
            "proof_photo_provided": random.random() > 0.5,
            "gps_mismatch": random.random() < 0.3,
        })

# ---- Save ----
pd.DataFrame(transactions).to_csv(os.path.join(OUTPUT_DIR, "transactions.csv"), index=False)
pd.DataFrame(deliveries).to_csv(os.path.join(OUTPUT_DIR, "deliveries.csv"), index=False)
pd.DataFrame(sellers).to_csv(os.path.join(OUTPUT_DIR, "sellers.csv"), index=False)

print(f"Generated {len(sellers)} sellers, {len(transactions)} transactions, {len(deliveries)} deliveries.")
print(f"Saved to {OUTPUT_DIR}/")