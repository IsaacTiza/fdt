"""
Generates demo history by sampling real PaySim rows, backdating them across N days,
and running them through the real prediction logic so is_simulated=True records
exercise the same inference + persistence path as live traffic.

Run manually AFTER training (never runs automatically):
    python seed_data.py
    python seed_data.py --data D:\\path\\to\\PS_20174392719_1491204439457_log.csv

Fraud rows are DELIBERATELY oversampled (~10% of the seed) relative to PaySim's true
fraud rate (about 0.13%). A pure random sample of 400 rows would contain almost no
fraud, and the dashboard would look like the system detects nothing. Seed data
distribution != production traffic distribution; state this in the report.
Note: PaySim fraud only occurs in TRANSFER and CASH_OUT transactions.
"""

import argparse
import asyncio
import random
import uuid
from datetime import datetime, timedelta, timezone

import pandas as pd

from app.models.paysim import TARGET_COLUMN, add_data_arg, read_paysim
from app.services.ml_service import ModelNotLoadedError, load_model, predict_single
from app.db.mongo import transactions_collection

NUM_TRANSACTIONS = 400
NUM_FRAUD_SAMPLE = 40  # ~10% of seed data, visible across all action tiers
DAYS_BACK = 30

LOCATIONS = ["Lagos", "Abuja", "Port Harcourt", "Ibadan", "Kano", "London", "New York"]


async def clear_seeded_data():
    result = await transactions_collection.delete_many({"is_simulated": True})
    print(f"Cleared {result.deleted_count} previously seeded transactions.")


async def seed(data_path=None):
    if load_model() is None:
        raise ModelNotLoadedError("No trained model. Run python -m app.models.train_model")

    df = read_paysim(data_path)  # validates the file before anything is deleted
    await clear_seeded_data()

    fraud_sample = df[df[TARGET_COLUMN] == 1].sample(n=NUM_FRAUD_SAMPLE)
    normal_sample = df[df[TARGET_COLUMN] == 0].sample(n=NUM_TRANSACTIONS - NUM_FRAUD_SAMPLE)
    sample = pd.concat([fraud_sample, normal_sample]).sample(frac=1).reset_index(drop=True)

    records = []
    now = datetime.now(timezone.utc)

    for _, row in sample.iterrows():
        tx_type = str(row["type"])
        fields = {
            "type": tx_type,
            "amount": float(row["amount"]),
            "oldbalanceOrg": float(row["oldbalanceOrg"]),
            "newbalanceOrig": float(row["newbalanceOrig"]),
            "oldbalanceDest": float(row["oldbalanceDest"]),
            "newbalanceDest": float(row["newbalanceDest"]),
        }
        prediction = predict_single(fields)

        backdated_time = now - timedelta(
            days=random.uniform(0, DAYS_BACK),
            hours=random.uniform(0, 24),
        )
        records.append({
            "transaction_id": str(uuid.uuid4()),
            **fields,
            "location": random.choice(LOCATIONS),
            "device_id": f"device-{random.randint(1000, 9999)}",
            "fraud_probability": prediction.probability,
            "action": prediction.action,
            "timestamp": backdated_time,
            "is_simulated": True,
        })

    result = await transactions_collection.insert_many(records)
    print(f"Inserted {len(result.inserted_ids)} seeded transactions.")

    action_counts = {}
    for r in records:
        action_counts[r["action"]] = action_counts.get(r["action"], 0) + 1
    print(f"Action breakdown: {action_counts}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed demo transactions from PaySim.")
    add_data_arg(parser)
    args = parser.parse_args()
    try:
        asyncio.run(seed(args.data))
    except (FileNotFoundError, ValueError, ModelNotLoadedError) as err:
        raise SystemExit(f"[ERROR] {err}")
