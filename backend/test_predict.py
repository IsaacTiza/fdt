"""
Manual check of the trained model against real PaySim rows.
Run AFTER training (never runs automatically, nothing here is collected by pytest):

    python test_predict.py
    python test_predict.py --data D:\\path\\to\\PS_20174392719_1491204439457_log.csv --n 5
"""

import argparse

from app.models.paysim import TARGET_COLUMN, add_data_arg, read_paysim
from app.services.ml_service import load_model, predict_single


def check_row(row) -> bool:
    fields = {
        "type": str(row["type"]),
        "amount": float(row["amount"]),
        "oldbalanceOrg": float(row["oldbalanceOrg"]),
        "newbalanceOrig": float(row["newbalanceOrig"]),
        "oldbalanceDest": float(row["oldbalanceDest"]),
        "newbalanceDest": float(row["newbalanceDest"]),
    }
    p = predict_single(fields)
    actual = int(row[TARGET_COLUMN])
    print(f"{fields['type']:<9} amount={fields['amount']:>14,.2f} | "
          f"p={p.probability:.4f} | {p.action:<6} | actual isFraud={actual}")
    print(f"          why: {'; '.join(p.reasons) or '(no rules on path)'}")
    return (p.action != "allow") == bool(actual)


def main():
    parser = argparse.ArgumentParser(description="Spot-check the trained model on PaySim rows.")
    add_data_arg(parser)
    parser.add_argument("--n", type=int, default=5, help="rows per class to test (default 5)")
    args = parser.parse_args()

    if load_model() is None:
        raise SystemExit("[ERROR] No trained model. Run python -m app.models.train_model")
    try:
        df = read_paysim(args.data)
    except (FileNotFoundError, ValueError) as err:
        raise SystemExit(f"[ERROR] {err}")

    fraud = df[df[TARGET_COLUMN] == 1].sample(n=args.n)
    legit = df[df[TARGET_COLUMN] == 0].sample(n=args.n)
    print("--- real fraud rows ---")
    caught = sum(check_row(r) for _, r in fraud.iterrows())
    print("--- real legitimate rows ---")
    passed = sum(check_row(r) for _, r in legit.iterrows())
    print(f"\nFraud rows flagged (review/block): {caught}/{args.n}")
    print(f"Legit rows allowed:                {passed}/{args.n}")


if __name__ == "__main__":
    main()
