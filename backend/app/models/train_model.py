"""
Offline training script, decoupled from the FastAPI serving layer.

    python -m app.models.train_model
    python -m app.models.train_model --data D:\\path\\to\\PS_20174392719_1491204439457_log.csv

Trains ONE sklearn Pipeline (one-hot for `type` + DecisionTreeClassifier) on PaySim and
saves it as a single versioned joblib, so serving cannot drift from training.
"""

import argparse
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, export_text

from app.models.paysim import (
    ARTIFACTS_DIR, CATEGORICAL_COLUMNS, FEATURE_COLUMNS, NUMERIC_COLUMNS,
    TARGET_COLUMN, TRANSACTION_TYPES, add_data_arg, read_paysim,
)
from app.services.ml_service import BLOCK_THRESHOLD, REVIEW_THRESHOLD

# ---- Tune these ------------------------------------------------------------
MAX_DEPTH = 8            # cap tree depth: smaller = simpler, more explainable, less overfit
MIN_SAMPLES_LEAF = 50    # every leaf must hold at least this many training rows
# ----------------------------------------------------------------------------
TEST_SIZE = 0.2
RANDOM_STATE = 42
TREE_PRINT_DEPTH = 4     # depth shown in the printed rules (display only)


def build_pipeline() -> Pipeline:
    preprocess = ColumnTransformer(
        [
            ("type", OneHotEncoder(categories=[TRANSACTION_TYPES], dtype=np.float32), CATEGORICAL_COLUMNS),
            ("num", "passthrough", NUMERIC_COLUMNS),
        ],
        verbose_feature_names_out=False,
    )
    tree = DecisionTreeClassifier(
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    return Pipeline([("preprocess", preprocess), ("tree", tree)])


def _threshold_report(y_test, y_proba) -> None:
    """Precision/recall at the serving thresholds, so they can be re-tuned."""
    print("\nAt the serving thresholds (ml_service.py):")
    for label, cutoff in (("review or block", REVIEW_THRESHOLD), ("block only", BLOCK_THRESHOLD)):
        pred = y_proba >= cutoff
        print(f"  {label:<16} (p >= {cutoff:.2f}): flagged={int(pred.sum()):>8}  "
              f"recall={recall_score(y_test, pred, zero_division=0):.4f}  "
              f"precision={precision_score(y_test, pred, zero_division=0):.4f}")


def train_and_evaluate(data_path=None) -> str:
    df = read_paysim(data_path)  # raises a clear error if the file is missing
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    if y.nunique() < 2 or y.value_counts().min() < 2:
        raise ValueError("Need at least 2 fraud and 2 non-fraud rows to train and split.")

    # Stratified split FIRST, before anything else touches the data. The test set keeps
    # the real class distribution and is never resampled or used for fitting.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    del df, X, y
    print(f"Train set: {len(X_train):,} rows, fraud rate: {y_train.mean():.4%}")
    print(f"Test set:  {len(X_test):,} rows, fraud rate: {y_test.mean():.4%}")
    print(f"Tree: max_depth={MAX_DEPTH}, min_samples_leaf={MIN_SAMPLES_LEAF}, class_weight=balanced")

    model = build_pipeline()
    model.fit(X_train, y_train)

    # Evaluate on the untouched, real-distribution test set
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    recall = float(recall_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_test, y_proba))

    print(f"\nRecall:    {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"ROC AUC:   {roc_auc:.4f}")

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    print("\nConfusion matrix (rows = actual, cols = predicted):")
    print(f"                 pred legit   pred fraud")
    print(f"  actual legit   {tn:>10,}   {fp:>10,}")
    print(f"  actual fraud   {fn:>10,}   {tp:>10,}")

    _threshold_report(y_test, y_proba)

    preprocess, tree = model.named_steps["preprocess"], model.named_steps["tree"]
    encoded_names = list(preprocess.get_feature_names_out())
    print(f"\nDecision tree rules (first {TREE_PRINT_DEPTH} levels):")
    print(export_text(tree, feature_names=encoded_names, max_depth=TREE_PRINT_DEPTH))

    # Roll the one-hot columns (type_*) back up to the model's input columns
    feature_importances = {c: 0.0 for c in FEATURE_COLUMNS}
    for name, value in zip(encoded_names, tree.feature_importances_.tolist()):
        feature_importances["type" if name.startswith("type_") else name] += value

    # Version = UTC timestamp, shared by artifact filename and metrics document.
    model_version = datetime.now(timezone.utc).strftime("v%Y%m%d_%H%M%S")
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    model_path = ARTIFACTS_DIR / f"model_{model_version}.joblib"
    joblib.dump(model, model_path)
    print(f"Model saved: {model_path}")

    log_metrics(model_version, recall, precision, f1, roc_auc, feature_importances)
    return model_version


def log_metrics(model_version, recall, precision, f1, roc_auc, feature_importances) -> None:
    """Insert one document into MongoDB model_metrics. A DB failure does not lose the model."""
    try:
        from pymongo import MongoClient
        from app.config import settings

        client = MongoClient(settings.MONGO_URI, serverSelectionTimeoutMS=10000)
        try:
            client[settings.DB_NAME]["model_metrics"].insert_one({
                "model_version": model_version,
                "trained_at": datetime.now(timezone.utc),
                "recall": recall,
                "precision": precision,
                "f1": f1,
                "roc_auc": roc_auc,
                "feature_importances": feature_importances,
            })
        finally:
            client.close()
        print(f"Metrics logged to model_metrics collection, version {model_version}")
    except Exception as exc:  # noqa: BLE001 - report any DB/config failure, keep the saved model
        print(f"\n[WARNING] Model saved, but metrics were NOT written to MongoDB: {exc}")
        print("Fix backend/.env / network access and re-run training to log metrics.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the PaySim fraud decision tree.")
    add_data_arg(parser)
    args = parser.parse_args()
    try:
        train_and_evaluate(args.data)
    except (FileNotFoundError, ValueError) as err:
        raise SystemExit(f"\n[ERROR] {err}")
