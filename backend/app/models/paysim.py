"""
Single source of truth for the PaySim data contract.

Training, serving (schemas.py / ml_service.py), seed_data.py and test_predict.py
all import from here, so the model's input columns cannot drift between them.
This module must stay free of environment/database imports.
"""

from pathlib import Path
from typing import Literal, get_args

import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = BACKEND_DIR / "artifacts"
DEFAULT_DATA_PATH = BACKEND_DIR / "data" / "PS_20174392719_1491204439457_log.csv"

TransactionType = Literal["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]
TRANSACTION_TYPES: list[str] = list(get_args(TransactionType))

CATEGORICAL_COLUMNS = ["type"]
NUMERIC_COLUMNS = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
FEATURE_COLUMNS = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS
TARGET_COLUMN = "isFraud"

# Fields in the request that are logged but never fed to the model.
METADATA_FIELDS = {"location", "device_id"}


def resolve_data_path(path: str | Path | None) -> Path:
    """Return the dataset path, or raise a clear error saying where to put the file."""
    data_path = Path(path) if path else DEFAULT_DATA_PATH
    if not data_path.exists():
        raise FileNotFoundError(
            f"PaySim dataset not found at: {data_path}\n"
            f"Download it from Kaggle (PaySim synthetic financial dataset) and place the file at:\n"
            f"  {DEFAULT_DATA_PATH}\n"
            f"or pass a different location with --data <path>."
        )
    return data_path


def read_paysim(path: str | Path | None = None) -> pd.DataFrame:
    """
    Read only the columns the model needs, with memory-friendly dtypes
    (the full file has 6M+ rows). Raises ValueError on missing columns,
    unknown transaction types, or missing values.
    """
    data_path = resolve_data_path(path)
    wanted = FEATURE_COLUMNS + [TARGET_COLUMN]

    header = pd.read_csv(data_path, nrows=0).columns
    missing = [c for c in wanted if c not in header]
    if missing:
        raise ValueError(
            f"{data_path.name} is missing required columns: {missing}. "
            f"Expected PaySim columns, found: {list(header)}"
        )

    dtypes = {c: "float32" for c in NUMERIC_COLUMNS}
    dtypes["type"] = "category"
    dtypes[TARGET_COLUMN] = "int8"
    df = pd.read_csv(data_path, usecols=wanted, dtype=dtypes)[wanted]

    unknown = set(df["type"].cat.categories) - set(TRANSACTION_TYPES)
    if unknown:
        raise ValueError(f"Unknown transaction type(s) in data: {sorted(unknown)}")
    if df.isna().any().any():
        raise ValueError("Dataset contains missing values in model columns; clean it first.")
    return df


def add_data_arg(parser) -> None:
    parser.add_argument(
        "--data",
        default=None,
        help=f"Path to the PaySim CSV (default: {DEFAULT_DATA_PATH})",
    )
