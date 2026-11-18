"""
Loads the most recently trained model artifact (a single sklearn Pipeline:
one-hot encoding for `type` + DecisionTreeClassifier) and exposes prediction
to the rest of the app. Picks the latest model by filename timestamp, so
retraining (train_model.py) and serving are decoupled: retrain, restart the
API, it picks up the new version.

If no model file exists the API still starts; predict_single raises
ModelNotLoadedError, which the router turns into a 503.
"""

from pathlib import Path
from typing import NamedTuple

import joblib
import pandas as pd

from app.models.paysim import ARTIFACTS_DIR, METADATA_FIELDS
from app.models.schemas import TransactionInput

# Three-tier decision boundaries on fraud_probability.
# TODO(after first training run): these two values are PLACEHOLDERS carried over from
# the previous system. They MUST be re-tuned against the test-set output of
# `python -m app.models.train_model` (it prints precision/recall at both cutoffs).
# A depth-capped tree with class_weight="balanced" yields coarse, non-calibrated
# probabilities, so the right cutoffs may differ a lot from these.
BLOCK_THRESHOLD = 0.85
REVIEW_THRESHOLD = 0.40

NO_MODEL_MESSAGE = "No trained model. Run python -m app.models.train_model"

# Friendly names for decision-path rules shown to the user.
_LABELS = {
    "amount": "amount",
    "oldbalanceOrg": "sender balance before",
    "newbalanceOrig": "sender balance after",
    "oldbalanceDest": "receiver balance before",
    "newbalanceDest": "receiver balance after",
}

_model = None
_model_version: str | None = None


class ModelNotLoadedError(RuntimeError):
    """Raised when no trained model artifact is available."""


class Prediction(NamedTuple):
    probability: float
    action: str
    reasons: list[str]


def _schema_model_fields() -> list[str]:
    return [f for f in TransactionInput.model_fields if f not in METADATA_FIELDS]


def _find_latest_model_path() -> Path | None:
    candidates = sorted(ARTIFACTS_DIR.glob("model_v*.joblib"))
    return candidates[-1] if candidates else None  # timestamp-named files sort chronologically


def _validate_model(model) -> None:
    """Fail loudly if the model's expected columns differ from the request schema."""
    expected = list(getattr(model, "feature_names_in_", []))
    schema_fields = _schema_model_fields()
    if set(expected) != set(schema_fields):
        raise RuntimeError(
            "Model/schema mismatch: the trained model expects columns "
            f"{expected} but the request schema provides {schema_fields}. "
            "Retrain with `python -m app.models.train_model` or fix schemas.py."
        )
    if not hasattr(model, "predict_proba") or not hasattr(model, "named_steps"):
        raise RuntimeError("Loaded artifact is not a fitted sklearn Pipeline with predict_proba.")
    if len(model.classes_) != 2:
        raise RuntimeError(f"Model must be binary, got classes {list(model.classes_)}.")


def load_model() -> str | None:
    """
    Load the latest model into memory. Returns its version, or None if no
    artifact exists (the API then starts anyway and /predict returns 503).
    Raises RuntimeError if an artifact exists but does not match the schema.
    """
    global _model, _model_version
    path = _find_latest_model_path()
    if path is None:
        print(f"[ml_service] WARNING: {NO_MODEL_MESSAGE} (looked in {ARTIFACTS_DIR})")
        return None
    model = joblib.load(path)
    _validate_model(model)
    _model = model
    _model_version = path.stem.replace("model_", "")
    print(f"[ml_service] Loaded model version: {_model_version} from {path.name}")
    return _model_version


def get_loaded_version() -> str:
    if _model_version is None:
        raise ModelNotLoadedError(NO_MODEL_MESSAGE)
    return _model_version


def _format_number(value: float) -> str:
    return f"{value:,.0f}" if abs(value) >= 100 else f"{value:,.2f}"


def _explain(model, X: pd.DataFrame) -> list[str]:
    """Decision-path rules for this single row, merged to the tightest bound per feature."""
    preprocess, tree = model[:-1], model[-1]
    Xt = preprocess.transform(X)
    names = list(preprocess.get_feature_names_out())
    t = tree.tree_

    lower: dict[str, float] = {}   # feature > value
    upper: dict[str, float] = {}   # feature <= value
    order: list[str] = []
    not_types: list[str] = []
    is_type: str | None = None

    for node in tree.decision_path(Xt).indices:
        if t.children_left[node] == t.children_right[node]:  # leaf
            continue
        name = names[t.feature[node]]
        threshold = float(t.threshold[node])
        went_left = float(Xt[0, t.feature[node]]) <= threshold
        if name.startswith("type_"):
            category = name[len("type_"):]
            if went_left:
                not_types.append(category)
            else:
                is_type = category
            continue
        if name not in order:
            order.append(name)
        if went_left:
            upper[name] = min(upper.get(name, threshold), threshold)
        else:
            lower[name] = max(lower.get(name, threshold), threshold)

    reasons: list[str] = []
    if is_type:
        reasons.append(f"type is {is_type}")
    elif not_types:
        reasons.append("type is not " + " or ".join(not_types))
    for name in order:
        label = _LABELS.get(name, name)
        if name in lower:
            reasons.append(f"{label} > {_format_number(lower[name])}")
        if name in upper:
            reasons.append(f"{label} <= {_format_number(upper[name])}")
    return reasons


def predict_single(features: dict) -> Prediction:
    """
    features: dict with the model's input fields (type, amount, oldbalanceOrg,
    newbalanceOrig, oldbalanceDest, newbalanceDest); extra keys are ignored.
    Raises ModelNotLoadedError if no model is available.
    """
    global _model
    if _model is None:
        load_model()  # pick up a model trained after the API started
    if _model is None:
        raise ModelNotLoadedError(NO_MODEL_MESSAGE)

    columns = list(_model.feature_names_in_)
    missing = [c for c in columns if c not in features]
    if missing:
        raise ValueError(f"Missing model input(s): {missing}")
    X = pd.DataFrame([[features[c] for c in columns]], columns=columns)

    probability = float(_model.predict_proba(X)[0, 1])
    return Prediction(probability, _decide_action(probability), _explain(_model, X))


def _decide_action(probability: float) -> str:
    if probability >= BLOCK_THRESHOLD:
        return "block"
    elif probability >= REVIEW_THRESHOLD:
        return "review"
    else:
        return "allow"
