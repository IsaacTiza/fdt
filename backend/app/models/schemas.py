from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime

from app.models.paysim import TransactionType


# ---------- /predict ----------

class TransactionInput(BaseModel):
    """
    What the client sends to POST /api/v1/predict.
    type + the four balances + amount are the model features (PaySim columns).
    Field names match the PaySim CSV and the trained pipeline's input columns;
    ml_service refuses to load a model whose columns differ from these fields.
    location/device_id are metadata only: NOT fed into the model, logged for
    dashboard/UI context.
    """
    type: TransactionType
    amount: float = Field(..., ge=0, description="Transaction amount")
    oldbalanceOrg: float = Field(..., ge=0, description="Sender balance before the transaction")
    newbalanceOrig: float = Field(..., ge=0, description="Sender balance after the transaction")
    oldbalanceDest: float = Field(..., ge=0, description="Receiver balance before the transaction")
    newbalanceDest: float = Field(..., ge=0, description="Receiver balance after the transaction")
    location: Optional[str] = None
    device_id: Optional[str] = None


class PredictionResponse(BaseModel):
    """What POST /api/v1/predict returns. The first three fields are the locked contract;
    `reasons` is additive (decision-path rules that produced this prediction)."""
    transaction_id: str
    fraud_probability: float
    action: Literal["allow", "review", "block"]
    reasons: list[str] = []


# ---------- transactions collection ----------

class TransactionRecord(BaseModel):
    """Full document shape stored in the transactions collection"""
    transaction_id: str
    type: TransactionType
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float
    location: Optional[str] = None
    device_id: Optional[str] = None
    fraud_probability: float
    action: Literal["allow", "review", "block"]
    timestamp: datetime
    is_simulated: bool = False


# ---------- model_metrics collection ----------

class ModelMetrics(BaseModel):
    """One document per training run"""
    model_version: str
    trained_at: datetime
    recall: float
    precision: float
    f1: float
    roc_auc: float
    feature_importances: dict[str, float]


# ---------- feedback collection ----------

class FeedbackInput(BaseModel):
    """Analyst verdict on a flagged transaction"""
    transaction_id: str
    verdict: Literal["confirmed_fraud", "false_positive"]
    analyst_note: Optional[str] = None


# ---------- analytics responses ----------

class AnalyticsSummary(BaseModel):
    total_transactions: int
    flagged_count: int
    fraud_rate: float
    avg_fraud_probability: float


class TrendPoint(BaseModel):
    date: str
    total: int
    flagged: int


class TrendResponse(BaseModel):
    days: int
    data: list[TrendPoint]