"""
POST /api/v1/predict — the core inference endpoint.
Contract: returns {transaction_id, fraud_probability, action} plus an additive "reasons" list.
Returns 503 if no trained model exists yet.
"""

import uuid
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, status

from app.dependencies import verify_api_key
from app.models.schemas import TransactionInput, PredictionResponse
from app.services.ml_service import ModelNotLoadedError, predict_single
from app.db.mongo import transactions_collection

router = APIRouter(prefix="/api/v1", tags=["predict"])


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    transaction: TransactionInput,
    api_key: str = Depends(verify_api_key),
):
    features = transaction.model_dump()

    try:
        prediction = predict_single(features)
    except ModelNotLoadedError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        )
    transaction_id = str(uuid.uuid4())

    # Persist the full record — logged regardless of action taken,
    # so model_metrics/dashboard analytics have real data to aggregate against.
    record = {
        "transaction_id": transaction_id,
        "type": transaction.type,
        "amount": transaction.amount,
        "oldbalanceOrg": transaction.oldbalanceOrg,
        "newbalanceOrig": transaction.newbalanceOrig,
        "oldbalanceDest": transaction.oldbalanceDest,
        "newbalanceDest": transaction.newbalanceDest,
        "location": transaction.location,
        "device_id": transaction.device_id,
        "fraud_probability": prediction.probability,
        "action": prediction.action,
        "timestamp": datetime.now(timezone.utc),
        "is_simulated": False,
    }
    await transactions_collection.insert_one(record)

    return PredictionResponse(
        transaction_id=transaction_id,
        fraud_probability=prediction.probability,
        action=prediction.action,
        reasons=prediction.reasons,
    )
