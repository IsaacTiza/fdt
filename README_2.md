# Fraud Detection System for Electronic Banking Transactions

A machine-learning fraud detection system: a FastAPI backend serving a trained Decision Tree (one scikit-learn Pipeline), MongoDB Atlas for persistence, and a React/TypeScript dashboard for monitoring and testing predictions. The model is trained on the PaySim dataset.

**Project topic:** *Development of a Machine Learning-Based Fraud Detection System for Electronic Banking Transactions*

## Quick start
```cmd
:: 1) Place the PaySim CSV in backend\data\ (or add --data <path> to the train command)
cd backend & venv\Scripts\activate
python -m app.models.train_model
:: 2) Terminal 1 (from backend\, venv active): start the API
uvicorn app.main:app --reload --port 8000
:: 3) Terminal 2 (from frontend\): start the UI
npm run dev
:: 4) Optional, after training (from backend\, venv active): seed demo data
python seed_data.py
```

## Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI, Motor (async MongoDB driver), scikit-learn |
| ML | Decision Tree (depth-capped, `class_weight="balanced"`) in a single Pipeline with one-hot encoding |
| Dataset | PaySim (synthetic mobile-money transactions) |
| Database | MongoDB Atlas |
| Frontend | React, Vite, TypeScript, Tailwind CSS, Recharts |

## Repository structure

```
backend/
  app/
    config.py            # Environment loading
    db/mongo.py          # MongoDB client + connection check
    dependencies.py      # Static API key auth
    main.py              # FastAPI app entrypoint
    models/
      paysim.py          # Shared data contract: columns, types, default data path, CSV reader
      schemas.py         # Pydantic request/response contracts
      train_model.py     # Training script (Decision Tree pipeline)
    routers/
      predict.py         # POST /api/v1/predict
      analytics.py       # /summary, /model-performance, /trend
    services/
      ml_service.py      # Model loading, thresholds, inference, decision-path reasons
  seed_data.py           # Populates DB with sampled real PaySim rows (manual, after training)
  test_predict.py        # Manual spot-check of the trained model (manual, after training)
  check_db.py            # Diagnostic script for DB record counts
  data/                  # Put the PaySim CSV here (git-ignored)
  artifacts/             # Trained model_v<UTC>.joblib files (git-ignored)
  requirements.txt

frontend/
  src/
    api/client.ts        # Typed fetch wrapper for the backend API
    components/          # Shared UI: Layout, StatCard, VerdictBadge
    pages/               # Dashboard, Predict, ModelPerformance, Trend
    types/index.ts       # TypeScript types mirroring backend schemas

setup_all.bat            # One-command setup for a fresh machine
SYSTEM_DOCUMENTATION.md  # Data, model, thresholds, limitations
```

## Setup

Run `setup_all.bat` from the repo root on a machine with Python 3.10+ and Node.js 18+. It creates the backend virtual environment, installs dependencies, and prepares `.env` files from the examples. On the first run only (it also creates `venv`). Then edit both env files with matching credentials, or copy your existing `.env` files into `backend/` and `frontend/`:
- `backend/.env` -> `MONGO_URI`, `DB_NAME`, `API_KEY`
- `frontend/.env` -> `VITE_API_URL` (backend address), `VITE_API_KEY` (must match the backend's `API_KEY`)

## Behaviour to know about

- **No model yet:** the API starts, and `/api/v1/predict` returns `503 "No trained model. Run python -m app.models.train_model"`. After training, restart the API (a model trained while the API runs is also picked up on the next prediction request if none was loaded).
- **Model inputs:** `type`, `amount`, sender balance before/after, receiver balance before/after. `location` and `device_id` are optional metadata that are stored but never given to the model.
- **Response:** `{transaction_id, fraud_probability, action, reasons}`. `reasons` lists the decision-tree rules that produced the prediction.
- **Thresholds:** block >= 0.85, review >= 0.40, otherwise allow. These are placeholders that must be re-tuned after the first training run (the training script prints precision/recall at both cutoffs). See `SYSTEM_DOCUMENTATION.md`.
- **Tuning the tree:** `MAX_DEPTH` and `MIN_SAMPLES_LEAF` are constants at the top of `backend/app/models/train_model.py`.
- **Authentication** is a static shared API key. Explicitly non-production: no rotation, no per-user scoping, no rate limiting.

## Known limitations

See `SYSTEM_DOCUMENTATION.md` section 7 for the full list, including synthetic data, untuned thresholds, seed data distribution, and the auth model.
