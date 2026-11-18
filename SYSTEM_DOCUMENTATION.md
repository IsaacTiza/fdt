# Fraud Detection System: Data & Model Documentation

## 1. What the system takes as input

Every prediction request carries six model inputs (PaySim column names) plus optional metadata:

| Field | Meaning |
|---|---|
| `type` | One of `PAYMENT`, `TRANSFER`, `CASH_OUT`, `DEBIT`, `CASH_IN`. |
| `amount` | Transaction amount (>= 0). |
| `oldbalanceOrg` / `newbalanceOrig` | Sender's balance before / after the transaction (>= 0). |
| `oldbalanceDest` / `newbalanceDest` | Receiver's balance before / after the transaction (>= 0). |
| `location`, `device_id` | Optional metadata. Stored with the transaction, **never given to the model**. |

## 2. The dataset

The model is trained on **PaySim** (`PS_20174392719_1491204439457_log.csv`), a *synthetic* dataset produced by a simulator calibrated on aggregated logs of a real mobile-money service. It has roughly 6.36 million rows, of which about 0.13% are fraud. Columns: `step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud`.

Things worth knowing about it:
- Fraud only occurs in `TRANSFER` and `CASH_OUT` transactions. The other three types never contain fraud, so the model will learn "these types are safe" from the data.
- Merchant receivers (`PAYMENT`, `DEBIT`) have no recorded balance, so their receiver balances are 0 in the data.
- Because it is simulated, the fraud pattern is cleaner than real-world fraud (see section 7).

## 3. Columns used and dropped

**Used:** `type`, `amount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`. Target: `isFraud`.

**Dropped, on purpose:**
- `nameOrig`, `nameDest`: account identifiers. A model that keys on them memorizes accounts rather than learning behaviour, and they do not exist for a new customer.
- `isFlaggedFraud`: the simulator's own rule output (large transfers), so using it would let the model copy a rule instead of learning from the data.
- `step`: the simulation clock (hour index). It describes where a row sits in the simulated month, not anything about the transaction itself.

## 4. Model and training methodology

- **Pipeline:** a single scikit-learn `Pipeline`: one-hot encoding of `type` (fixed category list), numeric columns passed through, then `DecisionTreeClassifier`. It is saved as **one joblib** (`backend/artifacts/model_v<UTC>.joblib`), so serving applies exactly the preprocessing that training used.
- **Why a Decision Tree:** the prediction can be read as a short list of rules, which is what the "Why this result" panel shows. It is also cheap to train on millions of rows.
- **Imbalance handling:** `class_weight="balanced"`. With 6M+ rows there is plenty of real fraud signal (about 8,000 fraud rows) and no need to generate synthetic rows. Class weighting changes how the tree is fitted but never touches the test set.
- **Overfitting controls:** `MAX_DEPTH` (default 8) and `MIN_SAMPLES_LEAF` (default 50), constants at the top of `train_model.py`.
- **Split:** a stratified 80/20 split with `random_state=42` happens *before* anything else, so the test set keeps the real class distribution and is used only for evaluation.
- **Reported metrics:** recall, precision, F1, ROC-AUC on the untouched test set, the confusion matrix, and the first levels of the tree as text. These are computed at the tree's default decision rule. The training script additionally prints precision/recall at the two serving thresholds.
- **Logging:** each run saves a versioned artifact and inserts one document into MongoDB `model_metrics` (`model_version, trained_at, recall, precision, f1, roc_auc, feature_importances`). Feature importances are summed per input column (all `type_*` one-hot columns roll up into `type`). If MongoDB is unreachable the model is still saved and a warning is printed.

No metric values are quoted in this document, because they come from your own training run.

## 5. Serving and the decision-path "reasons"

```
POST /api/v1/predict  (FastAPI + Pydantic validation: type must be one of the five, numbers >= 0)
        |
ml_service.predict_single()
  -> loads the latest model artifact; refuses to load one whose columns differ from the request schema
  -> fraud_probability = pipeline.predict_proba(...)[1]
  -> reasons = the rules on this row's path through the tree (merged to the tightest bound per feature)
        |
Three-tier decision:
  probability >= 0.85 -> "block"
  probability >= 0.40 -> "review"
  otherwise           -> "allow"
        |
Persisted to MongoDB (transactions collection: type, amount, four balances, metadata, probability, action)
        |
Analytics endpoints: /summary, /model-performance, /trend  ->  React frontend
```

If no model file exists, `/predict` returns `503 "No trained model. Run python -m app.models.train_model"`. The `reasons` list explains which tree rules fired for this input (for example "amount > 181,000"). It describes the model's logic, not proven real-world causes.

## 6. Thresholds

`BLOCK_THRESHOLD = 0.85` and `REVIEW_THRESHOLD = 0.40` (named constants in `ml_service.py`) are **placeholders** carried over from an earlier version of the project. They must be re-tuned after the first training run:
- A depth-capped tree with balanced class weights produces coarse probabilities (many rows share the same leaf value), and they are not calibrated. Cutoffs that make sense for other models may put almost everything in one tier here.
- Use the printed precision/recall at both cutoffs, and ideally a threshold sweep, then pick values that reflect the cost of wrongly blocking a legitimate customer versus missing real fraud.

## 7. Named limitations (state these in the report)

- **Synthetic data.** PaySim is simulated. Scores on it are likely to be much higher than the same approach would achieve on real bank data, and should be presented as a demonstration of the pipeline, not a production accuracy claim.
- **Single-transaction view.** The model sees one transaction. It has no customer history, velocity, device, or location features.
- **Simulator artefacts.** Balance-based signals (for example, a sender account emptied to 0) are partly a property of how PaySim generates fraud; real fraudsters behave less uniformly.
- **Thresholds untuned** (section 6).
- **Seed data is deliberately oversampled** for fraud (about 10% versus the true ~0.13%) so all three tiers show up in the demo dashboard. It does not reflect production traffic.
- **Static API key** authentication: no rotation, no per-user scoping, no rate limiting. Explicitly non-production.
- **MongoDB Atlas IP allow-list:** confirm and state the current access policy (open or restricted) as a named trade-off.
- **Feedback endpoint:** a `FeedbackInput` schema and a `feedback` collection exist, but no endpoint is implemented. State it as descoped, or implement it before submission.
- **Model updates need a restart:** the API loads the newest artifact at startup (or on the first prediction if none was loaded).

## 8. Future work

- Train on real labelled bank data with history and velocity features.
- Calibrate probabilities and choose thresholds from a precision-recall sweep with an explicit cost model.
- Compare against other models on the same split and report the comparison.
