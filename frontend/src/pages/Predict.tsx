import { useState } from "react";
import VerdictBadge from "../components/VerdictBadge";
import { predictTransaction, ApiError } from "../api/client";
import { TRANSACTION_TYPES } from "../types";
import type { ActionTier, PredictResponse, TransactionInput, TransactionType } from "../types";

type NumericField = "amount" | "oldbalanceOrg" | "newbalanceOrig" | "oldbalanceDest" | "newbalanceDest";

// Form values stay as strings so a blank field is "blank", never silently 0.
interface FormState {
  type: TransactionType | "";
  amount: string;
  oldbalanceOrg: string;
  newbalanceOrig: string;
  oldbalanceDest: string;
  newbalanceDest: string;
}

type FormErrors = Partial<Record<keyof FormState, string>>;

const emptyForm: FormState = {
  type: "",
  amount: "",
  oldbalanceOrg: "",
  newbalanceOrig: "",
  oldbalanceDest: "",
  newbalanceDest: "",
};

const TYPE_LABELS: Record<TransactionType, string> = {
  PAYMENT: "Payment (pay a merchant)",
  TRANSFER: "Transfer (to another account)",
  CASH_OUT: "Cash out (withdraw cash)",
  DEBIT: "Debit (send to a bank)",
  CASH_IN: "Cash in (deposit cash)",
};

const NUMERIC_FIELDS: { name: NumericField; label: string; help: string }[] = [
  { name: "amount", label: "Amount", help: "How much money is being moved." },
  { name: "oldbalanceOrg", label: "Sender balance before", help: "Sender's account balance before the transaction." },
  { name: "newbalanceOrig", label: "Sender balance after", help: "Sender's account balance after the transaction." },
  {
    name: "oldbalanceDest",
    label: "Receiver balance before",
    help: "Receiver's balance before. Enter 0 for merchants (e.g. Payment); the dataset records no balance for them.",
  },
  {
    name: "newbalanceDest",
    label: "Receiver balance after",
    help: "Receiver's balance after. Enter 0 for merchants (e.g. Payment).",
  },
];

/**
 * Example inputs for one-click testing. These are illustrative values only. They are not
 * rows from the dataset, and no verdict is claimed for them: the verdict comes from the
 * trained model.
 */
const SAMPLES: { label: string; description: string; data: FormState }[] = [
  {
    label: "Small payment",
    description: "A small PAYMENT from a well-funded account.",
    data: {
      type: "PAYMENT", amount: "150", oldbalanceOrg: "5000", newbalanceOrig: "4850",
      oldbalanceDest: "0", newbalanceDest: "0",
    },
  },
  {
    label: "Mid-size cash in",
    description: "A CASH_IN deposit that tops up the sender's balance.",
    data: {
      type: "CASH_IN", amount: "20000", oldbalanceOrg: "30000", newbalanceOrig: "50000",
      oldbalanceDest: "400000", newbalanceDest: "380000",
    },
  },
  {
    label: "Large transfer, sender emptied",
    description: "A large TRANSFER that leaves the sender account at zero.",
    data: {
      type: "TRANSFER", amount: "1500000", oldbalanceOrg: "1500000", newbalanceOrig: "0",
      oldbalanceDest: "0", newbalanceDest: "1500000",
    },
  },
];

const TIER_EXPLANATION: Record<ActionTier, string> = {
  allow: "Looks like normal activity, so the transaction can go through.",
  review: "Some risk signals were found, so a person should check this before it goes through.",
  block: "Strong fraud signals were found, so the transaction should be stopped.",
};

const TIER_BAR: Record<ActionTier, string> = {
  allow: "bg-tier-allow",
  review: "bg-tier-review",
  block: "bg-tier-block",
};

function validate(form: FormState): FormErrors {
  const errors: FormErrors = {};
  if (form.type === "") errors.type = "Choose a transaction type.";
  for (const { name } of NUMERIC_FIELDS) {
    const raw = form[name].trim();
    if (raw === "") errors[name] = "Required. Enter 0 if the value really is zero.";
    else if (!Number.isFinite(Number(raw))) errors[name] = "Enter a valid number.";
    else if (Number(raw) < 0) errors[name] = "Must be 0 or more.";
  }
  return errors;
}

function formatPercent(p: number): string {
  const pct = p * 100;
  if (pct > 0 && pct < 0.1) return "<0.1%";
  return `${pct.toFixed(1)}%`;
}

function friendlyError(err: unknown): string {
  if (err instanceof ApiError) {
    if (err.status === 503)
      return "The fraud model has not been trained yet. Train it first (python -m app.models.train_model), restart the backend, then try again.";
    if (err.status === 401) return "The API key was rejected. Check that VITE_API_KEY matches the backend's API_KEY.";
    if (err.status === 422) return "The server rejected these values. Check every field is a non-negative number.";
    return err.message;
  }
  if (err instanceof TypeError) return "Could not reach the backend. Check that it is running and VITE_API_URL is correct.";
  return err instanceof Error ? err.message : "Prediction failed";
}

export default function Predict() {
  const [form, setForm] = useState<FormState>(emptyForm);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [submitted, setSubmitted] = useState(false);

  const errors: FormErrors = submitted ? validate(form) : {};

  function handleChange(field: keyof FormState, value: string) {
    setForm((prev) => ({ ...prev, [field]: value }));
  }

  function loadSample(data: FormState) {
    setForm(data);
    setResult(null);
    setError(null);
    setSubmitted(false);
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setSubmitted(true);
    setError(null);
    setResult(null);
    if (Object.keys(validate(form)).length > 0 || form.type === "") return;

    const payload: TransactionInput = {
      type: form.type,
      amount: Number(form.amount),
      oldbalanceOrg: Number(form.oldbalanceOrg),
      newbalanceOrig: Number(form.newbalanceOrig),
      oldbalanceDest: Number(form.oldbalanceDest),
      newbalanceDest: Number(form.newbalanceDest),
    };
    setSubmitting(true);
    try {
      setResult(await predictTransaction(payload));
    } catch (err) {
      setError(friendlyError(err));
    } finally {
      setSubmitting(false);
    }
  }

  const inputClass =
    "border border-gray-200 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-primary/30";

  return (
    <div className="flex flex-col gap-6 max-w-3xl">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Predict</h1>
        <p className="text-sm text-gray-500">
          Load an example or enter a transaction yourself, then run a prediction.
        </p>
      </div>

      <div className="bg-surface rounded-xl border border-gray-100 shadow-sm p-6">
        <h2 className="font-semibold text-gray-900 mb-1">Example inputs</h2>
        <p className="text-sm text-gray-500 mb-4">
          Click a card to fill the form. These are illustrative examples; the actual verdict comes from the
          trained model.
        </p>
        <div className="grid grid-cols-3 gap-3">
          {SAMPLES.map((sample) => (
            <button
              key={sample.label}
              type="button"
              onClick={() => loadSample(sample.data)}
              className="text-left border border-gray-200 rounded-lg p-4 hover:border-primary hover:bg-primary/5 transition-colors"
            >
              <div className="text-[10px] font-semibold tracking-wide text-gray-400 mb-1">EXAMPLE INPUT</div>
              <div className="font-semibold text-sm text-gray-900">{sample.label}</div>
              <div className="text-xs text-gray-500 mt-1">{sample.description}</div>
              <div className="text-xs text-gray-400 mt-2">
                Amount: {Number(sample.data.amount).toLocaleString()}
              </div>
            </button>
          ))}
        </div>
      </div>

      <form
        onSubmit={handleSubmit}
        noValidate
        className="bg-surface rounded-xl border border-gray-100 shadow-sm p-6 flex flex-col gap-4"
      >
        <label className="flex flex-col gap-1 text-sm">
          <span className="font-medium text-gray-700">Transaction type</span>
          <select
            value={form.type}
            onChange={(e) => handleChange("type", e.target.value)}
            className={inputClass}
          >
            <option value="">Select a type…</option>
            {TRANSACTION_TYPES.map((t) => (
              <option key={t} value={t}>
                {TYPE_LABELS[t]}
              </option>
            ))}
          </select>
          <span className="text-xs text-gray-400">What kind of money movement this is.</span>
          {errors.type && <span className="text-xs text-tier-block">{errors.type}</span>}
        </label>

        <div className="grid grid-cols-2 gap-4">
          {NUMERIC_FIELDS.map(({ name, label, help }) => (
            <label key={name} className="flex flex-col gap-1 text-sm">
              <span className="font-medium text-gray-700">{label}</span>
              <input
                type="number"
                step="any"
                min="0"
                inputMode="decimal"
                value={form[name]}
                onChange={(e) => handleChange(name, e.target.value)}
                className={inputClass}
              />
              <span className="text-xs text-gray-400">{help}</span>
              {errors[name] && <span className="text-xs text-tier-block">{errors[name]}</span>}
            </label>
          ))}
        </div>

        <button
          type="submit"
          disabled={submitting}
          className="self-start bg-primary hover:bg-primary-hover text-white font-semibold px-5 py-2.5 rounded-lg transition-colors disabled:opacity-50"
        >
          {submitting ? "Predicting..." : "Run Prediction"}
        </button>
      </form>

      {error && <div className="text-tier-block text-sm">{error}</div>}

      {result && (
        <div className="bg-surface rounded-xl border border-gray-100 shadow-sm p-6 flex flex-col gap-4">
          <div className="flex items-center gap-4">
            <VerdictBadge action={result.action} />
            <span className="text-sm text-gray-600">
              Fraud probability: <strong>{formatPercent(result.fraud_probability)}</strong>
            </span>
          </div>

          <div
            className="h-2 w-full rounded-full bg-gray-100 overflow-hidden"
            role="progressbar"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={Math.round(result.fraud_probability * 100)}
          >
            <div
              className={`h-full rounded-full ${TIER_BAR[result.action]}`}
              style={{ width: `${Math.min(100, Math.max(0, result.fraud_probability * 100))}%` }}
            />
          </div>

          <p className="text-sm text-gray-700">{TIER_EXPLANATION[result.action]}</p>

          <div>
            <h3 className="font-semibold text-sm text-gray-900 mb-1">Why this result</h3>
            {result.reasons.length > 0 ? (
              <ul className="list-disc pl-5 text-sm text-gray-600 flex flex-col gap-0.5">
                {result.reasons.map((reason) => (
                  <li key={reason}>{reason}</li>
                ))}
              </ul>
            ) : (
              <p className="text-sm text-gray-500">No decision rules were reported for this prediction.</p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
