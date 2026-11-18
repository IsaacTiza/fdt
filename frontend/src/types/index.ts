export const TRANSACTION_TYPES = ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"] as const;
export type TransactionType = (typeof TRANSACTION_TYPES)[number];

// Mirrors app/models/schemas.py TransactionInput (PaySim column names).
export interface TransactionInput {
  type: TransactionType;
  amount: number;
  oldbalanceOrg: number;   // sender balance before
  newbalanceOrig: number;  // sender balance after
  oldbalanceDest: number;  // receiver balance before
  newbalanceDest: number;  // receiver balance after
}

export type ActionTier = "allow" | "review" | "block";

export interface PredictResponse {
  transaction_id: string;
  fraud_probability: number;
  action: ActionTier;
  reasons: string[];
}

export interface SummaryResponse {
  total_transactions: number;
  flagged_count: number;
  fraud_rate: number;
  avg_fraud_probability: number;
}

export interface ModelPerformanceResponse {
  model_version: string;
  trained_at: string;
  recall: number;
  precision: number;
  f1: number;
  roc_auc: number;
  feature_importances: Record<string, number>;
}

// CORRECTED: backend returns { days, data: TrendPoint[] }, not a bare array.
export interface TrendPoint {
  date: string;
  total: number;
  flagged: number;
}

export interface TrendResponse {
  days: number;
  data: TrendPoint[];
}
