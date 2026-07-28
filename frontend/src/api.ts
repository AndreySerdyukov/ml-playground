// Thin client for the backend: types mirror the service's pydantic schemas.

export type FeatureSpec = {
  name: string;
  type: "number" | "category";
  choices?: string[] | null;
  label?: string | null;
  unit?: string | null;
  example?: number | string | null;
};

export type ThresholdPoint = {
  threshold: number;
  precision: number;
  recall: number;
};

export type ModelInfo = {
  name: string;
  task: "regression" | "classification";
  target: string;
  description: string;
  features: FeatureSpec[];
  emoji: string;
  category: string;
  target_unit?: string | null;
  typical_error_pct?: number | null;
  is_stub: boolean;
  // Binary classification only: interactive threshold in the UI.
  positive_class?: string | null;
  default_threshold?: number | null;
  threshold_curve?: ThresholdPoint[] | null;
};

export type PredictResponse = {
  model_name: string;
  task: "regression" | "classification";
  prediction: number | string;
  probabilities?: Record<string, number> | null;
  // For uplift: outcome probabilities under the two treatment scenarios (with_treatment / without_treatment).
  scenarios?: Record<string, number> | null;
};

export type BatchPredictResponse = {
  model_name: string;
  task: "regression" | "classification";
  target: string;
  target_unit?: string | null;
  columns: string[];
  rows: Record<string, unknown>[];
  count: number;
};

const BASE = import.meta.env.VITE_API_URL ?? "";

export async function fetchModels(): Promise<ModelInfo[]> {
  const res = await fetch(`${BASE}/api/models`);
  if (!res.ok) throw new Error(`Failed to load models (${res.status})`);
  return res.json();
}

export async function predict(
  modelName: string,
  features: Record<string, unknown>,
): Promise<PredictResponse> {
  const res = await fetch(`${BASE}/api/models/${modelName}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ features }),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail.detail ?? `Prediction failed (${res.status})`);
  }
  return res.json();
}

// Batch predict: upload a CSV/Excel file, get a prediction for each row.
export async function predictBatch(
  modelName: string,
  file: File,
): Promise<BatchPredictResponse> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BASE}/api/models/${modelName}/predict-batch`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail.detail ?? `Batch prediction failed (${res.status})`);
  }
  return res.json();
}
