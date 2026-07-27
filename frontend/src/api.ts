// Тонкий клиент к backend: типы совпадают с pydantic-схемами сервиса.

export type FeatureSpec = {
  name: string;
  type: "number" | "category";
  choices?: string[] | null;
  label?: string | null;
  unit?: string | null;
  example?: number | string | null;
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
  is_stub: boolean;
};

export type PredictResponse = {
  model_name: string;
  task: "regression" | "classification";
  prediction: number | string;
  probabilities?: Record<string, number> | null;
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
