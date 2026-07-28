import { useState, type FormEvent } from "react";
import { predict, type ModelInfo, type PredictResponse } from "../api";
import { Combobox } from "./Combobox";
import { ResultCard } from "./ResultCard";

// Значения по умолчанию из примеров модели (форма приходит предзаполненной).
function initialValues(model: ModelInfo): Record<string, string> {
  const values: Record<string, string> = {};
  for (const f of model.features) {
    values[f.name] = f.example != null ? String(f.example) : "";
  }
  return values;
}

// Одиночный предикт: форма признаков + карточка результата. Ремоунтится по key={model.name}.
export function FeatureForm({ model }: { model: ModelInfo }) {
  const [values, setValues] = useState<Record<string, string>>(() => initialValues(model));
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [submitted, setSubmitted] = useState<Record<string, string> | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  function set(name: string, value: string): void {
    setValues((prev) => ({ ...prev, [name]: value }));
  }

  async function onSubmit(event: FormEvent): Promise<void> {
    event.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const features: Record<string, unknown> = {};
      for (const f of model.features) {
        const raw = values[f.name] ?? "";
        features[f.name] = f.type === "number" ? Number(raw) : raw;
      }
      const res = await predict(model.name, features);
      setResult(res);
      setSubmitted({ ...values });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setResult(null);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="grid gap-8 lg:grid-cols-[1fr_360px]">
      <form onSubmit={onSubmit} className="rounded-apple border border-hair p-6 sm:p-8">
        <div className="grid gap-5 sm:grid-cols-2">
          {model.features.map((f) => (
            <div key={f.name} className="flex flex-col">
              <span className="mb-1.5 text-sm text-slate">
                {f.label ?? f.name}
                {f.unit ? `, ${f.unit}` : ""}
              </span>
              {f.type === "category" && f.choices ? (
                <Combobox
                  value={values[f.name] ?? ""}
                  onChange={(v) => set(f.name, v)}
                  options={f.choices}
                />
              ) : (
                <input
                  type="number"
                  step="any"
                  className="field"
                  value={values[f.name] ?? ""}
                  onChange={(e) => set(f.name, e.target.value)}
                  required
                />
              )}
            </div>
          ))}
        </div>

        <div className="mt-7 flex items-center gap-5">
          <button type="submit" disabled={loading} className="btn-primary">
            {loading ? "Predicting…" : "Predict"}
          </button>
          <button
            type="button"
            onClick={() => setValues(initialValues(model))}
            className="text-[15px] text-link"
          >
            Try an example
          </button>
        </div>

        {error && <p className="mt-4 text-sm text-red-600">{error}</p>}
      </form>

      <ResultCard model={model} result={result} inputs={submitted} />
    </div>
  );
}
