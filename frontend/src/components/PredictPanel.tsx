import { useState } from "react";
import type { ModelInfo } from "../api";
import { BatchPredict } from "./BatchPredict";
import { FeatureForm } from "./FeatureForm";

// Prediction mode toggle: single input or a batch from a file.
export function PredictPanel({ model }: { model: ModelInfo }) {
  const [mode, setMode] = useState<"single" | "batch">("single");
  const tabs = [
    { id: "single", label: "Single" },
    { id: "batch", label: "Batch (CSV/Excel)" },
  ] as const;

  return (
    <div>
      <div className="mb-6 inline-flex rounded-full bg-mist p-1 text-sm">
        {tabs.map((t) => (
          <button
            key={t.id}
            type="button"
            onClick={() => setMode(t.id)}
            className={`rounded-full px-4 py-1.5 transition ${
              mode === t.id ? "bg-surface text-ink shadow-sm" : "text-slate hover:text-ink"
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {mode === "single" ? <FeatureForm model={model} /> : <BatchPredict model={model} />}
    </div>
  );
}
