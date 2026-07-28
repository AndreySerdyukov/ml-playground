import { Link, useParams } from "react-router-dom";
import { PredictPanel } from "../components/PredictPanel";
import { ModelIcon } from "../components/ModelIcon";
import type { ModelInfo } from "../api";

// Страница модели: шапка + форма признаков + результат.
export function ModelPage({ models }: { models: ModelInfo[] }) {
  const { name } = useParams();
  const model = models.find((m) => m.name === name);

  if (models.length === 0) {
    return (
      <div className="mx-auto max-w-content px-6 py-24 text-center text-slate">Loading…</div>
    );
  }
  if (!model) {
    return (
      <div className="mx-auto max-w-content px-6 py-24 text-center">
        <p className="text-slate">Model “{name}” not found.</p>
        <Link to="/" className="mt-4 inline-block text-link">
          ← Back to models
        </Link>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-content px-6 py-12 sm:py-16">
      <div className="mb-8 flex items-start gap-4">
        <ModelIcon name={model.name} className="h-11 w-11 shrink-0 text-ink" />
        <div>
          <h1 className="text-3xl font-semibold capitalize tracking-tight text-ink sm:text-4xl">
            {model.name}
          </h1>
          <p className="mt-1 text-sm text-slate">
            {model.category} · predicts {model.target}
            {model.is_stub && (
              <span className="ml-2 rounded-full bg-mist px-2 py-0.5 text-[11px] text-slate">
                demo weights
              </span>
            )}
          </p>
          <p className="mt-3 max-w-2xl text-[15px] text-slate">{model.description}</p>
        </div>
      </div>

      <PredictPanel key={model.name} model={model} />
    </div>
  );
}
