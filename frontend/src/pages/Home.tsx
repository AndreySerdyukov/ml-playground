import { Hero } from "../components/Hero";
import { ModelCard } from "../components/ModelCard";
import { ModelIcon } from "../components/ModelIcon";
import type { ModelInfo } from "../api";

// Главная: хиро + сетка карточек моделей + секция фич.
export function Home({
  models,
  loading,
  error,
}: {
  models: ModelInfo[];
  loading: boolean;
  error: string | null;
}) {
  return (
    <>
      <Hero />

      <section className="mx-auto max-w-content px-6 pb-20">
        {error && <p className="text-center text-sm text-red-600">{error}</p>}
        {loading && <p className="text-center text-sm text-slate">Loading models…</p>}
        <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-3">
          {models.map((m) => (
            <ModelCard key={m.name} model={m} />
          ))}
        </div>
      </section>

      <section className="border-t border-hair">
        <div className="mx-auto grid max-w-content gap-12 px-6 py-16 text-center sm:grid-cols-3">
          <Feature
            icon="model-agnostic-forms"
            title="Model-agnostic"
            text="Every input form builds itself from the model's own feature spec."
          />
          <Feature
            icon="instant-prediction"
            title="Instant predictions"
            text="Enter values, get an answer right away, straight in the browser."
          />
          <Feature
            icon="notebook-to-browser"
            title="Notebook to browser"
            text="Trained models served behind one clean, versionable API."
          />
        </div>
      </section>
    </>
  );
}

function Feature({ icon, title, text }: { icon: string; title: string; text: string }) {
  return (
    <div className="flex flex-col items-center">
      <ModelIcon name={icon} className="h-8 w-8 text-ink" />
      <h3 className="mt-4 text-lg font-semibold text-ink">{title}</h3>
      <p className="mt-1.5 max-w-xs text-sm text-slate">{text}</p>
    </div>
  );
}
