import { Hero } from "../components/Hero";
import { ModelCard } from "../components/ModelCard";
import type { ModelInfo } from "../api";

// Главная: хиро + сетка карточек моделей.
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
      <section className="mx-auto max-w-content px-6 pb-24">
        {error && <p className="text-center text-sm text-red-600">{error}</p>}
        {loading && <p className="text-center text-sm text-slate">Loading models…</p>}
        <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-3">
          {models.map((m) => (
            <ModelCard key={m.name} model={m} />
          ))}
        </div>
      </section>
    </>
  );
}
