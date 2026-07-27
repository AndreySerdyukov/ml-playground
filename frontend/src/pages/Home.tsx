import type { ReactNode } from "react";
import { Hero } from "../components/Hero";
import { ModelCard } from "../components/ModelCard";
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
            title="Model-agnostic"
            text="Every input form builds itself from the model's own feature spec."
            icon={
              <>
                <path d="M4 7h16M4 12h16M4 17h16" />
                <circle cx="9" cy="7" r="1.6" />
                <circle cx="15" cy="12" r="1.6" />
                <circle cx="8" cy="17" r="1.6" />
              </>
            }
          />
          <Feature
            title="Instant predictions"
            text="Enter values, get an answer right away, straight in the browser."
            icon={<path d="M13 3 4 14h6l-1 7 9-11h-6l1-7Z" />}
          />
          <Feature
            title="Notebook to browser"
            text="Trained models served behind one clean, versionable API."
            icon={
              <>
                <rect x="3" y="4" width="18" height="16" rx="2" />
                <path d="M3 9h18M8 14l2 2-2 2M13 18h3" />
              </>
            }
          />
        </div>
      </section>
    </>
  );
}

function Feature({ title, text, icon }: { title: string; text: string; icon: ReactNode }) {
  return (
    <div className="flex flex-col items-center">
      <svg
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth={1.4}
        strokeLinecap="round"
        strokeLinejoin="round"
        className="h-8 w-8 text-ink"
        aria-hidden="true"
      >
        {icon}
      </svg>
      <h3 className="mt-4 text-lg font-semibold text-ink">{title}</h3>
      <p className="mt-1.5 max-w-xs text-sm text-slate">{text}</p>
    </div>
  );
}
