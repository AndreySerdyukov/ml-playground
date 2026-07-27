import { Link } from "react-router-dom";
import type { ModelInfo } from "../api";

// Карточка модели в каталоге: эмодзи, название, категория/таргет, CTA.
export function ModelCard({ model }: { model: ModelInfo }) {
  return (
    <Link
      to={`/models/${model.name}`}
      className="group flex flex-col items-center rounded-apple bg-mist px-6 py-10 text-center transition duration-200 hover:scale-[1.02]"
    >
      <div className="text-5xl">{model.emoji}</div>
      <h3 className="mt-4 text-2xl font-semibold capitalize text-ink">{model.name}</h3>
      <p className="mt-1 text-sm text-slate">
        {model.category} · {model.target}
      </p>
      <span className="mt-5 text-[15px] text-link">Try it ›</span>
    </Link>
  );
}
