import { Link } from "react-router-dom";
import type { ModelInfo } from "../api";
import { ModelIcon } from "./ModelIcon";

// Model card in the catalog: emoji, name, category/target, CTA.
export function ModelCard({ model }: { model: ModelInfo }) {
  return (
    <Link
      to={`/models/${model.name}`}
      className="group flex flex-col items-center rounded-apple bg-mist px-6 py-10 text-center transition duration-200 hover:scale-[1.02]"
    >
      <ModelIcon name={model.name} className="h-11 w-11 text-ink" />
      <h3 className="mt-5 text-2xl font-semibold capitalize text-ink">{model.name}</h3>
      <p className="mt-1 text-sm text-slate">
        {model.category} · {model.target}
      </p>
      <span className="mt-5 text-[15px] text-link">Try it ›</span>
    </Link>
  );
}
