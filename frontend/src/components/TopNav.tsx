import { Link, NavLink } from "react-router-dom";
import type { ModelInfo } from "../api";

// Стеклянный sticky-навбар: бренд слева, модели справа (переключение).
export function TopNav({ models }: { models: ModelInfo[] }) {
  return (
    <header className="glass-nav sticky top-0 z-50 border-b border-hair">
      <nav className="mx-auto flex h-12 max-w-content items-center gap-6 px-6">
        <Link to="/" className="shrink-0 text-[15px] font-semibold tracking-tight text-ink">
          ML Playground
        </Link>
        <div className="flex flex-1 items-center gap-5 overflow-x-auto text-[12px]">
          {models.map((m) => (
            <NavLink
              key={m.name}
              to={`/models/${m.name}`}
              className={({ isActive }) =>
                `whitespace-nowrap capitalize transition ${
                  isActive ? "text-ink" : "text-slate hover:text-ink"
                }`
              }
            >
              {m.name}
            </NavLink>
          ))}
        </div>
      </nav>
    </header>
  );
}
