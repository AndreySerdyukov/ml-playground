import { Link, NavLink } from "react-router-dom";
import type { ModelInfo } from "../api";
import { ModelIcon } from "./ModelIcon";
import { ThemeToggle } from "./ThemeToggle";

// Стеклянный sticky-навбар: бренд слева, модели справа (переключение).
export function TopNav({ models }: { models: ModelInfo[] }) {
  return (
    <header className="glass-nav sticky top-0 z-50 border-b border-hair">
      <nav className="mx-auto flex h-12 max-w-content items-center gap-7 px-6 text-[14px]">
        <Link
          to="/"
          className="flex shrink-0 items-center gap-2 font-medium tracking-tight text-ink"
        >
          <ModelIcon name="logo" className="h-[22px] w-[22px]" />
          ML Playground
        </Link>
        <div className="flex flex-1 items-center gap-6 overflow-x-auto">
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
        <ThemeToggle />
      </nav>
    </header>
  );
}
