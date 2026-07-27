import type { ReactNode } from "react";

// Свои монохромные line-иконки моделей (в стиле Apple, тонкий штрих, currentColor).
const PATHS: Record<string, ReactNode> = {
  wine: (
    <>
      <path d="M8 3h8l-.7 6.2a3.4 3.4 0 0 1-6.6 0L8 3Z" />
      <path d="M12 15.6V21" />
      <path d="M8.5 21h7" />
    </>
  ),
  diamonds: (
    <>
      <path d="M6 3h12l3 5-9 13L3 8l3-5Z" />
      <path d="M3 8h18" />
      <path d="M9.5 8 12 21l2.5-13" />
    </>
  ),
  cars: (
    <>
      <path d="M3 14l2-5a2.5 2.5 0 0 1 2.3-1.6h9.4A2.5 2.5 0 0 1 19 9l2 5" />
      <path d="M3.5 14h17v3a1 1 0 0 1-1 1H4.5a1 1 0 0 1-1-1v-3Z" />
      <circle cx="7.5" cy="18" r="1.7" />
      <circle cx="16.5" cy="18" r="1.7" />
    </>
  ),
  bayesian: (
    <>
      <rect x="2.5" y="6" width="19" height="12" rx="2" />
      <circle cx="12" cy="12" r="2.6" />
      <path d="M6 12h.02M18 12h.02" />
    </>
  ),
  loans: (
    <>
      <path d="M12 3.5 21 8H3l9-4.5Z" />
      <path d="M4 8v9M9 8v9M15 8v9M20 8v9" />
      <path d="M3 20.5h18" />
    </>
  ),
  uplift: (
    <>
      <path d="M3 16l5.5-5.5 3 3L20.5 6" />
      <path d="M14.5 6h6v6" />
    </>
  ),
  default: (
    <>
      <rect x="4" y="4" width="16" height="16" rx="4" />
      <path d="M9 12h6M12 9v6" />
    </>
  ),
};

export function ModelIcon({ name, className }: { name: string; className?: string }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.4}
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
      aria-hidden="true"
    >
      {PATHS[name] ?? PATHS.default}
    </svg>
  );
}
