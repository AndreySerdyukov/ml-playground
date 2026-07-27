// Брендовый набор line-иконок (из ~/personal/ml-playground/brand). Монохром, currentColor.
// Ключи: логотип-глиф, 6 моделей и 3 фиче-иконки. Цвет/размер задаём через className.
const PATHS: Record<string, string[]> = {
  logo: [
    "M6.5 3.5h11a3 3 0 0 1 3 3v11a3 3 0 0 1-3 3h-11a3 3 0 0 1-3-3v-11a3 3 0 0 1 3-3z",
    "M10 8.5l6 3.5-6 3.5z",
  ],
  wine: ["M8 3.5h8v3.5a4 4 0 0 1-8 0z", "M12 11.5v6.5", "M8.5 18.5h7"],
  diamonds: ["M6.5 4.5h11l3 5-8.5 10-8.5-10z", "M3.5 9.5h17", "M9 9.5L12 19.5 15 9.5"],
  cars: [
    "M3.5 16v-3l2.5-.5 2.5-4h7l3 4 2.5.5v3",
    "M6 12.5h12",
    "M5.6 16a1.9 1.9 0 1 0 3.8 0a1.9 1.9 0 1 0 -3.8 0",
    "M14.6 16a1.9 1.9 0 1 0 3.8 0a1.9 1.9 0 1 0 -3.8 0",
    "M9.4 16h5.2",
  ],
  bayesian: ["M3 17.5c4.2 0 3.6-11 9-11s4.8 11 9 11", "M2.5 17.5h19", "M12 6.5v11"],
  loans: [
    "M4.5 6.5h15a2 2 0 0 1 2 2v7a2 2 0 0 1-2 2h-15a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2z",
    "M9.4 12a2.6 2.6 0 1 0 5.2 0a2.6 2.6 0 1 0 -5.2 0",
    "M5.5 12h1",
    "M17.5 12h1",
  ],
  uplift: ["M3.5 18.5h17", "M3.5 15.5l5-3.5 4 2.5 6.5-7.5", "M14.5 7h4.5v4.5"],
  "model-agnostic-forms": [
    "M4.5 3.5h15a1 1 0 0 1 1 1v15a1 1 0 0 1-1 1h-15a1 1 0 0 1-1-1v-15a1 1 0 0 1 1-1z",
    "M7.5 8.5h9",
    "M7.5 12h9",
    "M7.5 15.5h5",
  ],
  "instant-prediction": ["M13.5 3l-7 11h5l-1 7 7-11h-5z"],
  "notebook-to-browser": [
    "M4.5 4.5h15a1 1 0 0 1 1 1v13a1 1 0 0 1-1 1h-15a1 1 0 0 1-1-1v-13a1 1 0 0 1 1-1z",
    "M3.5 8.5h17",
    "M6.4 6.5h.2",
    "M8.9 6.5h.2",
    "M12 10.5v5",
    "M9.5 13l2.5 2.5 2.5-2.5",
  ],
};

export function ModelIcon({ name, className }: { name: string; className?: string }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.5}
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
      aria-hidden="true"
    >
      {(PATHS[name] ?? PATHS.logo).map((d) => (
        <path key={d} d={d} />
      ))}
    </svg>
  );
}
