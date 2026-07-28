import { useEffect, useState } from "react";

type Theme = "light" | "dark";

// Стартовое значение берём из класса, уже выставленного анти-фликер скриптом в index.html.
function currentTheme(): Theme {
  return document.documentElement.classList.contains("dark") ? "dark" : "light";
}

// Переключатель светлой/тёмной темы: флипает класс .dark на <html> и помнит выбор.
export function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>(currentTheme);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", theme === "dark");
    try {
      localStorage.setItem("theme", theme);
    } catch {
      // приватный режим без localStorage – игнорируем
    }
  }, [theme]);

  const isDark = theme === "dark";
  return (
    <button
      type="button"
      onClick={() => setTheme(isDark ? "light" : "dark")}
      aria-label={isDark ? "Switch to light theme" : "Switch to dark theme"}
      title={isDark ? "Light" : "Dark"}
      className="shrink-0 rounded-full p-1.5 text-slate transition hover:text-ink"
    >
      <svg
        viewBox="0 0 24 24"
        className="h-[18px] w-[18px]"
        fill="none"
        stroke="currentColor"
        strokeWidth={1.6}
        strokeLinecap="round"
        strokeLinejoin="round"
        aria-hidden="true"
      >
        {isDark ? (
          // солнце → переключиться на светлую
          <>
            <circle cx="12" cy="12" r="4" />
            <path d="M12 2.5v2M12 19.5v2M2.5 12h2M19.5 12h2M5.1 5.1l1.4 1.4M17.5 17.5l1.4 1.4M18.9 5.1l-1.4 1.4M6.5 17.5l-1.4 1.4" />
          </>
        ) : (
          // луна → переключиться на тёмную: сплошной серп (читается крупнее) + звёздочки
          <>
            <path
              d="M21 12.8A9 9 0 1 1 11.2 3 7.2 7.2 0 0 0 21 12.8Z"
              fill="currentColor"
              stroke="none"
            />
            <path
              d="M18.6 3.5Q18.6 5 20.1 5Q18.6 5 18.6 6.5Q18.6 5 17.1 5Q18.6 5 18.6 3.5Z"
              fill="currentColor"
              stroke="none"
            />
            <path
              d="M15.5 6.4Q15.5 7.3 16.4 7.3Q15.5 7.3 15.5 8.2Q15.5 7.3 14.6 7.3Q15.5 7.3 15.5 6.4Z"
              fill="currentColor"
              stroke="none"
            />
          </>
        )}
      </svg>
    </button>
  );
}
