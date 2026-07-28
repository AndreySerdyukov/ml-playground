/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        // Семантические токены на CSS-переменных: значения флипаются под .dark (см. index.css).
        canvas: "rgb(var(--canvas) / <alpha-value>)", // фон страницы
        surface: "rgb(var(--surface) / <alpha-value>)", // поля/дропдауны/поднятые карточки
        ink: "rgb(var(--ink) / <alpha-value>)", // основной текст / сплошные действия
        slate: "rgb(var(--slate) / <alpha-value>)", // вторичный текст
        mist: "rgb(var(--mist) / <alpha-value>)", // фон секций/карточек результата
        hair: "rgb(var(--hair) / <alpha-value>)", // hairline-границы
        link: "rgb(var(--link) / <alpha-value>)", // акцентный синий Apple
      },
      fontFamily: {
        sans: [
          "-apple-system",
          "BlinkMacSystemFont",
          '"SF Pro Display"',
          '"SF Pro Text"',
          '"Helvetica Neue"',
          "Helvetica",
          "Arial",
          "sans-serif",
        ],
      },
      maxWidth: { content: "980px" },
      borderRadius: { apple: "18px" },
    },
  },
  plugins: [],
};
