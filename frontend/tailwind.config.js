/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#1d1d1f", // primary text / чёрные действия
        slate: "#6e6e73", // вторичный текст
        mist: "#f5f5f7", // фон секций/карточек
        hair: "#d2d2d7", // hairline-границы
        link: "#0066cc", // акцентный синий Apple (скупо)
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
