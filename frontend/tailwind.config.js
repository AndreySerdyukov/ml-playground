/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        // Semantic tokens on CSS variables: values flip under .dark (see index.css).
        canvas: "rgb(var(--canvas) / <alpha-value>)", // page background
        surface: "rgb(var(--surface) / <alpha-value>)", // fields/dropdowns/raised cards
        ink: "rgb(var(--ink) / <alpha-value>)", // primary text / solid actions
        slate: "rgb(var(--slate) / <alpha-value>)", // secondary text
        mist: "rgb(var(--mist) / <alpha-value>)", // background of sections/result cards
        hair: "rgb(var(--hair) / <alpha-value>)", // hairline borders
        link: "rgb(var(--link) / <alpha-value>)", // Apple accent blue
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
