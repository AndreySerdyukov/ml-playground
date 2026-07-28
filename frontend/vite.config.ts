import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Proxy /api to the backend in dev mode to avoid running into CORS.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target: "http://localhost:8000", changeOrigin: true },
      "/health": { target: "http://localhost:8000", changeOrigin: true },
    },
  },
});
