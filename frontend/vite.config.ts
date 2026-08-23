import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In development the frontend runs on 5173 and the API on 8000, so /api is proxied --
// that keeps the browser on one origin and makes CORS a non-issue during development.
// In production one uvicorn process serves both from the same origin, so the proxy and
// the CORS configuration are both dead weight that never runs.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: true,
  },
});
