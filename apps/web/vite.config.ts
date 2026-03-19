import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: true,
    host: true,
    allowedHosts: true,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8001",
        changeOrigin: true,
      },
      "/media": {
        target: "http://127.0.0.1:8001",
        changeOrigin: true,
      },
      "/docs": {
        target: "http://127.0.0.1:8001",
        changeOrigin: true,
      },
      "/openapi.json": {
        target: "http://127.0.0.1:8001",
        changeOrigin: true,
      },
    },
  },
});
