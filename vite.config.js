import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";
export default defineConfig({
  base: "./",
  plugins: [vue()],
  server: {
    proxy: {
      "/api": "http://127.0.0.1:8765",
      "/docs": "http://127.0.0.1:8765",
      "/openapi.json": "http://127.0.0.1:8765",
      "/redoc": "http://127.0.0.1:8765",
    },
  },
});
