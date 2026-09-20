import { defineConfig } from "vite";
import { tanstackStart } from "@tanstack/react-start/plugin/vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import tsconfigPaths from "vite-tsconfig-paths";
import { nitro } from "nitro/vite";

export default defineConfig(({ command, mode }) => ({
  base: mode === "pages" ? process.env["GIANO_BASE_PATH"] || "/giano/" : "/",
  server: { host: "127.0.0.1", port: 8080 },
  resolve: { dedupe: ["react", "react-dom"] },
  plugins: [
    tsconfigPaths(),
    tailwindcss(),
    tanstackStart({
      server: { entry: "server" },
      ...(mode === "pages"
        ? { spa: { enabled: true, prerender: { outputPath: "/index.html" } } }
        : {}),
    }),
    ...(command === "build" && mode !== "pages"
      ? [nitro({ preset: "vercel" })]
      : []),
    react(),
  ],
}));
