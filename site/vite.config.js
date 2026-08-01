import { resolve } from "node:path";
import { defineConfig } from "vite";
import { pages } from "./scripts/pages.mjs";

export default defineConfig({
  base: "/runD/",
  build: {
    outDir: "dist",
    emptyOutDir: true,
    rollupOptions: {
      input: Object.fromEntries(
        pages.map((page) => [
          page.replaceAll("/", "-").replace(".html", ""),
          resolve(import.meta.dirname, page),
        ]),
      ),
    },
  },
});
