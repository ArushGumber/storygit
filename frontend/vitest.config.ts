import { defineConfig } from "vitest/config";

// The client is plain TypeScript with no DOM dependency, so it runs in node -- no jsdom,
// no setup file, no fixture harness. Components are covered by the end-to-end smoke and by
// the screenshot audit, which is where component bugs actually show up.
export default defineConfig({
  test: {
    environment: "node",
    include: ["src/**/*.test.ts"],
  },
});
