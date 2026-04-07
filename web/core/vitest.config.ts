import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    // Default to node; use `// @vitest-environment jsdom` per-file for React/DOM tests
    environment: 'node',
    include: ['src/**/*.test.ts', 'src/**/*.test.tsx'],
  },
});
