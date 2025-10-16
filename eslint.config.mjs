import eslint from "@eslint/js";
import { defineConfig } from "eslint/config";
import tseslint from "typescript-eslint";

export default defineConfig([
  eslint.configs.recommended,
  tseslint.configs.recommended,
  {
    files: ["**/*.ts", "**/*.tsx", "**/*.d.ts"], // Apply these rules only to TypeScript files
    rules: {
      "@typescript-eslint/no-explicit-any": "off",
    },
  },
]);
