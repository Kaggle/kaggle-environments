import eslint from "@eslint/js";
import { defineConfig } from "eslint/config";
import tseslint from "typescript-eslint";
import react from "eslint-plugin-react";
import reactHooks from "eslint-plugin-react-hooks";

export default defineConfig([
  eslint.configs.recommended,
  tseslint.configs.recommended,
  {
    files: ["**/*.ts", "**/*.tsx", "**/*.d.ts"],
    plugins: {
      react,
      "react-hooks": reactHooks,
    },
    settings: {
      react: {
        version: "detect",
      },
    },
    rules: {
      "@typescript-eslint/no-explicit-any": "off",
      ...react.configs.recommended.rules,
      ...reactHooks.configs.recommended.rules,
      "react/react-in-jsx-scope": "off",
    },
  },
]);
