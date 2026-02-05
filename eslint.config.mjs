import eslint from "@eslint/js";
import { defineConfig } from "eslint/config";
import tseslint from "typescript-eslint";
import react from "eslint-plugin-react";
import reactHooks from "eslint-plugin-react-hooks";
import jsxA11Y from "eslint-plugin-jsx-a11y";
import eslintPluginPlaywright from "eslint-plugin-playwright";
import globals from "globals";

export default defineConfig([
  eslint.configs.recommended,
  tseslint.configs.recommended,
  {
    files: ["**/*.js", "**/*.mjs", "**/*.cjs"],
    languageOptions: {
      globals: {
        ...globals.browser,
        ...globals.node,
      },
    },
  },
  {
    files: ["**/*.ts", "**/*.tsx", "**/*.d.ts"],
    languageOptions: {
      globals: {
        ...globals.browser, // For browser environment
        ...globals.node, // For Node.js environment
        ...globals.builtin, // For built-in globals
      },
    },
    plugins: {
      react,
      "react-hooks": reactHooks,
      "jsx-a11y": jsxA11Y,
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
      ...jsxA11Y.flatConfigs.recommended.rules,
      "react/react-in-jsx-scope": "off",
    },
  },
  {
    files: ["**/*.test.ts", "**/*.spec.ts"],
    plugins: {
      playwright: eslintPluginPlaywright,
    },
    rules: {
      ...eslintPluginPlaywright.configs.recommended.rules,
    },
  },
]);
