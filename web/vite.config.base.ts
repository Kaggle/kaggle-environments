import { defineConfig } from "vite";
import { resolve } from "path";
import checker from "vite-plugin-checker";

// https://vitejs.dev/config/
export default defineConfig({
  optimizeDeps: {
    exclude: ["@kaggle-environments/core"]
  },
  plugins: [
    checker({ typescript: true, overlay: false }),
    {
      name: "custom-header-plugin",
      configureServer(server) {
        const originalPrintUrls = server.printUrls;
        server.printUrls = () => {
          const name = process.env.VITE_CUSTOM_HEADER_NAME;
          const path = process.env.VITE_CUSTOM_HEADER_PATH;
          if (name && path) {
            const header = `\n  ┃ Running: ${name}\n  ┃ Path:    ${path}\n`;
            process.stdout.write(header);
          }
          originalPrintUrls();
        };
      }
    }
  ]
});
