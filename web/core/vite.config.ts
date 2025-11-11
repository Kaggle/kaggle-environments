import { defineConfig } from 'vite';
import { resolve } from 'path';
import dts from 'vite-plugin-dts';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [dts()],
  build: {
    lib: {
      entry: resolve(__dirname, 'src/index.ts'),
      name: '@kaggle-environments/core',
      fileName: 'index',
    },
    outDir: 'dist',
    rollupOptions: {
      external: ['preact', 'preact/hooks', 'htm'],
    },
  },
});
