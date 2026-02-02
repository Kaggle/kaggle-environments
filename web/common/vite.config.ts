import { defineConfig } from 'vite';
import { resolve } from 'path';
import dts from 'vite-plugin-dts';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [dts()],
  build: {
    lib: {
      entry: resolve(__dirname, 'src/index.ts'),
      name: '@kaggle-environments/common',
      fileName: 'index',
    },
    outDir: 'dist',
    rollupOptions: {
      external: [
        'react',
        'react-dom',
        'react/jsx-runtime',
        'react-router',
        'react-markdown',
        'react-virtuoso',
        '@kaggle-environments/core',
        // Externalize all MUI and emotion packages
        /^@mui\/.*/,
        /^@emotion\/.*/,
      ],
      output: {
        globals: {
          react: 'React',
          'react-dom': 'ReactDOM',
          'react/jsx-runtime': 'jsxRuntime',
          'react-router': 'ReactRouter',
          'react-markdown': 'ReactMarkdown',
          'react-virtuoso': 'ReactVirtuoso',
          '@kaggle-environments/core': 'KaggleEnvironmentsCore',
        },
      },
    },
  },
});
