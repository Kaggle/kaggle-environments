import { defineConfig } from 'vite';
import { resolve } from 'path';
import dts from 'vite-plugin-dts';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [dts()],
  resolve: {
    alias: {
      react: 'preact/compat',
      'react-dom': 'preact/compat',
      'react/jsx-runtime': 'preact/jsx-runtime',
    },
  },
  build: {
    lib: {
      entry: resolve(__dirname, 'src/index.ts'),
      name: '@kaggle-environments/common',
      fileName: 'index',
    },
    outDir: 'dist',
    rollupOptions: {
      external: [
        'preact',
        'preact/compat',
        'preact/jsx-runtime',
        'preact/hooks',
        'styled-components',
        '@mui/material',
        '@mui/material/useMediaQuery',
        '@kaggle-environments/core',
        'react-virtuoso',
      ],
      output: {
        globals: {
          preact: 'preact',
          'preact/compat': 'preactCompat',
          'preact/jsx-runtime': 'preactJsxRuntime',
          'preact/hooks': 'preactHooks',
          'styled-components': 'styled',
          '@mui/material': 'MaterialUI',
          '@mui/material/useMediaQuery': 'useMediaQuery',
          '@kaggle-environments/core': 'KaggleEnvironmentsCore',
          'react-virtuoso': 'ReactVirtuoso',
        },
      },
    },
  },
});
