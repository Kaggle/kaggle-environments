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
        'styled-components',
        '@mui/material',
        '@mui/material/useMediaQuery',
        '@kaggle-environments/core',
        'react-virtuoso',
      ],
      output: {
        globals: {
          react: 'React',
          'react-dom': 'ReactDOM',
          'react/jsx-runtime': 'jsxRuntime',
          'react-router': 'ReactRouter',
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
