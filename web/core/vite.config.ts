import { defineConfig } from 'vite';
import { resolve } from 'path';
import dts from 'vite-plugin-dts';
import react from '@vitejs/plugin-react';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react(), dts()],
  build: {
    lib: {
      entry: resolve(__dirname, 'src/index.ts'),
      name: '@kaggle-environments/core',
      fileName: 'index',
    },
    outDir: 'dist',
    rollupOptions: {
      external: [
        'react',
        'react-dom',
        'react/jsx-runtime',
        '@mui/material',
        '@mui/material/styles',
        '@mui/material/useMediaQuery',
        '@emotion/react',
        '@emotion/styled',
        'react-markdown',
        'react-virtuoso',
      ],
      output: {
        globals: {
          'react': 'React',
          'react-dom': 'ReactDOM',
          'react/jsx-runtime': 'jsxRuntime',
          '@mui/material': 'MuiMaterial',
          '@mui/material/styles': 'MuiStyles',
          '@mui/material/useMediaQuery': 'useMediaQuery',
          '@emotion/react': 'emotionReact',
          '@emotion/styled': 'emotionStyled',
          'react-markdown': 'ReactMarkdown',
          'react-virtuoso': 'reactVirtuoso',
        },
      },
    },
  },
  server: {
    port: 5173,
  },
});
