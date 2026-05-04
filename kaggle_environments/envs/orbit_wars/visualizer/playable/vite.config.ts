import { defineConfig, mergeConfig } from 'vite';
import react from '@vitejs/plugin-react';
import baseConfig from '../../../../../web/vite.config.base';

export default mergeConfig(
  baseConfig,
  defineConfig({
    plugins: [react()],
    worker: {
      format: 'es',
    },
  })
);
