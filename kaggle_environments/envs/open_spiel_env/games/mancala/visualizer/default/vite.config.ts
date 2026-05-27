import react from '@vitejs/plugin-react';
import { defineConfig, mergeConfig } from 'vite';
import baseConfig from '../../../../../../../web/vite.config.base';

export default mergeConfig(
  baseConfig,
  defineConfig({
    publicDir: 'replays',
    plugins: [react()],
  })
);
