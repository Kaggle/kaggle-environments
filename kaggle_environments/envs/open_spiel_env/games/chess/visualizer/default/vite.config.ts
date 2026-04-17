import { defineConfig, mergeConfig } from 'vite';
import react from '@vitejs/plugin-react';
import baseConfig from '../../../../../../../web/vite.config.base';

// https://vitejs.dev/config/
// React plugin is needed because this visualizer uses ReplayAdapter which uses React internally
export default mergeConfig(
  baseConfig,
  defineConfig({
    plugins: [react()],
  })
);
