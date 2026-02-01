import { defineConfig, mergeConfig } from 'vite';
import { viteStaticCopy } from 'vite-plugin-static-copy';
import baseConfig from '../../../../../web/vite.config.base';

// https://vitejs.dev/config/
export default mergeConfig(
  baseConfig,
  defineConfig({
    server: {
      fs: {
        allow: ['../../../../..'],
      },
    },
    plugins: [
      viteStaticCopy({
        targets: [
          {
            src: 'static/*',
            dest: 'static',
          },
        ],
      }),
    ],
  })
);
