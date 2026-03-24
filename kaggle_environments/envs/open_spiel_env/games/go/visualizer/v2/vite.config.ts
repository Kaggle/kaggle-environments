import { mergeConfig, type UserConfig } from 'vite';
import baseConfig from '../../../../../../../web/vite.config.base';

export default mergeConfig(baseConfig as UserConfig, {
  build: {
    // Prevent Vite from transforming files to base64. We deliberately want
    // external assets for caching efficiency.
    assetsInlineLimit: 0,
  },
});
