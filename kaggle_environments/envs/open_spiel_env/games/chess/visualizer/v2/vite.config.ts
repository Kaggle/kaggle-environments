import { defineConfig, mergeConfig } from 'vite';
import baseConfig from '../../../../../../../web/vite.config.base';

export default mergeConfig(baseConfig, defineConfig({}));
