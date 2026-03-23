import { createReplayVisualizer, ReplayAdapter } from '@kaggle-environments/core';
import { renderer } from './renderer';
import {
  connectFourTransformer,
  getConnectFourStepLabel,
  getConnectFourStepDescription,
} from './transformers/connectFourTransformer';
import { ConnectFourStep } from './transformers/connectFourReplayTypes';
import './style.css';

const app = document.getElementById('app');
if (!app) {
  throw new Error('Could not find app element');
}

if (import.meta.env?.DEV && import.meta.hot) {
  import.meta.hot.accept();
}

createReplayVisualizer(
  app,
  new ReplayAdapter({
    gameName: 'open_spiel_connect_four',
    renderer: renderer as any,
    ui: 'side-panel',
    transformer: (replay) => ({
      ...replay,
      steps: connectFourTransformer(replay),
      isTransformed: true,
    }),
    getStepLabel: (step) => getConnectFourStepLabel(step as ConnectFourStep),
    getStepDescription: (step) => getConnectFourStepDescription(step as ConnectFourStep),
  })
);
