import { createReplayVisualizer, ReplayAdapter } from '@kaggle-environments/core';
import { renderer } from './renderer';
import { getKargoStepRenderTime } from './timing';
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
    gameName: 'kargo',
    renderer: renderer as any,
    ui: 'inline',
    getStepRenderTime: (step, replayMode, speedModifier) => getKargoStepRenderTime(step, replayMode, speedModifier),
  })
);
