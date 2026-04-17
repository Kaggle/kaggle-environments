import { createReplayVisualizer, ReplayAdapter } from '@kaggle-environments/core';
import { renderer } from './repeated_poker_renderer';
import {
  repeatedPokerTransformerV2,
  getPokerStepRenderTime,
  getPokerStepInterestingEvents,
} from './transformers/v2/repeatedPokerTransformerV2';
import { getPokerStepLabel, getPokerStepDescription } from './transformers/v1/repeatedPokerTransformer';
import { RepeatedPokerStep } from './transformers/v2/poker-steps-types';
import './style.css';

const app = document.getElementById('app');
if (app) {
  if (import.meta.env?.DEV && import.meta.hot) {
    import.meta.hot.accept();
  }
  createReplayVisualizer(
    app,
    new ReplayAdapter({
      gameName: 'open_spiel_repeated_poker',
      renderer: renderer,
      ui: 'side-panel',
      transformer: (replay) => ({
        ...replay,
        steps: repeatedPokerTransformerV2(replay),
        isTransformed: true,
      }),
      getStepLabel: (step) => getPokerStepLabel(step as RepeatedPokerStep),
      getStepDescription: (step) => getPokerStepDescription(step as RepeatedPokerStep),
      getStepRenderTime: (step, replayMode, speedModifier) =>
        getPokerStepRenderTime(step as RepeatedPokerStep, replayMode, speedModifier),
      getInterestingEvents: (steps) => getPokerStepInterestingEvents(steps as RepeatedPokerStep[]),
    })
  );
}
