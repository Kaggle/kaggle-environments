import { useCallback } from 'react';
import { createReplayVisualizer, ReplayAdapter, defaultGetStepRenderTime } from '@kaggle-environments/core';
import { goTransformer } from './transformers/goTransformer';
import GameRenderer from './components/GameRenderer';
import useGameStore from './stores/useGameStore';
import './App.css';

export default function App() {
  const init = useCallback((element: HTMLDivElement) => {
    const gameName = 'open_spiel_go';
    const ui = 'side-panel';
    const adapter = new ReplayAdapter({
      gameName,
      GameRenderer,
      ui,
      transformer: (replay) => ({
        ...replay,
        steps: goTransformer(replay),
        isTransformed: true,
      }),
      getStepRenderTime: (step, replayMode, speedModifier) => {
        console.log("getStepRenderTime", step.step);
        const time = defaultGetStepRenderTime(step, replayMode, speedModifier);
        const game = useGameStore.getState().game;
        const state = game.currentState();

        const isPass = state.pass;
        const isDoublePass = isPass && game._moves.at(-2).pass;
        const isFirstCapture =
            state.capturedPositions?.length &&
            state.capturedPositions?.length === state.blackStonesCaptured + state.whiteStonesCaptured;
        const isCriticalHit = state.capturedPositions?.length && state.capturedPositions.length >= 10;
        const isDragonLoss = state.capturedPositions?.length && state.capturedPositions.length >= 15;

        if (isDoublePass) return time * 2;
        if (isPass) return time * 2;
        if (isFirstCapture) return time * 2;
        if (isDragonLoss) return time * 2;
        if (isCriticalHit) return time * 2;

        return time;
      },
    });
    createReplayVisualizer(element, adapter);
  }, []);

  return <div id="container" ref={init} />;
}
