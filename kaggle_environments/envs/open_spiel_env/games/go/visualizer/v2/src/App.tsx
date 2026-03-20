import { useCallback } from 'react';
import { createReplayVisualizer, ReplayAdapter, defaultGetStepRenderTime } from '@kaggle-environments/core';
import { goTransformer } from './transformers/goTransformer';
import { GoStep } from './transformers/goReplayTypes';
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

        const [, move] = (step as GoStep).boardState.previous_move_a1!.split(' ');
        if (move === 'PASS') {
          game.pass();
        } else {
          type index = { [key: string]: number };
          const cols: index = {
            'a': 0,
            'b': 1,
            'c': 2,
            'd': 3,
            'e': 4,
            'f': 5,
            'g': 6,
            'h': 7,
            'j': 8,
            'k': 9,
            'l': 10,
            'm': 11,
            'n': 12,
            'o': 13,
            'p': 14,
            'q': 15,
            'r': 16,
            's': 17,
            't': 18,
          };
          const y = game.boardSize - parseInt(move.slice(1));
          const x = cols[move.charAt(0)];
          game.playAt(y, x);
        }

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
