import { memo, useEffect, useRef, useState } from 'react';
import { Game } from 'tenuki';
import { GameRendererProps } from '@kaggle-environments/core';
import { GoStep } from '../transformers/goReplayTypes';
import { tenukiLogger } from '../utils/tenukiLogger';
import Layout from './Layout';
import useGameStore from '../stores/useGameStore';

export default memo(function GameRenderer(options: GameRendererProps<GoStep[]>) {
  const isFirstRender = useRef(true);
  const [ready, setReady] = useState(false);
  const setState = useGameStore((state) => state.setState);

  useEffect(() => {
    if (isFirstRender.current) {
      isFirstRender.current = false;
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setReady(true);
    }

    const parameters = options.replay.configuration.openSpielGameParameters;
    const game = new Game({
      boardSize: parameters.board_size,
      komi: parameters.komi,
      scoring: 'area', // Tromp-Tailor Rules
    });

    for (const step of options.replay.steps) {
      if (step.step > options.step) break;

      const player = step.players.find((p) => p.isTurn);
      // A forfeit is a "turn" for timeline purposes but puts no stone on the
      // board -- there is no legal move to replay.
      if (player?.forfeited) continue;

      const move = player?.actionDisplayText;

      if (move === 'PASS') {
        game.pass();
      } else if (move) {
        const y = game.boardSize - parseInt(move.slice(1));
        const x = 'abcdefghjklmnopqrst'.indexOf(move.charAt(0));
        game.playAt(y, x);
      }
    }

    game.blackName = options.replay.info?.TeamNames.at(0);
    game.whiteName = options.replay.info?.TeamNames.at(1);
    game.step = options.step;
    game.gameStart = game.moveNumber() === 0;
    // Read the flag off the step the transformer marked terminal rather than
    // inferring it from `step > moveNumber()`: a forfeit step contributes no
    // move, so the move count lags and the heuristic fired a step early.
    game.gameOver = options.replay.steps.at(options.step)?.isTerminal ?? false;

    tenukiLogger(game);

    setState(game, options);
  }, [options, setState]);

  if (!ready) return null;

  return <Layout />;
});
