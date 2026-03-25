import { memo, useEffect, useRef, useState } from 'react';
import { Game } from 'tenuki';
import { GameRendererProps } from '@kaggle-environments/core';
import { GoStep } from '../transformers/goReplayTypes';
import { tenukiLogger } from '../utils/tenukiLogger';
import Layout from './Layout';
import useGameStore from '../stores/useGameStore';

export default memo(function GameRenderer(options: GameRendererProps<GoStep[]>) {
  console.log("GameRenderer")
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

      const move = step.players.find((p) => p.isTurn)?.actionDisplayText;

      if (move === 'PASS') {
        game.pass();
      } else if (move) {
        const y = game.boardSize - parseInt(move.slice(1));
        const x = 'abcdefghjklmnopqrst'.indexOf(move.charAt(0));
        game.playAt(y, x);
      }
    }

    game.blackName = options.agents.at(0).Name;
    game.whiteName = options.agents.at(1).Name;
    game.step = options.step;
    game.gameStart = game.moveNumber() === 0;
    game.gameOver = game.step > game.moveNumber();

    tenukiLogger(game);

    setState(game, options);
  }, [options, setState]);

  if (!ready) return null;

  return <Layout />;
});
