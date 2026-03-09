import { memo } from 'react';
import useGameStore from '../stores/useGameStore';

export default memo(function GameOverModal() {
  const game = useGameStore((state) => state.game);
  const options = useGameStore((state) => state.options);

  if (game.isOver()) {
    const state = game.currentState();
    const step = options?.replay.steps[options.step];

    const winner = step!.players.find((player) => player.reward === 1)!.id === 0 ? 'black' : 'white';
    const points = game.score();
    const captured = { white: state.blackStonesCaptured, black: state.whiteStonesCaptured };
    const passes = { white: state.whitePassStones, black: state.blackPassStones };

    const tokens = { black: 0, white: 0 };
    const durations: { [key: string]: number[] } = { black: [], white: [] };
    options?.replay.steps.forEach((step) => {
      const player = step.players.find((player) => player.generateReturns);

      if (!player) return;

      player!.generateReturns!.forEach((json) => {
        const ret = JSON.parse(json);

        const modelName: string = ret.model ?? ret.request_for_logging.model;
        const promptTokens: number = ret.prompt_tokens ?? 0;
        let generationTokens: number = ret.generation_tokens ?? 0;
        const reasoningTokens: number = ret.reasoning_tokens ?? 0;
        let totalTokens: number = ret.total_tokens ?? 0;

        if (modelName.includes('grok') || modelName.includes('gemini')) {
          generationTokens = totalTokens - promptTokens;
        }

        if (totalTokens === 0) {
          totalTokens = promptTokens + generationTokens + reasoningTokens;
        }

        if (player.id) {
          tokens.white += totalTokens;
          durations.white.push(ret.duration_success_only_secs);
        } else {
          tokens.black += totalTokens;
          durations.black.push(ret.duration_success_only_secs);
        }
      });
    });

    const timePerMove = {
      'black': Math.round(durations.black.reduce((a, b) => a + b) / durations.black.length),
      'white': Math.round(durations.white.reduce((a, b) => a + b) / durations.white.length),
    };

    console.log('game over', winner, points, captured, passes, tokens, timePerMove);
  }

  return null;
});
