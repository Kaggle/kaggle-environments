import { memo, useEffect, useState } from 'react';
import useGameStore from '../stores/useGameStore';
import usePreferences from '../stores/usePreferences';
import useHeroAnimation from '../stores/useHeroAnimation';
import knightRiv from '../assets/kaggle_knight.riv?url';
import styles from './GameOverModal.module.css';
import { Ribbon } from './Ribbon.tsx';

function formatDuration(totalSeconds: number): string {
  const h = Math.floor(totalSeconds / 3600);
  const m = Math.floor((totalSeconds % 3600) / 60);
  const s = Math.floor(totalSeconds % 60);
  return [h, m, s].map((n) => String(n).padStart(2, '0')).join(':');
}

function avg(nums: number[]): number {
  if (nums.length === 0) return 0;
  return nums.reduce((a, b) => a + b, 0) / nums.length;
}

interface StatRow {
  label: string;
  black: string | number;
  white: string | number;
}

export default memo(function GameOverModal() {
  const game = useGameStore((state) => state.game);
  const options = useGameStore((state) => state.options);
  const showAnimations = usePreferences((state) => state.showAnimations);
  const play = useHeroAnimation((s) => s.play);
  const cancel = useHeroAnimation((s) => s.cancel);
  const [showStats, setShowStats] = useState(!showAnimations);

  useEffect(() => {
    if (!showAnimations) {
      setShowStats(true);
      return;
    }
    setShowStats(false);
    play(knightRiv, () => setShowStats(true));
    return cancel;
  }, [showAnimations, play, cancel]);

  if (!options) return null;

  const state = game.currentState();
  const step = options.replay.steps[options.step];

  const winnerPlayer = step.players.find((player) => player.reward === 1);
  const winnerColor = winnerPlayer?.id === 0 ? 'black' : 'white';
  const teamNames = options.replay.info?.TeamNames as string[] | undefined;
  const blackName = teamNames?.[0] ?? 'Black';
  const whiteName = teamNames?.[1] ?? 'White';
  const winnerName = winnerColor === 'black' ? blackName : whiteName;

  const points = game.score();
  const captured = { black: state.whiteStonesCaptured, white: state.blackStonesCaptured };
  const passes = { black: state.blackPassStones, white: state.whitePassStones };

  const tokens = { black: 0, white: 0 };
  const durations: { black: number[]; white: number[] } = { black: [], white: [] };

  for (const replayStep of options.replay.steps) {
    const player = replayStep.players.find((p) => p.generateReturns);
    if (!player?.generateReturns) continue;

    for (const json of player.generateReturns) {
      const ret = JSON.parse(json);
      const modelName: string = ret.model ?? ret.request_for_logging?.model ?? '';
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

      const color = player.id ? 'white' : 'black';
      tokens[color] += totalTokens;
      durations[color].push(ret.duration_success_only_secs ?? 0);
    }
  }

  const totalMoves = options.replay.steps.length;
  const allDurations = [...durations.black, ...durations.white];
  const gameDuration = allDurations.reduce((a, b) => a + b, 0);

  const rows: StatRow[] = [
    { label: 'Stones Captured', black: captured.black, white: captured.white },
    { label: 'Point Total', black: points.black, white: points.white },
    { label: 'No. of Passes', black: passes.black, white: passes.white },
    { label: 'Tokens Used', black: tokens.black.toLocaleString(), white: tokens.white.toLocaleString() },
    {
      label: 'Avg. Time per Move',
      black: `${Math.round(avg(durations.black))}s`,
      white: `${Math.round(avg(durations.white))}s`,
    },
  ];

  if (!showStats) return null;

  return (
    <div className={styles.modal}>
      <div className="ribbon">
        <Ribbon>Winner is {winnerName}!</Ribbon>
      </div>
      <div className={styles.meta}>
        Game Duration: {formatDuration(gameDuration)}
        <br />
        Total Moves: {totalMoves}
      </div>
      <table className={styles.table}>
        <thead>
          <tr>
            <th />
            <th>{blackName}</th>
            <th>{whiteName}</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.label}>
              <td>{row.label}</td>
              <td>{row.black}</td>
              <td>{row.white}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
});
