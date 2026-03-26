import { useEffect, useRef } from 'react';
import { Ribbon } from './Ribbon.tsx';
import useGameStore from '../stores/useGameStore';
import styles from './GameOver.module.css';

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

export default function GameOver() {
  const dialogRef = useRef<HTMLDialogElement>(null);
  const game = useGameStore((state) => state.game);
  const options = useGameStore((state) => state.options);

  useEffect(() => {
    const dialog = dialogRef.current;
    if (dialog && !dialog.open) {
      dialog.show();
      dialog.focus();
    }
  }, []);

  const step = options.replay.steps.at(options.step);
  // TODO(pim-at-stink): https://github.com/Stinkstudios/kaggle-ai-visualiser/issues/15
  if (!step) return null;
  if (!game.gameOver) return null;

  const state = game.currentState();
  const points = game.score();
  const winnerColor = points.black > points.white ? 'black' : 'white';
  const blackName = game.blackName ?? 'Black';
  const whiteName = game.whiteName ?? 'White';
  const winnerName = winnerColor === 'black' ? blackName : whiteName;
  const captured = { black: state.whiteStonesCaptured, white: state.blackStonesCaptured };
  const passes = { black: state.blackPassStones, white: state.whitePassStones };
  const tokens = { black: 0, white: 0 };
  const durations: { black: number[]; white: number[] } = { black: [], white: [] };

  for (const replayStep of options.replay.steps) {
    const player = replayStep.players.find((player) => player.generateReturns);
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

  const totalMoves = game.moveNumber();
  const allDurations = [...durations.black, ...durations.white];
  const gameDuration = allDurations.reduce((a, b) => a + b, 0);

  const rows: StatRow[] = [
    {
      label: 'Stones Captured',
      black: captured.black,
      white: captured.white,
    },
    {
      label: 'Points',
      black: points.black,
      white: points.white,
    },
    {
      label: 'Number of Passes',
      black: passes.black,
      white: passes.white,
    },
    {
      label: 'Tokens Used',
      black: tokens.black.toLocaleString(),
      white: tokens.white.toLocaleString(),
    },
    {
      label: 'Average Time per Move',
      black: `${Math.round(avg(durations.black))}s`,
      white: `${Math.round(avg(durations.white))}s`,
    },
  ];

  return (
    <dialog ref={dialogRef} className={styles.modal} aria-label="Game over" tabIndex={-1}>
      <div className="ribbon">
        <Ribbon>
          <h2 className={styles.heading}>Winner is {winnerName}!</h2>
        </Ribbon>
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
      <p className={styles.rulesNote}>
        Scoring follows{' '}
        <a href="https://tromp.github.io/go.html" target="_blank" rel="noopener noreferrer">
          Tromp-Taylor rules
        </a>
      </p>
    </dialog>
  );
}
