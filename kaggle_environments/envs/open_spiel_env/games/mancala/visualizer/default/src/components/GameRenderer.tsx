import { memo, useMemo } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import type { GameRendererProps } from '@kaggle-environments/core';
import Pit from './Pit';
import MancalaStore from './MancalaStore';
import { BOTTOM_ROW, MancalaObservation, STORE_LEFT, STORE_RIGHT, TOP_ROW } from '../types';

import goose1Idle from '../assets/goose_1_idle.png';
import goose1Pensive from '../assets/goose_1_pensive.png';
import goose1Elated from '../assets/goose_1_elated.png';
import goose1Sad from '../assets/goose_1_sad.png';
import goose2Idle from '../assets/goose_2_idle.png';
import goose2Pensive from '../assets/goose_2_pensive.png';
import goose2Elated from '../assets/goose_2_elated.png';
import goose2Sad from '../assets/goose_2_sad.png';
import boardImg from '../assets/mancala_board.png';
import backdropImg from '../assets/natural_paper_backdrop.png';

const GOOSE_SPRITES: Record<0 | 1, Record<'idle' | 'pensive' | 'elated' | 'sad', string>> = {
  0: { idle: goose1Idle, pensive: goose1Pensive, elated: goose1Elated, sad: goose1Sad },
  1: { idle: goose2Idle, pensive: goose2Pensive, elated: goose2Elated, sad: goose2Sad },
};

function parseObservation(step: any): MancalaObservation | null {
  const raw = step?.[0]?.observation?.observationString;
  if (typeof raw !== 'string') return null;
  try {
    return JSON.parse(raw) as MancalaObservation;
  } catch {
    return null;
  }
}

function findObservation(steps: any[], idx: number): MancalaObservation | null {
  for (let i = idx; i < steps.length; i++) {
    const obs = parseObservation(steps[i]);
    if (obs) return obs;
  }
  for (let i = idx - 1; i >= 0; i--) {
    const obs = parseObservation(steps[i]);
    if (obs) return obs;
  }
  return null;
}

function sowingPath(source: number, count: number, player: 0 | 1): number[] {
  const skip = player === 0 ? STORE_LEFT : STORE_RIGHT; // skip opponent's store
  const path: number[] = [];
  let idx = source;
  while (path.length < count) {
    idx = (idx + 1) % 14;
    if (idx === skip) continue;
    path.push(idx);
  }
  return path;
}

function gooseSprite(player: 0 | 1, state: MancalaObservation) {
  if (state.is_terminal) {
    if (state.winner === 'draw' || state.winner === null) return GOOSE_SPRITES[player].idle;
    return state.winner === player ? GOOSE_SPRITES[player].elated : GOOSE_SPRITES[player].sad;
  }
  return state.current_player === player ? GOOSE_SPRITES[player].pensive : GOOSE_SPRITES[player].idle;
}

function GameRenderer({ replay, step }: GameRendererProps) {
  const steps = (replay.steps as any[]) ?? [];
  const state = useMemo(() => findObservation(steps, step), [steps, step]);
  const prevState = useMemo(() => findObservation(steps, Math.max(0, step - 1)), [steps, step]);

  const teamNames = ((replay.info as Record<string, any>)?.TeamNames as string[]) || [];
  const player1Name = teamNames[0] || 'Player 1';
  const player2Name = teamNames[1] || 'Player 2';

  if (!state) {
    return (
      <div className="mancala-root mancala-loading" style={{ backgroundImage: `url(${backdropImg})` }}>
        Loading…
      </div>
    );
  }

  const lastAction = state.last_action ?? null;
  const sowing = useMemo(() => {
    if (!prevState || lastAction == null) return null;
    const player = typeof prevState.current_player === 'number' ? (prevState.current_player as 0 | 1) : null;
    if (player !== 0 && player !== 1) return null;
    const count = prevState.board[lastAction];
    if (!count) return null;
    const path = sowingPath(lastAction, count, player);
    return { source: lastAction, path, lastDest: path[path.length - 1], total: path.length };
  }, [prevState, lastAction]);

  const pathStep = useMemo(() => {
    const m = new Map<number, number>();
    if (sowing) sowing.path.forEach((idx, i) => m.set(idx, i + 1));
    return m;
  }, [sowing]);

  // Capture: when the last seed lands in an empty pit on the mover's own side,
  // that seed plus all stones in the opposite pit are scooped into the mover's store.
  // Detect by looking at prev/current board state at the opposite pit.
  const capturedPit = useMemo(() => {
    if (!sowing || !prevState) return null;
    const player = typeof prevState.current_player === 'number' ? (prevState.current_player as 0 | 1) : null;
    if (player !== 0 && player !== 1) return null;
    const dest = sowing.lastDest;
    const ownSide = player === 0 ? dest >= 1 && dest <= 6 : dest >= 8 && dest <= 13;
    if (!ownSide) return null;
    const opposite = 14 - dest;
    const tookOpposite = (prevState.board[opposite] ?? 0) > 0 && (state.board[opposite] ?? 0) === 0;
    const destEmptied = (state.board[dest] ?? 0) === 0;
    return tookOpposite && destEmptied ? opposite : null;
  }, [sowing, prevState, state]);

  function pitProps(idx: number) {
    const step = pathStep.get(idx);
    return {
      isSource: sowing?.source === idx,
      pathStep: step,
      pathTotal: sowing?.total,
      isLastDest: sowing?.lastDest === idx,
      isCaptured: capturedPit === idx,
    };
  }

  const player0Score = state.scores[0];
  const player1Score = state.scores[1];

  function outcome(player: 0 | 1): 'winner' | 'loser' | undefined {
    if (!state!.is_terminal || state!.winner === 'draw' || state!.winner === null) return undefined;
    return state!.winner === player ? 'winner' : 'loser';
  }

  return (
    <div className="mancala-root" style={{ backgroundImage: `url(${backdropImg})` }}>
      <div className="mancala-header">
        <PlayerCard
          name={player1Name}
          score={player0Score}
          spriteUrl={gooseSprite(0, state)}
          active={state.current_player === 0 && !state.is_terminal}
          mirrored={false}
          outcome={outcome(0)}
        />
        <div className="mancala-title-block">
          <div className="mancala-title">
            <span className="mancala-diamond">◆</span>
            <h1>MANCALA</h1>
            <span className="mancala-diamond">◆</span>
          </div>
          <AnimatePresence mode="wait">
            {state.is_terminal && (
              <motion.div
                className="mancala-status"
                key={String(state.winner)}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
              >
                {state.winner === 'draw'
                  ? "Game Over: It's a draw!"
                  : state.winner === 0
                    ? `${player1Name} Wins!`
                    : `${player2Name} Wins!`}
              </motion.div>
            )}
          </AnimatePresence>
        </div>
        <PlayerCard
          name={player2Name}
          score={player1Score}
          spriteUrl={gooseSprite(1, state)}
          active={state.current_player === 1 && !state.is_terminal}
          mirrored
          outcome={outcome(1)}
        />
      </div>

      <div className="mancala-board" style={{ backgroundImage: `url(${boardImg})` }}>
        <div className="mancala-store-slot mancala-store-slot-left">
          <MancalaStore count={state.board[STORE_LEFT]} side="left" pitIndex={STORE_LEFT} {...pitProps(STORE_LEFT)} />
        </div>

        <div className="mancala-pits-grid">
          <div className="mancala-row mancala-row-top">
            {TOP_ROW.map((idx) => (
              <Pit key={idx} pitIndex={idx} count={state.board[idx]} isTopRow arrowDir="left" {...pitProps(idx)} />
            ))}
          </div>
          <div className="mancala-row mancala-row-bottom">
            {BOTTOM_ROW.map((idx) => (
              <Pit key={idx} pitIndex={idx} count={state.board[idx]} arrowDir="right" {...pitProps(idx)} />
            ))}
          </div>
        </div>

        <div className="mancala-store-slot mancala-store-slot-right">
          <MancalaStore
            count={state.board[STORE_RIGHT]}
            side="right"
            pitIndex={STORE_RIGHT}
            {...pitProps(STORE_RIGHT)}
          />
        </div>
      </div>
    </div>
  );
}

interface PlayerCardProps {
  name: string;
  score: number;
  spriteUrl: string;
  active: boolean;
  mirrored: boolean;
  outcome?: 'winner' | 'loser';
}

function PlayerCard({ name, score, spriteUrl, active, mirrored, outcome }: PlayerCardProps) {
  const classes = [
    'mancala-player-card',
    active ? 'is-active' : '',
    outcome === 'winner' ? 'is-winner' : '',
    outcome === 'loser' ? 'is-loser' : '',
  ]
    .filter(Boolean)
    .join(' ');
  return (
    <div className={classes}>
      <div className="mancala-player-frame">
        <motion.img
          key={spriteUrl}
          initial={{ scale: 0.8, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          src={spriteUrl}
          alt={`${name} mascot`}
          style={{ transform: mirrored ? 'scaleX(-1)' : undefined }}
        />
      </div>
      <div className="mancala-player-name">
        <span className="mancala-player-display-name" title={name}>
          {name}
        </span>
        <div className="mancala-player-score">
          <span className="mancala-score-value">{score}</span>
          <span className="mancala-score-label">Points</span>
        </div>
      </div>
    </div>
  );
}

export default memo(GameRenderer);
