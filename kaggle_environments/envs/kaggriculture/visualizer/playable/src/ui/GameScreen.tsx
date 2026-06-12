import { useMemo } from 'react';
import type { PlayerAction } from '../engine/types';
import { ActionPanel } from './ActionPanel';
import { FarmView } from './FarmView';
import { GameOverModal } from './GameOverModal';
import { HUD } from './HUD';
import { useGameWorker, type SetupResult } from './useGameWorker';

interface Props {
  setup: SetupResult;
  onExit(): void;
}

export function GameScreen({ setup, onExit }: Props) {
  const { state, busy, error, stepGame, reset } = useGameWorker(setup);

  const humanPlayerId = useMemo<number | null>(() => {
    const idx = setup.slots.findIndex((s) => s.kind === 'human');
    return idx >= 0 ? idx : null;
  }, [setup.slots]);

  const playerNames = useMemo(
    () => setup.slots.map((s, i) => (s.kind === 'human' ? `Player ${i + 1} (You)` : `Player ${i + 1} (${s.agentId})`)),
    [setup.slots]
  );

  const handleSubmit = (action: PlayerAction) => {
    if (humanPlayerId === null) return;
    void stepGame({ [humanPlayerId]: action });
  };

  const handleAiStep = () => {
    if (humanPlayerId !== null) return;
    void stepGame({});
  };

  if (error && !state) {
    return <div className="placeholder">Worker error: {error}</div>;
  }
  if (!state) return <div className="placeholder">Initializing engine…</div>;

  return (
    <>
      <HUD state={state} busy={busy} error={error} onReset={() => void reset()} onExit={onExit} />
      <div className="game-body">
        <div className="game-main">
          <FarmView state={state} config={setup.config} playerNames={playerNames} />
        </div>
        {humanPlayerId !== null ? (
          <ActionPanel state={state} player={humanPlayerId} busy={busy} onSubmit={handleSubmit} />
        ) : (
          <aside className="action-panel">
            <h3>AI vs AI</h3>
            <p className="action-hints">No human players — click Step to advance the simulation.</p>
            <button type="button" onClick={handleAiStep} disabled={busy || state.done}>
              Step
            </button>
          </aside>
        )}
      </div>
      {state.done && (
        <GameOverModal
          state={state}
          slots={setup.slots}
          humanPlayerId={humanPlayerId}
          onReplay={() => void reset()}
          onExit={onExit}
        />
      )}
    </>
  );
}
