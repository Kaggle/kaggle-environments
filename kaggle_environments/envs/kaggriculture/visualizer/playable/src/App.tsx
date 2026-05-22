import { useMemo } from 'react';
import { resolveConfig } from './engine/state';
import { FarmView } from './ui/FarmView';
import { useGameWorker, type SetupResult } from './ui/useGameWorker';
import type { SlotConfig } from './worker/protocol';

const PLAYER_NAMES = ['Starter', 'Random'];

const SLOTS: SlotConfig[] = [
  { kind: 'ai', agentId: 'starter' },
  { kind: 'ai', agentId: 'random' },
];

export function App() {
  const setup = useMemo<SetupResult>(
    () => ({
      config: resolveConfig({ seed: 1 }),
      numAgents: SLOTS.length,
      slots: SLOTS,
    }),
    []
  );

  const { state, busy, error, stepGame, reset } = useGameWorker(setup);

  return (
    <div className="app">
      <div className="toolbar">
        <button type="button" onClick={() => stepGame({})} disabled={busy || !state || state.done}>
          Step
        </button>
        <button type="button" onClick={() => reset()} disabled={busy}>
          Reset
        </button>
        <span className="toolbar-status">
          {state ? `Day ${state.day + 1} / Hour ${state.hour + 1} / Step ${state.step}` : 'loading…'}
          {state?.done ? ' — done' : ''}
          {error ? ` — error: ${error}` : ''}
        </span>
      </div>
      {state ? (
        <FarmView state={state} config={setup.config} playerNames={PLAYER_NAMES} />
      ) : (
        <div className="placeholder">Initializing engine…</div>
      )}
    </div>
  );
}
