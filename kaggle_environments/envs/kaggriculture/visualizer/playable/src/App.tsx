import { useMemo } from 'react';
import { resolveConfig } from './engine/state';
import { ActionPanel } from './ui/ActionPanel';
import { FarmView } from './ui/FarmView';
import { useGameWorker, type SetupResult } from './ui/useGameWorker';
import type { SlotConfig } from './worker/protocol';

const HUMAN_PLAYER_ID = 0;
const PLAYER_NAMES = ['You', 'Starter AI'];

const SLOTS: SlotConfig[] = [{ kind: 'human' }, { kind: 'ai', agentId: 'starter' }];

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
        <button type="button" onClick={() => reset()} disabled={busy}>
          Reset
        </button>
        <span className="toolbar-status">
          {state ? `Day ${state.day + 1} / Hour ${state.hour + 1} / Step ${state.step}` : 'loading…'}
          {state?.done ? ' — done' : ''}
          {error ? ` — error: ${error}` : ''}
        </span>
      </div>
      <div className="game-body">
        <div className="game-main">
          {state ? (
            <FarmView state={state} config={setup.config} playerNames={PLAYER_NAMES} />
          ) : (
            <div className="placeholder">Initializing engine…</div>
          )}
        </div>
        {state && (
          <ActionPanel
            state={state}
            player={HUMAN_PLAYER_ID}
            busy={busy}
            onSubmit={(action) => void stepGame({ [HUMAN_PLAYER_ID]: action })}
          />
        )}
      </div>
    </div>
  );
}
