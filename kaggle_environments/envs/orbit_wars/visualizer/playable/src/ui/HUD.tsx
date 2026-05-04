import type { GameState } from '../engine/types';
import type { SlotConfig } from '../worker/protocol';
import { ownerColor } from '../render/colors';
import { AGENTS } from '../ai';

interface Props {
  state: GameState;
  slots: SlotConfig[];
  pendingCount: number;
  busy: boolean;
  onStep(): void;
  onReset(): void;
  onExit(): void;
}

export function HUD({ state, slots, pendingCount, busy, onStep, onReset, onExit }: Props) {
  return (
    <aside className="hud">
      <h2>Step {state.step} / 500</h2>

      <div className="scores">
        {slots.map((slot, i) => {
          const label = slot.kind === 'human' ? 'You' : (AGENTS[slot.agentId]?.label ?? slot.agentId);
          return (
            <div key={i} className="score-row">
              <span className="swatch" style={{ background: ownerColor(i) }} />
              <span style={{ flex: 1 }}>{label}</span>
              <strong>{state.scores[i] ?? 0}</strong>
            </div>
          );
        })}
      </div>

      <div className="pending">Queued orders: {pendingCount}</div>

      <button className="hud-mobile-hide" onClick={onStep} disabled={busy || state.done}>
        {busy ? 'Working…' : 'Step (Space)'}
      </button>
      <button onClick={onReset} disabled={busy}>
        Reset
      </button>
      <button onClick={onExit}>New Game</button>

      <div className="hud-mobile-hide" style={{ fontSize: 12, color: '#aaaab0', marginTop: 12 }}>
        <p>Click an owned planet to open its fleet menu.</p>
        <p>Drag from the planet to aim a fleet, set the count, then Send.</p>
        <p>Press Space (or click Step) to advance one turn.</p>
      </div>
    </aside>
  );
}
