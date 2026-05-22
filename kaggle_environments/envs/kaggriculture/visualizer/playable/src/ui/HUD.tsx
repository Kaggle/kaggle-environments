import type { GameState } from '../engine/types';

interface Props {
  state: GameState;
  busy: boolean;
  error: string | null;
  onReset(): void;
  onExit(): void;
}

export function HUD({ state, busy, error, onReset, onExit }: Props) {
  return (
    <header className="hud">
      <div className="hud-meta">
        {busy && <span className="hud-tag">working…</span>}
        {state.done && <span className="hud-tag hud-tag-done">game over</span>}
        {error && <span className="hud-tag hud-tag-error">error: {error}</span>}
      </div>
      <div className="hud-buttons">
        <button type="button" onClick={onReset} disabled={busy}>
          Reset
        </button>
        <button type="button" onClick={onExit}>
          New Game
        </button>
      </div>
    </header>
  );
}
