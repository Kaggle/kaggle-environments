import { useEffect, useState } from 'react';
import type { Action, Planet } from '../engine/types';

interface Props {
  planet: Planet;
  /** Pixel position to anchor the menu (top-left). */
  anchor: { left: number; top: number };
  aimAngle: number | null;
  ships: number;
  /** Ships already queued from this planet in earlier pending actions. */
  alreadyQueued: number;
  onShipsChange(n: number): void;
  onSend(action: Action): void;
  onCancel(): void;
}

export function FleetMenu({ planet, anchor, aimAngle, ships, alreadyQueued, onShipsChange, onSend, onCancel }: Props) {
  const [angleText, setAngleText] = useState('0');

  // Keep angle input in sync with mouse-aim.
  useEffect(() => {
    if (aimAngle !== null) {
      setAngleText(((aimAngle * 180) / Math.PI).toFixed(1));
    }
  }, [aimAngle]);

  const maxShips = Math.max(0, Math.floor(planet.ships) - alreadyQueued);
  const shipsClamped = Math.max(1, Math.min(maxShips, Math.floor(ships) || 0));

  const send = () => {
    if (shipsClamped <= 0) return;
    const angleDeg = Number(angleText);
    const angle = Number.isFinite(angleDeg) ? (angleDeg * Math.PI) / 180 : (aimAngle ?? 0);
    onSend([planet.id, angle, shipsClamped]);
  };

  return (
    <div className="fleet-menu" style={{ left: anchor.left, top: anchor.top }}>
      <h3>
        Planet #{planet.id} — {maxShips} available
        {alreadyQueued > 0 && <span style={{ color: '#ffd166', fontWeight: 'normal' }}> ({alreadyQueued} queued)</span>}
      </h3>
      <div className="row">
        <label>Ships</label>
        <input
          type="number"
          min={1}
          max={maxShips}
          value={shipsClamped}
          onChange={(e) => onShipsChange(Math.floor(Number(e.target.value) || 0))}
        />
        <button onClick={() => onShipsChange(maxShips)} title="Send all">
          All
        </button>
      </div>
      <div className="row">
        <label>Angle</label>
        <input type="number" step={1} value={angleText} onChange={(e) => setAngleText(e.target.value)} />
        <span style={{ color: '#aaaab0', fontSize: 12 }}>°</span>
      </div>
      <div className="actions">
        <button onClick={send}>Send</button>
        <button onClick={onCancel}>Cancel</button>
      </div>
      <div className="mobile-hint" style={{ marginTop: 6, color: '#aaaab0', fontSize: 11 }}>
        Aim with the mouse. Right-click or Esc to cancel.
      </div>
    </div>
  );
}
