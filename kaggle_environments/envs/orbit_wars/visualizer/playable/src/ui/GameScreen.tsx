import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { BOARD_SIZE } from '../engine/constants';
import type { Action, Planet } from '../engine/types';
import { ownerColor } from '../render/colors';
import { TARGET_HIT_RADIUS } from '../render/overlay';
import { GameCanvas } from './GameCanvas';
import { FleetMenu } from './FleetMenu';
import { HUD } from './HUD';
import { useGameWorker } from './useGameWorker';
import type { SetupResult } from './SetupScreen';

interface Props {
  setup: SetupResult;
  onExit: () => void;
}

const HUMAN_PLAYER_ID = 0;

export function GameScreen({ setup, onExit }: Props) {
  const { state, busy, error, stepGame, reset } = useGameWorker(setup);
  const [pendingActions, setPendingActions] = useState<Action[]>([]);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [aimAngle, setAimAngle] = useState<number | null>(null);
  const [aimShips, setAimShips] = useState<number>(0);
  const [aimTarget, setAimTarget] = useState<{ x: number; y: number } | null>(null);
  const [aimPlaced, setAimPlaced] = useState(false);
  const [draggingTarget, setDraggingTarget] = useState(false);
  const wrapRef = useRef<HTMLDivElement | null>(null);

  // The selected planet snapshots from the latest state (may move between steps).
  const selected: Planet | null = useMemo(() => {
    if (!state || selectedId === null) return null;
    return state.planets.find((p) => p.id === selectedId) ?? null;
  }, [state, selectedId]);

  const closeMenu = useCallback(() => {
    setSelectedId(null);
    setAimAngle(null);
    setAimTarget(null);
    setAimPlaced(false);
    setDraggingTarget(false);
  }, []);

  const handleStep = useCallback(() => {
    if (busy || !state || state.done) return;
    const actions: Record<number, Action[]> = { [HUMAN_PLAYER_ID]: pendingActions };
    setPendingActions([]);
    closeMenu();
    void stepGame(actions);
  }, [busy, state, pendingActions, stepGame, closeMenu]);

  // Spacebar = step
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.code === 'Space') {
        const target = e.target as HTMLElement | null;
        if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable)) return;
        e.preventDefault();
        handleStep();
      } else if (e.code === 'Escape') {
        closeMenu();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [handleStep, closeMenu]);

  if (error) return <div style={{ padding: 20, color: '#ff6666' }}>Worker error: {error}</div>;
  if (!state) return <div style={{ padding: 20 }}>Loading…</div>;

  // Compute the screen-space anchor for the fleet menu (desktop only).
  // While the user is still picking an aim direction (target not placed yet),
  // park the menu in the top-right corner so it can't sit under the cursor.
  // On mobile the CSS overrides this anchor and pins the menu to the bottom.
  let menuAnchor: { left: number; top: number } | null = null;
  if (selected && wrapRef.current) {
    const rect = wrapRef.current.getBoundingClientRect();
    if (!aimPlaced) {
      menuAnchor = { left: rect.width - 250, top: 10 };
    } else {
      const canvas = wrapRef.current.querySelector('canvas');
      const cRect = canvas?.getBoundingClientRect();
      if (cRect) {
        const scale = cRect.width / BOARD_SIZE;
        const px = cRect.left - rect.left + selected.x * scale;
        const py = cRect.top - rect.top + selected.y * scale;
        const offset = selected.radius * scale + 12;
        const left = px + offset > rect.width - 240 ? px - offset - 230 : px + offset;
        const top = Math.max(10, Math.min(py - 40, rect.height - 220));
        menuAnchor = { left, top };
      }
    }
  }

  return (
    <div className="game">
      <div className="game-canvas-wrap" ref={wrapRef}>
        <GameCanvas
          state={state}
          selected={selected}
          aimAngle={aimAngle}
          aimShips={aimShips}
          aimTarget={aimTarget}
          aimPlaced={aimPlaced}
          dragging={draggingTarget}
          pendingActions={pendingActions}
          humanPlayerId={HUMAN_PLAYER_ID}
          onPointerDown={(planet, bx, by, pointerType) => {
            const isTouch = pointerType !== 'mouse';
            if (!selected) {
              if (planet && planet.owner === HUMAN_PLAYER_ID) {
                const queued = pendingActions.filter(([pid]) => pid === planet.id).reduce((acc, [, , s]) => acc + s, 0);
                const available = Math.max(0, Math.floor(planet.ships) - queued);
                setSelectedId(planet.id);
                setAimShips(available);
                if (isTouch) {
                  // Auto-place a target a few units past the planet, pointed at the sun,
                  // so the player can immediately see the trajectory and drag to refine.
                  const dx = BOARD_SIZE / 2 - planet.x;
                  const dy = BOARD_SIZE / 2 - planet.y;
                  const dist = Math.hypot(dx, dy) || 1;
                  const off = planet.radius + 5;
                  const tx = planet.x + (dx / dist) * off;
                  const ty = planet.y + (dy / dist) * off;
                  setAimTarget({ x: tx, y: ty });
                  setAimAngle(Math.atan2(ty - planet.y, tx - planet.x));
                  setAimPlaced(true);
                  setDraggingTarget(false);
                } else {
                  setAimTarget({ x: bx, y: by });
                  setAimAngle(Math.atan2(by - planet.y, bx - planet.x));
                  setAimPlaced(false);
                  setDraggingTarget(false);
                }
              }
              return;
            }
            // Planet selected. If target hasn't been placed yet, this click locks it.
            if (!aimPlaced) {
              setAimTarget({ x: bx, y: by });
              setAimAngle(Math.atan2(by - selected.y, bx - selected.x));
              setAimPlaced(true);
              setDraggingTarget(true); // allow press+drag to fine-tune in one motion
              return;
            }
            // Target already placed: drag if press lands on it, else deselect.
            // Touch gets a larger hit slop since fingers are imprecise.
            if (aimTarget) {
              const dx = bx - aimTarget.x;
              const dy = by - aimTarget.y;
              const slop = isTouch ? TARGET_HIT_RADIUS * 2 : TARGET_HIT_RADIUS;
              if (dx * dx + dy * dy <= slop * slop) {
                setDraggingTarget(true);
                return;
              }
            }
            closeMenu();
          }}
          onPointerMove={(bx, by) => {
            if (!selected) return;
            // Follow pointer while previewing (target not yet placed) or while dragging it.
            if (!aimPlaced || draggingTarget) {
              setAimTarget({ x: bx, y: by });
              setAimAngle(Math.atan2(by - selected.y, bx - selected.x));
            }
          }}
          onPointerUp={() => {
            setDraggingTarget(false);
          }}
          onContextMenu={closeMenu}
        />
        {selected && menuAnchor && (
          <FleetMenu
            planet={selected}
            anchor={menuAnchor}
            aimAngle={aimAngle}
            ships={aimShips}
            alreadyQueued={pendingActions.filter(([pid]) => pid === selected.id).reduce((acc, [, , s]) => acc + s, 0)}
            onShipsChange={setAimShips}
            onSend={(action) => {
              setPendingActions((p) => [...p, action]);
              closeMenu();
            }}
            onCancel={closeMenu}
          />
        )}
        {state.done && (
          <div className="modal-bg">
            <div className="modal">
              <h2>Game Over</h2>
              <p>
                {state.winners.length === 0 && 'Draw — no ships left.'}
                {state.winners.length === 1 && (
                  <>
                    <span style={{ color: ownerColor(state.winners[0]), fontWeight: 'bold' }}>
                      {state.winners[0] === HUMAN_PLAYER_ID ? 'You win!' : `Player ${state.winners[0] + 1} wins`}
                    </span>
                  </>
                )}
                {state.winners.length > 1 && `Tie: players ${state.winners.map((w) => w + 1).join(', ')}`}
              </p>
              <p style={{ fontSize: 13, color: '#aaaab0' }}>Final scores: {state.scores.join(' — ')}</p>
              <div style={{ display: 'flex', justifyContent: 'center', gap: 8 }}>
                <button onClick={() => void reset()}>Replay (same setup)</button>
                <button onClick={onExit}>New Game</button>
              </div>
            </div>
          </div>
        )}
      </div>
      <HUD
        state={state}
        slots={setup.slots}
        pendingCount={pendingActions.length}
        busy={busy}
        onStep={handleStep}
        onReset={() => {
          setPendingActions([]);
          closeMenu();
          void reset();
        }}
        onExit={onExit}
      />
      {!state.done && (
        <button
          className="fab-step"
          onClick={handleStep}
          disabled={busy}
          aria-label="Advance one turn"
          title="Advance one turn"
        >
          »
        </button>
      )}
    </div>
  );
}
