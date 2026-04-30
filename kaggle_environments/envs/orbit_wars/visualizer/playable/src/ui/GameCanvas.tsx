import { useEffect, useMemo, useRef, useState } from 'react';
import { renderGame } from '../render/renderGame';
import { drawOverlay } from '../render/overlay';
import { drawPendingFleets } from '../render/pending';
import { planetAt, screenToBoard } from '../render/hitTest';
import { fleetSpeed } from '../engine/interpreter';
import { DEFAULT_SHIP_SPEED } from '../engine/constants';
import { TARGET_HIT_RADIUS } from '../render/overlay';
import type { Action, GameState, Planet } from '../engine/types';

interface Props {
  state: GameState;
  selected: Planet | null;
  aimAngle: number | null;
  aimShips: number;
  aimTarget: { x: number; y: number } | null;
  aimPlaced: boolean;
  dragging: boolean;
  pendingActions: Action[];
  humanPlayerId: number;
  onMouseDown(planet: Planet | null, boardX: number, boardY: number): void;
  onMouseMove(boardX: number, boardY: number): void;
  onMouseUp(boardX: number, boardY: number): void;
  onContextMenu(): void;
}

export function GameCanvas({
  state,
  selected,
  aimAngle,
  aimShips,
  aimTarget,
  aimPlaced,
  dragging,
  pendingActions,
  humanPlayerId,
  onMouseDown,
  onMouseMove,
  onMouseUp,
  onContextMenu,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [mouseBoard, setMouseBoard] = useState<{ x: number; y: number } | null>(null);

  const stepSpeed = fleetSpeed(Math.max(1, aimShips), DEFAULT_SHIP_SPEED);

  const pendingShipsByPlanet = useMemo(() => {
    const m = new Map<number, number>();
    for (const [pid, , ships] of pendingActions) {
      m.set(pid, (m.get(pid) ?? 0) + ships);
    }
    return m;
  }, [pendingActions]);

  const renderSettings = useMemo(
    () => ({ showFleetNumbers: true, showProductionDots: true, pendingShipsByPlanet }),
    [pendingShipsByPlanet]
  );

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    const cssSize = Math.max(100, Math.floor(canvas.getBoundingClientRect().width));
    if (canvas.width !== Math.round(cssSize * dpr)) {
      canvas.width = Math.round(cssSize * dpr);
      canvas.height = Math.round(cssSize * dpr);
    }
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    renderGame(ctx, state, cssSize, renderSettings);
    drawPendingFleets(ctx, cssSize, state, pendingActions, humanPlayerId);
    drawOverlay(ctx, cssSize, state, selected, aimAngle, aimShips, stepSpeed, mouseBoard, aimTarget);
  }, [
    state,
    selected,
    aimAngle,
    aimShips,
    stepSpeed,
    mouseBoard,
    pendingActions,
    renderSettings,
    humanPlayerId,
    aimTarget,
  ]);

  // Re-render on resize
  useEffect(() => {
    const handler = () => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const dpr = window.devicePixelRatio || 1;
      const cssSize = Math.max(100, Math.floor(canvas.getBoundingClientRect().width));
      canvas.width = Math.round(cssSize * dpr);
      canvas.height = Math.round(cssSize * dpr);
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      renderGame(ctx, state, cssSize, renderSettings);
      drawPendingFleets(ctx, cssSize, state, pendingActions, humanPlayerId);
      drawOverlay(ctx, cssSize, state, selected, aimAngle, aimShips, stepSpeed, mouseBoard, aimTarget);
    };
    window.addEventListener('resize', handler);
    return () => window.removeEventListener('resize', handler);
  }, [
    state,
    selected,
    aimAngle,
    aimShips,
    stepSpeed,
    mouseBoard,
    pendingActions,
    renderSettings,
    humanPlayerId,
    aimTarget,
  ]);

  // Cursor priority: grabbing > grab (over placed target) > crosshair (selected) > default.
  let cursor: string | undefined;
  if (selected) {
    if (dragging) {
      cursor = 'grabbing';
    } else if (aimPlaced && aimTarget && mouseBoard) {
      const dx = mouseBoard.x - aimTarget.x;
      const dy = mouseBoard.y - aimTarget.y;
      cursor = dx * dx + dy * dy <= TARGET_HIT_RADIUS * TARGET_HIT_RADIUS ? 'grab' : 'crosshair';
    } else {
      cursor = 'crosshair';
    }
  }

  return (
    <canvas
      ref={canvasRef}
      style={cursor ? { cursor } : undefined}
      onMouseDown={(e) => {
        const c = canvasRef.current;
        if (!c) return;
        const { x, y } = screenToBoard(c, e);
        onMouseDown(planetAt(state, x, y), x, y);
      }}
      onMouseMove={(e) => {
        const c = canvasRef.current;
        if (!c) return;
        const { x, y } = screenToBoard(c, e);
        setMouseBoard({ x, y });
        onMouseMove(x, y);
      }}
      onMouseLeave={() => setMouseBoard(null)}
      onMouseUp={(e) => {
        const c = canvasRef.current;
        if (!c) return;
        const { x, y } = screenToBoard(c, e);
        onMouseUp(x, y);
      }}
      onContextMenu={(e) => {
        e.preventDefault();
        onContextMenu();
      }}
    />
  );
}
