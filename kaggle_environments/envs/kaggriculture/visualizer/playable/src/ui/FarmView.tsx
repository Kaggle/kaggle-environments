/**
 * React wrapper around the default visualizer's imperative render pipeline.
 * Mounts an empty <div>, builds the shell + cached refs once, then calls
 * `renderObservation` whenever the live GameState changes.
 */

import { useEffect, useRef } from 'react';
import {
  buildShell,
  collectRefs,
  renderObservation,
  type BoardSize,
  type LayoutRefs,
} from '../../../default/src/renderFarm';
import type { Config, GameState } from '../engine/types';
import { createPriceHistory, stateToView, type PriceHistoryTracker } from './buildView';

interface FarmViewProps {
  state: GameState;
  config: Config;
  playerNames: string[];
}

export function FarmView({ state, config, playerNames }: FarmViewProps) {
  const hostRef = useRef<HTMLDivElement | null>(null);
  const refsRef = useRef<LayoutRefs | null>(null);
  const historyRef = useRef<PriceHistoryTracker | null>(null);
  const shellKeyRef = useRef<string>('');

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    const board: BoardSize = { rows: config.boardSize, cols: config.boardSize };
    const key = `${board.rows}x${board.cols}|${playerNames.join('|')}`;
    if (shellKeyRef.current !== key) {
      buildShell(host, board, playerNames);
      refsRef.current = collectRefs(host, board);
      shellKeyRef.current = key;
    }
    if (!historyRef.current) {
      historyRef.current = createPriceHistory(config.turnsPerDay);
    }
    const priceHistory = historyRef.current.record(state);
    const view = stateToView(state, priceHistory);
    if (refsRef.current) renderObservation(refsRef.current, view, config);
  }, [state, config, playerNames]);

  return <div ref={hostRef} className="farm-view-root" />;
}
