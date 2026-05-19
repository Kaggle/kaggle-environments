import { AnimatePresence } from 'motion/react';
import Stone from './Stone';
import { pathTint } from '../types';

interface MancalaStoreProps {
  count: number;
  side: 'left' | 'right';
  pitIndex: number;
  isSource?: boolean;
  pathStep?: number;
  pathTotal?: number;
  isLastDest?: boolean;
  isCaptured?: boolean; // stores can't be captured; accepted only because pitProps is shared
}

export default function MancalaStore({ count, side, pitIndex, pathStep, pathTotal, isLastDest }: MancalaStoreProps) {
  const stones = Array.from({ length: count }, (_, i) => i);
  const onPath = pathStep !== undefined && pathTotal !== undefined;
  const classes = [
    'mancala-store',
    `mancala-store-${side}`,
    onPath ? 'is-on-path' : '',
    isLastDest ? 'is-last-dest' : '',
  ]
    .filter(Boolean)
    .join(' ');
  // Flow through stores is always vertical: right store exits upward to top row,
  // left store exits downward to bottom row. We only render the exit arrow -- entry
  // is already implied by the previous pit's arrow.
  const style = onPath
    ? ({ ['--path-bg' as any]: pathTint(pathStep!, pathTotal!, 0.36) } as React.CSSProperties)
    : undefined;
  return (
    <div className={classes} style={style}>
      {onPath && !isLastDest && (
        <span className={`mancala-store-arrow mancala-store-arrow-${side}`} aria-hidden>
          ➤
        </span>
      )}
      <div className="mancala-store-stones">
        <AnimatePresence mode="popLayout">
          {stones.map((i) => (
            <Stone key={`s${pitIndex}-${i}`} seed={pitIndex * 97 + i} />
          ))}
        </AnimatePresence>
      </div>
      <div className="mancala-store-count">{count}</div>
    </div>
  );
}
