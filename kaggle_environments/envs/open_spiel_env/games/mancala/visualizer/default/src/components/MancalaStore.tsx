import { AnimatePresence } from 'motion/react';
import Stone from './Stone';

interface MancalaStoreProps {
  count: number;
  label: string;
  side: 'left' | 'right';
  pitIndex: number;
  isSource?: boolean;
  pathStep?: number;
  pathTotal?: number;
  isLastDest?: boolean;
}

export default function MancalaStore({
  count,
  label,
  side,
  pitIndex,
  pathStep,
  pathTotal,
  isLastDest,
}: MancalaStoreProps) {
  const stones = Array.from({ length: count }, (_, i) => i);
  const onPath = pathStep !== undefined && pathTotal !== undefined;
  const intensity = onPath ? 0.22 + (pathStep! / pathTotal!) * 0.5 : 0;
  const classes = [
    'mancala-store',
    `mancala-store-${side}`,
    onPath ? 'is-on-path' : '',
    isLastDest ? 'is-last-dest' : '',
  ]
    .filter(Boolean)
    .join(' ');
  const style = onPath
    ? ({ ['--path-bg' as any]: `rgba(0, 138, 187, ${intensity})` } as React.CSSProperties)
    : undefined;
  return (
    <div className={classes} style={style}>
      <div className="mancala-store-label">{label}</div>
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
