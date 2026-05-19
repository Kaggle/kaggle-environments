import { AnimatePresence } from 'motion/react';
import Stone from './Stone';
import { pathTint } from '../types';

interface PitProps {
  count: number;
  isTopRow?: boolean;
  pitIndex: number;
  isSource?: boolean;
  pathStep?: number;
  pathTotal?: number;
  isLastDest?: boolean;
  isCaptured?: boolean;
  arrowDir?: 'left' | 'right';
}

export default function Pit({
  count,
  isTopRow,
  pitIndex,
  isSource,
  pathStep,
  pathTotal,
  isLastDest,
  isCaptured,
  arrowDir,
}: PitProps) {
  const stones = Array.from({ length: count }, (_, i) => i);
  const onPath = pathStep !== undefined && pathTotal !== undefined;
  const classes = [
    'mancala-pit',
    isSource ? 'is-source' : '',
    onPath ? 'is-on-path' : '',
    isLastDest ? 'is-last-dest' : '',
    isCaptured ? 'is-captured' : '',
  ]
    .filter(Boolean)
    .join(' ');
  const showArrow = (isSource || onPath) && !isLastDest && arrowDir;
  const style = onPath ? ({ ['--path-bg' as any]: pathTint(pathStep!, pathTotal!) } as React.CSSProperties) : undefined;
  return (
    <div className={classes} style={style}>
      <div className="mancala-pit-stones">
        <AnimatePresence mode="popLayout">
          {stones.map((i) => (
            <Stone key={`p${pitIndex}-${i}`} seed={pitIndex * 31 + i} />
          ))}
        </AnimatePresence>
      </div>
      {showArrow && (
        <span className={`mancala-arrow mancala-arrow-${arrowDir}`} aria-hidden>
          ➤
        </span>
      )}
      <div className={`mancala-pit-count ${isTopRow ? 'is-top' : 'is-bottom'}`}>{count}</div>
    </div>
  );
}
