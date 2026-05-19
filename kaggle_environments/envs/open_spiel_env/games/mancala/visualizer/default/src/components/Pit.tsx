import { AnimatePresence } from 'motion/react';
import Stone from './Stone';

interface PitProps {
  count: number;
  isTopRow?: boolean;
  pitIndex: number;
  isSource?: boolean;
  pathStep?: number;
  pathTotal?: number;
  isLastDest?: boolean;
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
  arrowDir,
}: PitProps) {
  const stones = Array.from({ length: count }, (_, i) => i);
  const onPath = pathStep !== undefined && pathTotal !== undefined;
  const intensity = onPath ? 0.18 + (pathStep! / pathTotal!) * 0.42 : 0;
  const classes = [
    'mancala-pit',
    isSource ? 'is-source' : '',
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
      <div className="mancala-pit-stones">
        <AnimatePresence mode="popLayout">
          {stones.map((i) => (
            <Stone key={`p${pitIndex}-${i}`} seed={pitIndex * 31 + i} />
          ))}
        </AnimatePresence>
      </div>
      {isSource && arrowDir && (
        <span className={`mancala-arrow mancala-arrow-${arrowDir}`}>{arrowDir === 'right' ? '➤' : '➤'}</span>
      )}
      <div className={`mancala-pit-count ${isTopRow ? 'is-top' : 'is-bottom'}`}>{count}</div>
    </div>
  );
}
