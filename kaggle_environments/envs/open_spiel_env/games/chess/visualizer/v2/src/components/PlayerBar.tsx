import pieceWhitePath from '../assets/images/player-color-w.webp';
import pieceBlackPath from '../assets/images/player-color-b.webp';
import bishopBlackPath from '../assets/images/bishop-b-small.webp';
import bishopWhitePath from '../assets/images/bishop-w-small.webp';
import kingBlackPath from '../assets/images/king-b-small.webp';
import kingWhitePath from '../assets/images/king-w-small.webp';
import knightBlackPath from '../assets/images/knight-b-small.webp';
import knightWhitePath from '../assets/images/knight-w-small.webp';
import pawnBlackPath from '../assets/images/pawn-b-small.webp';
import pawnWhitePath from '../assets/images/pawn-w-small.webp';
import queenBlackPath from '../assets/images/queen-b-small.webp';
import queenWhitePath from '../assets/images/queen-w-small.webp';
import rookBlackPath from '../assets/images/rook-b-small.webp';
import rookWhitePath from '../assets/images/rook-w-small.webp';
import useGameStore from '../stores/useGameStore.ts';
import { BrandLogo } from './BrandLogo.tsx';
import styles from './Playerbar.module.css';

const pieceColorImages = {
  b: pieceBlackPath,
  w: pieceWhitePath,
} as const;

const pieceImages = {
  b: {
    pawn: pawnBlackPath,
    knight: knightBlackPath,
    bishop: bishopBlackPath,
    rook: rookBlackPath,
    queen: queenBlackPath,
    king: kingBlackPath,
  },
  w: {
    pawn: pawnWhitePath,
    knight: knightWhitePath,
    bishop: bishopWhitePath,
    rook: rookWhitePath,
    queen: queenWhitePath,
    king: kingWhitePath,
  },
} as const;

type PieceName = keyof (typeof pieceImages)['b'];

interface Props {
  color: 'b' | 'w';
}

export function PlayerBar({ color }: Props) {
  const game = useGameStore((state) => state.game);
  const headers = game.getHeaders();
  const name = headers[color];
  const opponent = color === 'w' ? 'b' : 'w';

  // Temporary array of captures, we will do this properly later.
  const temp__captures: PieceName[] = [
    'pawn',
    'knight',
    'pawn',
    'queen',
    'pawn',
    'pawn',
    'knight',
    'pawn',
    'queen',
    'pawn',
    'pawn',
    'knight',
    'pawn',
    'queen',
    'pawn',
    'pawn',
    'knight',
    'pawn',
    'queen',
    'pawn',
  ];

  return (
    <div className={styles.playerBar} data-player={color}>
      <div className={`${styles.player} squiggle-border`}>
        <div className={`grid-pile ${styles.logo}`}>
          <img src={pieceColorImages[color]} alt="" width="64" height="64" />
          <BrandLogo name={name} />
        </div>
        <p>{name}</p>
      </div>
      <div className={styles.captures}>
        {temp__captures.map((piece, index) => (
          <img
            key={index}
            src={pieceImages[opponent][piece]}
            alt={`captured ${opponent} ${piece}`}
            width="64"
            height="64"
          />
        ))}
      </div>
    </div>
  );
}
