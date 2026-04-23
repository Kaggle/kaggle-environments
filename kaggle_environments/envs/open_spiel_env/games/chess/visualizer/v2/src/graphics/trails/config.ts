import type { PieceType } from '../constants';

export const MOVE_THRESHOLD = 0.5; // px — ignore sub-pixel jitter

export interface TrailConfig {
  textures: string[];
  size: [number, number]; // scale range (relative to squareSize)
  rate: number; // seconds between spawns (0 = every frame with movement — pair with max to cap)
  count: number; // particles per spawn
  lifetime: number; // seconds until fully faded
  alpha: number;
  drift?: number; // wander radius (relative to squareSize)
  follow?: number; // lerp factor — particles follow the piece
  max?: number; // max live particles per piece (for follow mode)
  spin?: boolean;
  align?: boolean; // rotate toward movement direction
}

export const TRAIL_CONFIGS: Record<PieceType, TrailConfig> = {
  // King: anxious, trembling squiggles
  k: {
    textures: ['particles/squiggle-1.png', 'particles/squiggle-2.png', 'particles/squiggle-3.png'],
    size: [0.06, 0.12],
    rate: 0.03,
    count: 1,
    lifetime: 0.5,
    alpha: 0.6,
    drift: 0.3,
    spin: true,
  },

  // Queen: heavy, confident puffs
  q: {
    textures: ['particles/puff1.png', 'particles/puff2.png', 'particles/puff3.png'],
    size: [0.2, 0.5],
    rate: 0.02,
    count: 1,
    lifetime: 0.8,
    alpha: 0.7,
    drift: 0.3,
  },

  // Bishop: flowing, graceful drift
  b: {
    textures: ['particles/bishop1.png', 'particles/bishop2.png'],
    size: [0.2, 0.4],
    rate: 0.015,
    count: 1,
    lifetime: 1.2,
    alpha: 0.65,
    drift: 0.5,
  },

  // Knight: shadow afterimage
  n: {
    textures: ['particles/knight-shadow.png'],
    size: [1, 1],
    rate: 0,
    count: 1,
    lifetime: 0.75,
    alpha: 0.5,
    follow: 0.2,
    max: 1,
  },

  // Rook: steady, direct debris
  r: {
    textures: ['particles/rook1.png', 'particles/rook2.png', 'particles/rook3.png', 'particles/rook4.png'],
    size: [0.08, 0.12],
    rate: 0.01,
    count: 3,
    lifetime: 0.6,
    alpha: 0.7,
  },

  // Pawn: zippy directional marks
  p: {
    textures: ['particles/pawn.png'],
    size: [0.02, 0.05],
    rate: 0.01,
    count: 1,
    lifetime: 0.3,
    alpha: 0.7,
    follow: 0.015,
    max: 5,
    align: true,
  },
};
