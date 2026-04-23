import { Sprite, Texture } from 'pixi.js';
import { animate } from 'motion';
import type { Chess } from 'chess.js';
import type { Engine } from '../engine';
import { squareToPixel } from '../coordinates';
import { GRID_LINE_WIDTH } from '../constants';
import type { PreferencesState } from '../../stores/usePreferences';

const HIGHLIGHT_COLOR = 0xbdeeff;
const FROM_ALPHA = 0.4;
const TO_ALPHA = 0.7;
const FADE_DURATION = 0.25; // seconds
const TO_DELAY = 0.18; // seconds — "to" square fades in slightly after "from"

export function syncHighlights(engine: Engine, chess: Chess, prefs: PreferencesState) {
  const { squareSize, boardOffset, resources } = engine;

  resources.highlights.removeChildren();

  if (!prefs.showHighlights) return;

  const lastMove = chess.history({ verbose: true }).at(-1);
  if (!lastMove) return;

  const squares = [
    { square: lastMove.from, delay: 0, alpha: FROM_ALPHA },
    { square: lastMove.to, delay: TO_DELAY, alpha: TO_ALPHA },
  ];

  // Highlights are inset by the grid line width so they sit inside the grid
  // lines rather than overlapping the top/left borders.
  const size = squareSize - GRID_LINE_WIDTH;

  for (const { square, delay, alpha } of squares) {
    const { x, y } = squareToPixel(square, squareSize, 'white', boardOffset);

    const sprite = Sprite.from(Texture.WHITE);
    sprite.tint = HIGHLIGHT_COLOR;
    sprite.x = x - squareSize / 2 + GRID_LINE_WIDTH;
    sprite.y = y - squareSize / 2 + GRID_LINE_WIDTH;
    sprite.width = size;
    sprite.height = size;
    resources.highlights.addChild(sprite);

    if (prefs.reducedMotion) {
      sprite.alpha = alpha;
    } else {
      sprite.alpha = 0;
      engine.animations.add(animate(sprite, { alpha }, { duration: FADE_DURATION, delay }));
    }
  }
}
