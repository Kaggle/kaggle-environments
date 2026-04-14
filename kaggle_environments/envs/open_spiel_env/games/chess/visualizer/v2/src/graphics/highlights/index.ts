import { Graphics, Sprite, Texture, type Renderer } from 'pixi.js';
import { animate } from 'motion';
import type { Chess } from 'chess.js';
import type { Engine } from '../engine';
import { squareToPixel } from '../coordinates';
import { GRID_LINE_WIDTH } from '../constants';

const HIGHLIGHT_COLOR = 0xbdeeff;
const HIGHLIGHT_ALPHA = 1;
const FADE_DURATION = 0.25; // seconds
const TO_DELAY = 0.18; // seconds — "to" square fades in slightly after "from"

const STROKE_WIDTH = 2;
const STROKE_COLOR = 0x000000; // Tinted by HIGHLIGHT_COLOR → darker shade of the highlight

/**
 * Pre-render a white square with a 2px inner stroke to a texture. Tinted at
 * runtime with HIGHLIGHT_COLOR so the fill becomes #BDEEFF and the stroke
 * becomes a darker shade. Only called once at init.
 */
export function createToHighlightTexture(renderer: Renderer, squareSize: number): Texture {
  const size = squareSize - GRID_LINE_WIDTH;
  const g = new Graphics();
  g.rect(0, 0, size, size).fill(0xffffff);
  g.rect(0, 0, size, size).stroke({ width: STROKE_WIDTH, color: STROKE_COLOR, alignment: 0 });
  const texture = renderer.generateTexture(g);
  g.destroy();
  return texture;
}

export function syncHighlights(engine: Engine, chess: Chess) {
  const { squareSize, boardOffset, resources } = engine;

  resources.highlights.removeChildren();

  const lastMove = chess.history({ verbose: true }).at(-1);
  if (!lastMove) return;

  const squares = [
    { square: lastMove.from, texture: Texture.WHITE, delay: 0 },
    { square: lastMove.to, texture: engine.toHighlightTexture, delay: TO_DELAY },
  ];

  // Highlights are inset by the grid line width so they sit inside the grid
  // lines rather than overlapping the top/left borders.
  const size = squareSize - GRID_LINE_WIDTH;

  for (const { square, texture, delay } of squares) {
    const { x, y } = squareToPixel(square, squareSize, 'white', boardOffset);

    const sprite = new Sprite({
      texture,
      tint: HIGHLIGHT_COLOR,
      x: x - squareSize / 2 + GRID_LINE_WIDTH,
      y: y - squareSize / 2 + GRID_LINE_WIDTH,
      width: size,
      height: size,
      alpha: 0,
    });
    resources.highlights.addChild(sprite);

    engine.animations.add(animate(sprite, { alpha: HIGHLIGHT_ALPHA }, { duration: FADE_DURATION, delay }));
  }
}
