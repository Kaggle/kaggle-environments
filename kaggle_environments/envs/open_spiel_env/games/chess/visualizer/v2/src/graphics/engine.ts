import { Application, Container, Texture } from 'pixi.js';
import { BOARD_PADDING_RATIO, BOARD_SIZE, LAYERS, type Layer } from './constants';

export function engine() {
  const app = new Application();

  const resources = {
    background: new Container(),
    highlights: new Container(),
    pieces: new Container(),
  } satisfies Record<Layer, Container>;

  // Animated pieces set zIndex = 1 to draw above stationary siblings.
  resources.pieces.sortableChildren = true;

  for (const layer of LAYERS) app.stage.addChild(resources[layer]);

  return {
    app,
    resources,
    textures: {} as Record<string, Texture>,
    toHighlightTexture: Texture.WHITE as Texture,
    squareSize: 0,
    boardOffset: 0,
    // Scrub bookkeeping + in-flight animations for syncPieces.
    lastUpdateTime: 0,
    lastStep: -1,
    animations: new Set<{ stop: () => void }>(),
  };
}

export async function initialiseEngine(engine: Engine, canvas: HTMLCanvasElement) {
  const { width, height } = canvas;

  await engine.app.init({
    canvas,
    width,
    height,
    antialias: false,
    backgroundAlpha: 0,
    resolution: window.devicePixelRatio,
    autoDensity: true,
  });

  // Round squareSize to an integer to avoid subpixeling.
  engine.squareSize = Math.floor(width / (BOARD_SIZE + 2 * BOARD_PADDING_RATIO));
  engine.boardOffset = (width - engine.squareSize * BOARD_SIZE) / 2;
}

export type Engine = ReturnType<typeof engine>;
