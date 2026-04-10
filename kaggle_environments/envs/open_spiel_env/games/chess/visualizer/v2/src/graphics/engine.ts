import { Application, Container, type Texture } from 'pixi.js';
import { BOARD_SIZE, LAYERS, type Layer } from './constants';

export function engine() {
  const app = new Application();

  const resources = {
    background: new Container(),
    pieces: new Container(),
  } satisfies Record<Layer, Container>;

  // Animated pieces set zIndex = 1 to draw above stationary siblings.
  resources.pieces.sortableChildren = true;

  for (const layer of LAYERS) app.stage.addChild(resources[layer]);

  return {
    app,
    resources,
    textures: {} as Record<string, Texture>,
    squareSize: 0,
    // Scrub bookkeeping + in-flight animations for syncPieces.
    lastUpdateTime: 0,
    lastStep: -1,
    animations: new Set<{ stop: () => void }>(),
  };
}

export async function initialiseEngine(engine: Engine, canvas: HTMLCanvasElement) {
  await engine.app.init({
    canvas,
    width: canvas.width,
    height: canvas.height,
    antialias: false,
    backgroundAlpha: 0,
    resolution: 1,
    autoDensity: false,
  });

  engine.squareSize = engine.app.renderer.width / BOARD_SIZE;
}

export type Engine = ReturnType<typeof engine>;
