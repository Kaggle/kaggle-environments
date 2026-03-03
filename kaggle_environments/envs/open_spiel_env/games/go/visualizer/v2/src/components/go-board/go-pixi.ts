import { Application, Assets, Container, Spritesheet } from 'pixi.js';
import type { CellValue, Captures } from './middleman.ts';
import { BOARD_PX, POT_AREA_HEIGHT, getNeighbors } from './constants.ts';
import { drawBoard } from './draw-board.ts';
import { diffGrids } from './diff-grid.ts';
import { createStonePair, posKey, resetPair, type StoneMap } from './stone-map.ts';
import { animateCapture, animateNeighborShockwave, animateStoneDrop } from './animate-stones.ts';
import { Marker } from './marker.ts';
import { Pots } from './pots.ts';
import spritesData from '../../graphics/sprites/sprites.json';
import spritesPng from '../../graphics/sprites/sprites.png';

export interface GoPixiProps {
  grid: CellValue[][];
  step: number;
  lastPlayed: { row: number; col: number } | null;
  captures: Captures;
}

export class GoPixi {
  private app: Application;
  private boardSize: number;
  private container: HTMLElement;

  private sheet: Spritesheet | null = null;
  private layers: { shadow: Container; stone: Container; effects: Container } | null = null;
  private marker: Marker | null = null;
  private pots: Pots | null = null;

  private stoneMap: StoneMap = new Map();
  private prevGrid: CellValue[][] | null = null;
  private prevStep = 0;
  private activeAnims: gsap.core.Animation[] = [];

  private initialized = false;
  private destroyed = false;
  private pendingProps: GoPixiProps | null = null;

  constructor(container: HTMLElement, boardSize: number) {
    this.container = container;
    this.boardSize = boardSize;
    this.app = new Application();
  }

  async init(): Promise<void> {
    const { app, boardSize } = this;

    await app.init({
      width: BOARD_PX,
      height: BOARD_PX + POT_AREA_HEIGHT,
      antialias: false,
      resolution: window.devicePixelRatio,
      backgroundAlpha: 0,
      autoDensity: true,
    });

    if (this.destroyed) { app.destroy(true, { children: true }); return; }

    this.container.appendChild(app.canvas);

    const texture = await Assets.load(spritesPng);
    const sheet = new Spritesheet(texture, spritesData);
    await sheet.parse();

    if (this.destroyed) { app.destroy(true, { children: true }); return; }

    this.sheet = sheet;

    app.stage.addChild(drawBoard(boardSize, sheet));

    // Layer setup
    const shadowLayer = new Container();
    const stoneLayer = new Container();
    const effectsLayer = new Container();
    app.stage.addChild(shadowLayer);
    app.stage.addChild(stoneLayer);
    app.stage.addChild(effectsLayer);
    this.layers = { shadow: shadowLayer, stone: stoneLayer, effects: effectsLayer };

    // Active-move marker
    this.marker = new Marker(sheet, stoneLayer, boardSize);

    // Capture pots
    this.pots = new Pots(sheet, app.stage);

    this.initialized = true;

    if (this.pendingProps) {
      const props = this.pendingProps;
      this.pendingProps = null;
      this.update(props);
    }
  }

  update(props: GoPixiProps): void {
    if (!this.initialized) {
      this.pendingProps = props;
      return;
    }

    const { grid, step, lastPlayed, captures } = props;
    const sheet = this.sheet!;
    const layers = this.layers!;
    const { boardSize } = this;

    // Kill in-flight animations and reset all sprites to rest state
    for (const anim of this.activeAnims) anim.kill();
    this.activeAnims = [];
    for (const pair of this.stoneMap.values()) resetPair(pair);
    this.marker!.reset();
    // Clean up any lingering particle sprites from killed animations
    layers.effects.removeChildren().forEach(c => c.destroy());
    this.pots!.reset();

    const isSingleStep = Math.abs(step - this.prevStep) === 1;
    const { added, removed } = diffGrids(this.prevGrid, grid);

    // Remove captured stones
    for (const { row, col } of removed) {
      const key = posKey(row, col);
      const pair = this.stoneMap.get(key);
      if (pair) {
        this.stoneMap.delete(key);
        if (isSingleStep) {
          layers.shadow.removeChild(pair.shadow);
          layers.stone.removeChild(pair.stone);
          layers.effects.addChild(pair.shadow);
          layers.effects.addChild(pair.stone);
          const tl = animateCapture(pair, sheet, layers.effects);
          this.activeAnims.push(tl);
        } else {
          layers.shadow.removeChild(pair.shadow);
          layers.stone.removeChild(pair.stone);
          pair.shadow.destroy();
          pair.stone.destroy();
        }
      }
    }

    // Add new stones
    for (const { row, col, value } of added) {
      const pair = createStonePair(row, col, value, boardSize, sheet);
      layers.shadow.addChild(pair.shadow);
      layers.stone.addChild(pair.stone);
      this.stoneMap.set(posKey(row, col), pair);

      if (isSingleStep) {
        const tl = animateStoneDrop(pair);
        this.activeAnims.push(tl);

        for (const n of getNeighbors(row, col, boardSize)) {
          const neighborPair = this.stoneMap.get(posKey(n.row, n.col));
          if (neighborPair) {
            const tw = animateNeighborShockwave(neighborPair, n.col - col, n.row - row);
            this.activeAnims.push(tw);
          }
        }
      }
    }

    // Position active-move marker on last-played stone
    this.activeAnims.push(...this.marker!.update(lastPlayed, this.stoneMap, isSingleStep));

    // Update prisoner sprites in pots
    this.activeAnims.push(...this.pots!.update(captures, isSingleStep));

    // Store for next diff
    this.prevGrid = grid;
    this.prevStep = step;
  }

  destroy(): void {
    this.destroyed = true;
    for (const anim of this.activeAnims) anim.kill();
    this.activeAnims = [];
    this.stoneMap = new Map();
    this.prevGrid = null;
    this.pendingProps = null;
    this.sheet = null;
    this.layers = null;
    this.marker = null;
    this.pots = null;
    if (this.initialized) {
      this.app.destroy(true, { children: true });
    }
    this.initialized = false;
  }
}
