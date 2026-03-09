import { Application, Assets, Container, Spritesheet } from 'pixi.js';
import { Sprite } from 'pixi.js';
import type { CellValue, Captures, GridPos, Territory } from '../types/game.ts';
import { BOARD_PX, POT_AREA_HEIGHT, getCellSize, getNeighbors, gridToPixel } from './constants.ts';
import { drawBoard } from './draw-board.ts';
import { diffGrids } from './diff-grid.ts';
import { createStonePair, posKey, resetPair, type StoneMap } from './stone-map.ts';
import {
  animateAtariWobble,
  animateCapture,
  animateNeighborShockwave,
  animateStoneDrop,
  animateTerritoryIn,
  animateTerritoryOut,
} from './animate-stones.ts';
import { Marker } from './marker.ts';
import { Pots } from './pots.ts';
import spritesData from './sprites/sprites.json';
import spritesPng from './sprites/sprites.png';

export interface GoPixiProps {
  grid: CellValue[][];
  step: number;
  lastPlayed: GridPos | null;
  captures: Captures;
  atari: GridPos[];
  territory: Territory;
  reducedMotion: boolean;
}

export class GoPixi {
  private app: Application;
  private boardSize: number;
  private container: HTMLElement;

  private sheet: Spritesheet | null = null;
  private layers: { shadow: Container; territory: Container; stone: Container; effects: Container } | null = null;
  private marker: Marker | null = null;
  private pots: Pots | null = null;

  private stoneMap: StoneMap = new Map();
  private prevTerritoryMap = new Map<string, string>();
  private territorySprites = new Map<string, Sprite>();
  private prevGrid: CellValue[][] | null = null;
  private prevStep = 0;
  private activeAnims: gsap.core.Animation[] = [];
  private atariAnims: gsap.core.Animation[] = [];

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

    if (this.destroyed) {
      app.destroy(true, { children: true });
      return;
    }

    this.container.appendChild(app.canvas);

    const texture = await Assets.load(spritesPng);
    const sheet = new Spritesheet(texture, spritesData);
    await sheet.parse();

    if (this.destroyed) {
      app.destroy(true, { children: true });
      return;
    }

    this.sheet = sheet;

    app.stage.addChild(drawBoard(boardSize, sheet));

    // Layer setup
    const shadowLayer = new Container();
    const territoryLayer = new Container();
    const stoneLayer = new Container();
    const effectsLayer = new Container();
    app.stage.addChild(shadowLayer);
    app.stage.addChild(territoryLayer);
    app.stage.addChild(stoneLayer);
    app.stage.addChild(effectsLayer);
    this.layers = { shadow: shadowLayer, territory: territoryLayer, stone: stoneLayer, effects: effectsLayer };

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

  private killAnimations(): void {
    for (const anim of this.activeAnims) anim.kill();
    for (const anim of this.atariAnims) anim.kill();
    this.activeAnims = [];
    this.atariAnims = [];
  }

  private resetScene(): void {
    this.killAnimations();
    for (const pair of this.stoneMap.values()) resetPair(pair);
    this.marker?.reset();
    if (this.layers) {
      for (const c of this.layers.effects.removeChildren()) c.destroy();
    }
    this.pots?.reset();
  }

  private updateTerritory(
    territory: Territory,
    isSingleStep: boolean,
    sheet: Spritesheet,
    layers: NonNullable<GoPixi['layers']>
  ): void {
    const territorySize = getCellSize(this.boardSize) * 0.3;
    const nextTerritoryMap = new Map<string, string>();
    const nextTerritorySprites = new Map<string, Sprite>();

    const place = (positions: GridPos[], texName: string) => {
      for (const { row, col } of positions) {
        const key = posKey(row, col);
        nextTerritoryMap.set(key, texName);

        const existing = this.territorySprites.get(key);
        if (existing && this.prevTerritoryMap.get(key) === texName) {
          nextTerritorySprites.set(key, existing);
          this.territorySprites.delete(key);
        } else {
          if (existing) {
            layers.territory.removeChild(existing);
            existing.destroy();
            this.territorySprites.delete(key);
          }
          const sprite = new Sprite(sheet.textures[texName]);
          sprite.anchor.set(0.5);
          const { x, y } = gridToPixel(row, col, this.boardSize);
          sprite.position.set(x, y);
          sprite.width = territorySize;
          sprite.height = territorySize;
          layers.territory.addChild(sprite);
          nextTerritorySprites.set(key, sprite);
          if (isSingleStep) {
            this.activeAnims.push(animateTerritoryIn(sprite, sprite.scale.x, sprite.scale.y));
          }
        }
      }
    };
    place(territory.black, 'black-territory.png');
    place(territory.white, 'white-territory.png');

    for (const [, sprite] of this.territorySprites) {
      if (isSingleStep) {
        this.activeAnims.push(animateTerritoryOut(sprite, layers.territory));
      } else {
        layers.territory.removeChild(sprite);
        sprite.destroy();
      }
    }
    this.prevTerritoryMap = nextTerritoryMap;
    this.territorySprites = nextTerritorySprites;
  }

  update(props: GoPixiProps): void {
    if (!this.initialized || !this.sheet || !this.layers || !this.marker || !this.pots) {
      this.pendingProps = props;
      return;
    }

    const { grid, step, lastPlayed, captures, atari, territory, reducedMotion } = props;
    const { sheet, layers, marker, pots, boardSize } = this;

    this.resetScene();

    const isSingleStep = !reducedMotion && Math.abs(step - this.prevStep) === 1;
    const { added, removed } = diffGrids(this.prevGrid, grid);

    // Remove captured stones
    for (const { row, col } of removed) {
      const key = posKey(row, col);
      const pair = this.stoneMap.get(key);
      if (pair) {
        this.stoneMap.delete(key);
        layers.shadow.removeChild(pair.shadow);
        layers.stone.removeChild(pair.stone);
        if (isSingleStep) {
          layers.effects.addChild(pair.shadow);
          layers.effects.addChild(pair.stone);
          this.activeAnims.push(animateCapture(pair, sheet, layers.effects));
        } else {
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
        this.activeAnims.push(animateStoneDrop(pair));

        for (const n of getNeighbors(row, col, boardSize)) {
          const neighborPair = this.stoneMap.get(posKey(n.row, n.col));
          if (neighborPair) {
            this.activeAnims.push(animateNeighborShockwave(neighborPair, n.col - col, n.row - row));
          }
        }
      }
    }

    this.activeAnims.push(...marker.update(lastPlayed, this.stoneMap, isSingleStep));
    this.activeAnims.push(...pots.update(captures, isSingleStep));
    this.updateTerritory(territory, isSingleStep, sheet, layers);

    // Wobble stones in atari (tracked separately — these loop infinitely)
    if (!reducedMotion) {
      for (const { row, col } of atari) {
        const pair = this.stoneMap.get(posKey(row, col));
        if (pair) {
          this.atariAnims.push(animateAtariWobble(pair));
        }
      }
    }

    this.prevGrid = grid;
    this.prevStep = step;
  }

  destroy(): void {
    this.destroyed = true;
    this.resetScene();
    this.stoneMap = new Map();
    this.prevTerritoryMap = new Map();
    this.territorySprites = new Map();
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
