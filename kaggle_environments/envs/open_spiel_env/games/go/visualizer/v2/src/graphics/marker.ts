import gsap from 'gsap';
import { Sprite, type Container, type Spritesheet } from 'pixi.js';
import type { GridPos } from '../types/game.ts';
import { getCellSize } from './constants.ts';
import { posKey, type StoneMap } from './stone-map.ts';

const MARKER_POP_SCALE = 1.6;
const MARKER_POP_ROTATION = 0.4;
const MARKER_SCALE_DURATION = 0.5;
const MARKER_ROTATION_DURATION = 0.6;

export class Marker {
  private sprite: Sprite;
  private restScaleX: number;
  private restScaleY: number;
  private sheet: Spritesheet;
  private stoneLayer: Container;

  constructor(sheet: Spritesheet, stoneLayer: Container, boardSize: number) {
    this.sheet = sheet;
    this.stoneLayer = stoneLayer;

    const markerSize = getCellSize(boardSize) * 0.88 * 0.45;
    const sprite = new Sprite(sheet.textures['black-active-marker.png']);
    sprite.anchor.set(0.5);
    sprite.width = markerSize;
    sprite.height = markerSize;
    sprite.visible = false;
    stoneLayer.addChild(sprite);

    this.sprite = sprite;
    this.restScaleX = sprite.scale.x;
    this.restScaleY = sprite.scale.y;
  }

  reset(): void {
    this.sprite.scale.set(this.restScaleX, this.restScaleY);
    this.sprite.rotation = 0;
  }

  update(lastPlayed: GridPos | null, stoneMap: StoneMap, isSingleStep: boolean): gsap.core.Animation[] {
    const anims: gsap.core.Animation[] = [];
    const { sprite, sheet, stoneLayer, restScaleX, restScaleY } = this;

    const lastPlayedKey = lastPlayed ? posKey(lastPlayed.row, lastPlayed.col) : null;
    const lastPair = lastPlayedKey ? stoneMap.get(lastPlayedKey) : undefined;

    if (lastPair) {
      sprite.texture = sheet.textures[lastPair.value === 'B' ? 'black-active-marker.png' : 'white-active-marker.png'];
      sprite.position.set(lastPair.stoneRest.x, lastPair.stoneRest.y);
      sprite.visible = true;
      stoneLayer.addChild(sprite);

      if (isSingleStep) {
        sprite.scale.set(restScaleX * MARKER_POP_SCALE, restScaleY * MARKER_POP_SCALE);
        sprite.rotation = MARKER_POP_ROTATION;
        const tl = gsap.timeline();
        tl.to(
          sprite.scale,
          {
            x: restScaleX,
            y: restScaleY,
            duration: MARKER_SCALE_DURATION,
            ease: 'elastic.out(1, 0.4)',
          },
          0
        );
        tl.to(
          sprite,
          {
            rotation: 0,
            duration: MARKER_ROTATION_DURATION,
            ease: 'elastic.out(1, 0.3)',
          },
          0
        );
        anims.push(tl);
      } else {
        sprite.scale.set(restScaleX, restScaleY);
        sprite.rotation = 0;
      }
    } else {
      sprite.visible = false;
    }

    return anims;
  }
}
