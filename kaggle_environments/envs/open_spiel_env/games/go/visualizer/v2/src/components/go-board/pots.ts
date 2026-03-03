import gsap from 'gsap';
import { Container, Sprite, type Spritesheet } from 'pixi.js';
import type { Captures } from './middleman.ts';
import {
  BOARD_PX,
  POT_AREA_HEIGHT,
  POT_MAX_PRISONERS,
  POT_PRISONER_SIZE,
  POT_SCATTER_RADIUS,
  POT_SIZE,
} from './constants.ts';
import { animatePotPoof } from './animate-stones.ts';

const POT_Y = BOARD_PX + POT_AREA_HEIGHT / 2;
const POT_LEFT_X = BOARD_PX * 0.25;
const POT_RIGHT_X = BOARD_PX * 0.75;
const POT_SHADOW_OFFSET = POT_PRISONER_SIZE * 0.06;

interface PrisonerInfo {
  sprite: Sprite;
  shadow: Sprite;
  restScaleX: number;
  restScaleY: number;
  shadowRestScaleX: number;
  shadowRestScaleY: number;
}

export class Pots {
  private left: PrisonerInfo[];
  private right: PrisonerInfo[];
  private potEffectsLayer: Container;
  private sheet: Spritesheet;
  private prevCaptures = { black: 0, white: 0 };

  constructor(sheet: Spritesheet, stage: Container) {
    this.sheet = sheet;

    const potShadowLayer = new Container();
    const potStoneLayer = new Container();
    const potEffectsLayer = new Container();
    stage.addChild(potShadowLayer);
    stage.addChild(potStoneLayer);

    const createPot = (centerX: number, texName: 'white.png' | 'black.png'): PrisonerInfo[] => {
      const pot = new Sprite(sheet.textures['pot.png']);
      pot.anchor.set(0.5);
      pot.position.set(centerX, POT_Y);
      pot.width = POT_SIZE;
      pot.height = POT_SIZE;
      potShadowLayer.addChild(pot);

      const prisoners: PrisonerInfo[] = [];
      for (let i = 0; i < POT_MAX_PRISONERS; i++) {
        const angle = Math.random() * Math.PI * 2;
        const r = POT_SCATTER_RADIUS * Math.sqrt(Math.random());
        const px = centerX + Math.cos(angle) * r;
        const py = POT_Y + Math.sin(angle) * r;

        const shadow = new Sprite(sheet.textures['shadow.png']);
        shadow.anchor.set(0.5);
        shadow.position.set(px - POT_SHADOW_OFFSET, py + POT_SHADOW_OFFSET);
        shadow.width = POT_PRISONER_SIZE;
        shadow.height = POT_PRISONER_SIZE;
        shadow.visible = false;
        potShadowLayer.addChild(shadow);

        const sprite = new Sprite(sheet.textures[texName]);
        sprite.anchor.set(0.5);
        sprite.position.set(px, py);
        sprite.width = POT_PRISONER_SIZE;
        sprite.height = POT_PRISONER_SIZE;
        sprite.visible = false;
        potStoneLayer.addChild(sprite);

        prisoners.push({
          sprite,
          shadow,
          restScaleX: sprite.scale.x,
          restScaleY: sprite.scale.y,
          shadowRestScaleX: shadow.scale.x,
          shadowRestScaleY: shadow.scale.y,
        });
      }
      return prisoners;
    };

    this.left = createPot(POT_LEFT_X, 'white.png');
    this.right = createPot(POT_RIGHT_X, 'black.png');
    this.potEffectsLayer = potEffectsLayer;

    // Pot effects layer on top of everything
    stage.addChild(potEffectsLayer);
  }

  reset(): void {
    this.potEffectsLayer.removeChildren().forEach((c) => c.destroy());
    for (const p of this.left) {
      p.sprite.scale.set(p.restScaleX, p.restScaleY);
      p.shadow.scale.set(p.shadowRestScaleX, p.shadowRestScaleY);
    }
    for (const p of this.right) {
      p.sprite.scale.set(p.restScaleX, p.restScaleY);
      p.shadow.scale.set(p.shadowRestScaleX, p.shadowRestScaleY);
    }
  }

  update(captures: Captures, isSingleStep: boolean): gsap.core.Animation[] {
    const anims: gsap.core.Animation[] = [];
    const { left, right, sheet, potEffectsLayer } = this;

    const leftCount = Math.min(captures.white, POT_MAX_PRISONERS);
    const rightCount = Math.min(captures.black, POT_MAX_PRISONERS);
    const prevLeft = Math.min(this.prevCaptures.white, POT_MAX_PRISONERS);
    const prevRight = Math.min(this.prevCaptures.black, POT_MAX_PRISONERS);

    const randomPotPos = (centerX: number) => {
      const angle = Math.random() * Math.PI * 2;
      const r = POT_SCATTER_RADIUS * Math.sqrt(Math.random());
      return { x: centerX + Math.cos(angle) * r, y: POT_Y + Math.sin(angle) * r };
    };

    const revealPrisoners = (list: PrisonerInfo[], prev: number, count: number, centerX: number) => {
      for (let i = 0; i < list.length; i++) {
        const visible = i < count;
        list[i].sprite.visible = visible;
        list[i].shadow.visible = visible;
        if (i >= prev && i < count) {
          const pos = randomPotPos(centerX);
          list[i].sprite.position.set(pos.x, pos.y);
          list[i].shadow.position.set(pos.x - POT_SHADOW_OFFSET, pos.y + POT_SHADOW_OFFSET);
        }
        if (!isSingleStep) {
          list[i].sprite.scale.set(list[i].restScaleX, list[i].restScaleY);
          list[i].shadow.scale.set(list[i].shadowRestScaleX, list[i].shadowRestScaleY);
        }
      }
    };
    revealPrisoners(left, prevLeft, leftCount, POT_LEFT_X);
    revealPrisoners(right, prevRight, rightCount, POT_RIGHT_X);

    // Scale-in + poof for newly arrived prisoners
    if (isSingleStep) {
      const animatePrisoner = (info: PrisonerInfo) => {
        const delay = Math.random() * 0.15;
        info.sprite.scale.set(0, 0);
        info.shadow.scale.set(0, 0);
        anims.push(
          gsap.to(info.sprite.scale, {
            x: info.restScaleX,
            y: info.restScaleY,
            duration: 0.4,
            delay,
            ease: 'elastic.out(1, 0.4)',
          })
        );
        anims.push(
          gsap.to(info.shadow.scale, {
            x: info.shadowRestScaleX,
            y: info.shadowRestScaleY,
            duration: 0.4,
            delay,
            ease: 'elastic.out(1, 0.4)',
          })
        );
        anims.push(animatePotPoof(info.sprite.x, info.sprite.y, sheet, potEffectsLayer));
      };
      for (let i = prevLeft; i < leftCount; i++) animatePrisoner(left[i]);
      for (let i = prevRight; i < rightCount; i++) animatePrisoner(right[i]);
    }

    // Store for next diff
    this.prevCaptures = { black: captures.black, white: captures.white };

    return anims;
  }
}
