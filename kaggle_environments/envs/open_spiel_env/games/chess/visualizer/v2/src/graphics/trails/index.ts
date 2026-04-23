import { Sprite } from 'pixi.js';
import type { Engine } from '../engine';
import type { PieceType } from '../constants';
import { MOVE_THRESHOLD, TRAIL_CONFIGS, type TrailConfig } from './config';

interface PieceTrackingState {
  x: number;
  y: number;
  lastSpawn: number;
  liveCount: number;
  hasBurst: boolean;
}

interface LiveParticle {
  sprite: Sprite;
  piece: Sprite;
  spawnTime: number;
  lifetime: number;
  startAlpha: number;
  startScale: number;
  originX: number;
  originY: number;
  driftDx: number;
  driftDy: number;
  followLerp: number;
  align: boolean;
  spin: boolean;
  baseRotation: number;
}

export interface TrailSystem {
  clear: () => void;
  destroy: () => void;
}

function rand(min: number, max: number): number {
  return min + Math.random() * (max - min);
}

function easeOutQuad(t: number): number {
  return t * (2 - t);
}

const LERP_REF_FPS = 60;
const DIRECTION_THRESHOLD_SQ = 0.01;

export function createTrails(engine: Engine): TrailSystem {
  const tracking = new Map<Sprite, PieceTrackingState>();
  const particles: LiveParticle[] = [];

  function spawn(
    piece: Sprite,
    config: TrailConfig,
    x: number,
    y: number,
    now: number,
    state: PieceTrackingState,
    dx: number,
    dy: number
  ) {
    const { squareSize, textures, resources } = engine;
    const textureName = config.textures[Math.floor(Math.random() * config.textures.length)];
    const texture = textures[textureName];
    if (!texture) return;

    const sizeFactor = rand(config.size[0], config.size[1]);
    const scale = (squareSize * sizeFactor) / texture.width;

    const directionAligned = config.spin || config.align;
    const baseRotation = directionAligned && (dx !== 0 || dy !== 0) ? Math.atan2(dy, dx) + Math.PI / 2 : 0;

    const sprite = new Sprite({ texture, anchor: 0.5 });
    sprite.scale.set(scale);
    sprite.alpha = config.alpha;
    sprite.rotation = baseRotation;

    let spawnX = x;
    let spawnY = y;
    if (config.scatter) {
      const angle = Math.random() * Math.PI * 2;
      const dist = Math.random() * config.scatter * squareSize;
      spawnX += Math.cos(angle) * dist;
      spawnY += Math.sin(angle) * dist;
    }
    sprite.position.set(spawnX, spawnY);

    let driftDx = 0;
    let driftDy = 0;
    let followLerp = 0;

    if (config.follow != null) {
      followLerp = config.follow;
    } else if (config.drift) {
      const angle = Math.random() * Math.PI * 2;
      const radius = config.drift * squareSize;
      driftDx = Math.cos(angle) * radius;
      driftDy = Math.sin(angle) * radius;
    }

    resources.trails.addChild(sprite);
    state.liveCount++;

    particles.push({
      sprite,
      piece,
      spawnTime: now,
      lifetime: config.lifetime,
      startAlpha: config.alpha,
      startScale: scale,
      originX: x,
      originY: y,
      driftDx,
      driftDy,
      followLerp,
      align: config.align ?? false,
      spin: config.spin ?? false,
      baseRotation,
    });
  }

  function pollPieces(now: number) {
    for (const child of engine.resources.animating.children) {
      pollSprite(child as Sprite, now);
    }
  }

  function pollSprite(sprite: Sprite, now: number) {
    const config = TRAIL_CONFIGS[sprite.label as PieceType];
    if (!config) return;

    const x = sprite.position.x;
    const y = sprite.position.y;

    const state = tracking.get(sprite);
    if (!state) {
      tracking.set(sprite, { x, y, lastSpawn: now, liveCount: 0, hasBurst: false });
      return;
    }

    const dx = x - state.x;
    const dy = y - state.y;
    state.x = x;
    state.y = y;

    if (dx * dx + dy * dy < MOVE_THRESHOLD * MOVE_THRESHOLD) return;
    if (config.burst && state.hasBurst) return;
    if (config.rate > 0 && now - state.lastSpawn < config.rate) return;
    if (config.max != null && state.liveCount >= config.max) return;

    state.lastSpawn = now;
    if (config.burst) state.hasBurst = true;

    for (let i = 0; i < config.count; i++) {
      spawn(sprite, config, x, y, now, state, dx, dy);
    }
  }

  function updateParticles(now: number, dt: number) {
    for (let i = particles.length - 1; i >= 0; i--) {
      const p = particles[i];
      const age = now - p.spawnTime;

      if (age >= p.lifetime) {
        engine.resources.trails.removeChild(p.sprite);
        const state = tracking.get(p.piece);
        if (state) state.liveCount--;
        particles.splice(i, 1);
        continue;
      }

      // Position
      if (p.followLerp > 0) {
        if (p.align) {
          const pdx = p.piece.position.x - p.sprite.x;
          const pdy = p.piece.position.y - p.sprite.y;
          if (pdx * pdx + pdy * pdy > DIRECTION_THRESHOLD_SQ) {
            p.sprite.rotation = Math.atan2(pdy, pdx) + Math.PI / 2;
          }
        }
        const factor = 1 - Math.pow(1 - p.followLerp, dt * LERP_REF_FPS);
        p.sprite.x += (p.piece.position.x - p.sprite.x) * factor;
        p.sprite.y += (p.piece.position.y - p.sprite.y) * factor;
      } else if (p.driftDx !== 0 || p.driftDy !== 0) {
        const progress = Math.min(age / p.lifetime, 1);
        const eased = easeOutQuad(progress);
        p.sprite.x = p.originX + p.driftDx * eased;
        p.sprite.y = p.originY + p.driftDy * eased;
      }

      // Fade + scale (ease-in quad — stays opaque early, fades fast at end)
      const t = Math.min(age / p.lifetime, 1);
      const remaining = 1 - t * t;
      p.sprite.alpha = p.startAlpha * remaining;
      p.sprite.scale.set(p.startScale * remaining);

      if (p.spin && !p.align) {
        p.sprite.rotation = p.baseRotation + t;
      }
    }
  }

  function tick() {
    const now = performance.now() / 1000;
    const dt = engine.app.ticker.deltaMS / 1000;
    pollPieces(now);
    updateParticles(now, dt);
  }

  engine.app.ticker.add(tick);

  return {
    clear() {
      engine.resources.trails.removeChildren();
      particles.length = 0;
      tracking.clear();
    },
    destroy() {
      this.clear();
      engine.app.ticker.remove(tick);
    },
  };
}
