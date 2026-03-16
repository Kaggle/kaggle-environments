import gsap from 'gsap';
import { Sprite, type Container, type Spritesheet } from 'pixi.js';
import type { StonePair } from './stone-map.ts';

// Drop animation
const DROP_SCALE = 1.5;
const DROP_FADE_DURATION = 0.12;
const DROP_SCALE_DURATION = 0.675;
const DROP_ROTATION = 0.6;
const DROP_ROTATION_DURATION = 0.9;
const DROP_ROTATION_DELAY = 0.1;
const SHADOW_LIFT_OFFSET = 4;

// Capture animation
const CAPTURE_SHRINK_DURATION = 0.15;
const CAPTURE_PARTICLE_COUNT = 6;
const CAPTURE_PARTICLE_SPREAD = 20;
const CAPTURE_PARTICLE_DURATION = 0.45;
const CAPTURE_PARTICLE_MAX_SCALE = 0.18;
const CAPTURE_PARTICLE_STAGGER = 0.03;
// Per-invocation variation ranges (multiply base values)
const CAPTURE_SPREAD_JITTER = 0.3; // ±30% on spread
const CAPTURE_DURATION_JITTER = 0.2; // ±20% on particle duration
const CAPTURE_SHRINK_JITTER = 0.25; // ±25% on shrink duration

// Territory marker animation
const TERRITORY_SCALE_DURATION = 0.2;

// Atari shiver animation
const ATARI_SHIVER_OFFSET = 0.6;
const ATARI_SHIVER_DURATION = 0.06;
const ATARI_SHIVER_STAGGER_MAX = 0.3;

// Shockwave animation
const SHOCKWAVE_PUSH_DISTANCE = 4;
const SHOCKWAVE_PUSH_DURATION = 0.12;
const SHOCKWAVE_RETURN_DURATION = 0.525;
const SHOCKWAVE_PUSH_DELAY = 0.06;
const SHOCKWAVE_RETURN_DELAY = 0.18;
const SHOCKWAVE_SHADOW_PUSH_DELAY = 0.09;
const SHOCKWAVE_SHADOW_RETURN_DELAY = 0.21;
const SHOCKWAVE_SHADOW_RETURN_DURATION = 0.45;

export function animateStoneDrop(pair: StonePair): gsap.core.Timeline {
  const tl = gsap.timeline();

  const { stoneRest, shadowRest } = pair;
  const restSx = stoneRest.scaleX;
  const restSy = stoneRest.scaleY;
  const shadowRestSx = shadowRest.scaleX;
  const shadowRestSy = shadowRest.scaleY;

  // Stone: oversized scale snaps down to rest
  pair.stone.scale.set(restSx * DROP_SCALE, restSy * DROP_SCALE);
  pair.stone.alpha = 0;

  tl.to(
    pair.stone,
    {
      alpha: 1,
      duration: DROP_FADE_DURATION,
      ease: 'power2.out',
    },
    0
  );
  tl.to(
    pair.stone.scale,
    {
      x: restSx,
      y: restSy,
      duration: DROP_SCALE_DURATION,
      ease: 'elastic.out(1, 0.4)',
    },
    0
  );

  // Stone: rotational wobble on landing
  pair.stone.rotation = DROP_ROTATION;
  tl.to(
    pair.stone,
    {
      rotation: 0,
      duration: DROP_ROTATION_DURATION,
      ease: 'elastic.out(1, 0.25)',
    },
    DROP_ROTATION_DELAY
  );

  // Shadow: matches stone scale, but starts offset further away
  // (stone "up high" casts shadow further from light source)
  const shadowRestX = shadowRest.x;
  const shadowRestY = shadowRest.y;
  pair.shadow.scale.set(shadowRestSx * DROP_SCALE, shadowRestSy * DROP_SCALE);
  pair.shadow.alpha = 0;
  pair.shadow.x = shadowRestX - SHADOW_LIFT_OFFSET;
  pair.shadow.y = shadowRestY + SHADOW_LIFT_OFFSET;

  tl.to(
    pair.shadow,
    {
      alpha: 1,
      duration: DROP_FADE_DURATION,
      ease: 'power2.out',
    },
    0
  );
  tl.to(
    pair.shadow,
    {
      x: shadowRestX,
      y: shadowRestY,
      duration: DROP_SCALE_DURATION,
      ease: 'elastic.out(1, 0.4)',
    },
    0
  );
  tl.to(
    pair.shadow.scale,
    {
      x: shadowRestSx,
      y: shadowRestSy,
      duration: DROP_SCALE_DURATION,
      ease: 'elastic.out(1, 0.4)',
    },
    0
  );

  return tl;
}

const PUFF_TEXTURES = ['puff1.png', 'puff2.png', 'puff3.png'] as const;

export function animateCapture(pair: StonePair, sheet: Spritesheet, effectsLayer: Container): gsap.core.Timeline {
  const tl = gsap.timeline();
  const { stone, shadow, stoneRest } = pair;
  const particles: Sprite[] = [];

  // Per-invocation variation so each capture feels slightly different
  const jitter = (range: number) => 1 + (Math.random() * 2 - 1) * range;
  const shrinkDur = CAPTURE_SHRINK_DURATION * jitter(CAPTURE_SHRINK_JITTER);
  const particleDur = CAPTURE_PARTICLE_DURATION * jitter(CAPTURE_DURATION_JITTER);
  const spread = CAPTURE_PARTICLE_SPREAD * jitter(CAPTURE_SPREAD_JITTER);

  // Shrink + fade stone and shadow
  tl.to(
    stone,
    {
      alpha: 0,
      duration: shrinkDur,
      ease: 'power2.in',
    },
    0
  );
  tl.to(
    stone.scale,
    {
      x: 0,
      y: 0,
      duration: shrinkDur,
      ease: 'power2.in',
    },
    0
  );
  tl.to(
    shadow,
    {
      alpha: 0,
      duration: shrinkDur,
      ease: 'power2.in',
    },
    0
  );
  tl.to(
    shadow.scale,
    {
      x: 0,
      y: 0,
      duration: shrinkDur,
      ease: 'power2.in',
    },
    0
  );

  // Spawn puff particles
  for (let i = 0; i < CAPTURE_PARTICLE_COUNT; i++) {
    const texName = PUFF_TEXTURES[Math.floor(Math.random() * PUFF_TEXTURES.length)];
    const puff = new Sprite(sheet.textures[texName]);
    puff.anchor.set(0.5);
    puff.position.set(stoneRest.x, stoneRest.y);
    puff.scale.set(0);
    puff.alpha = 1;
    effectsLayer.addChild(puff);
    particles.push(puff);

    const angle = (Math.PI * 2 * i) / CAPTURE_PARTICLE_COUNT + (Math.random() - 0.5) * 0.8;
    const dist = spread * (0.5 + Math.random() * 0.5);
    const targetX = stoneRest.x + Math.cos(angle) * dist;
    const targetY = stoneRest.y + Math.sin(angle) * dist;
    const peakScale = CAPTURE_PARTICLE_MAX_SCALE * (0.5 + Math.random() * 0.5);
    const stagger = i * CAPTURE_PARTICLE_STAGGER;

    // Scale up then down
    tl.to(
      puff.scale,
      {
        x: peakScale,
        y: peakScale,
        duration: particleDur * 0.4,
        ease: 'power2.out',
      },
      stagger
    );
    tl.to(
      puff.scale,
      {
        x: 0,
        y: 0,
        duration: particleDur * 0.6,
        ease: 'power2.in',
      },
      stagger + particleDur * 0.4
    );

    // Move outward
    tl.to(
      puff,
      {
        x: targetX,
        y: targetY,
        duration: particleDur,
        ease: 'power2.out',
      },
      stagger
    );

    // Fade out
    tl.to(
      puff,
      {
        alpha: 0,
        duration: particleDur * 0.5,
        ease: 'power2.in',
      },
      stagger + particleDur * 0.5
    );
  }

  // Cleanup on complete
  tl.call(() => {
    stone.destroy();
    shadow.destroy();
    for (const p of particles) {
      effectsLayer.removeChild(p);
      p.destroy();
    }
  });

  return tl;
}

export function animateTerritoryIn(sprite: Sprite, restScaleX: number, restScaleY: number): gsap.core.Tween {
  sprite.scale.set(0, 0);
  return gsap.to(sprite.scale, {
    x: restScaleX,
    y: restScaleY,
    duration: TERRITORY_SCALE_DURATION,
    ease: 'back.out(2)',
  });
}

export function animateTerritoryOut(sprite: Sprite, container: Container): gsap.core.Tween {
  return gsap.to(sprite.scale, {
    x: 0,
    y: 0,
    duration: TERRITORY_SCALE_DURATION,
    ease: 'power2.in',
    onComplete: () => {
      container.removeChild(sprite);
      sprite.destroy();
    },
  });
}

export function animateAtariWobble(pair: StonePair): gsap.core.Timeline {
  const tl = gsap.timeline({
    repeat: -1,
    yoyo: true,
    delay: Math.random() * ATARI_SHIVER_STAGGER_MAX,
  });

  const { x: sx, y: sy } = pair.stoneRest;
  const { x: shx, y: shy } = pair.shadowRest;

  // Rapid, tight positional jitter — trembling rather than swaying
  tl.fromTo(
    pair.stone,
    { x: sx - ATARI_SHIVER_OFFSET, y: sy - ATARI_SHIVER_OFFSET * 0.5 },
    { x: sx + ATARI_SHIVER_OFFSET, y: sy + ATARI_SHIVER_OFFSET * 0.5, duration: ATARI_SHIVER_DURATION, ease: 'none' }
  );
  tl.fromTo(
    pair.shadow,
    { x: shx - ATARI_SHIVER_OFFSET, y: shy - ATARI_SHIVER_OFFSET * 0.5 },
    { x: shx + ATARI_SHIVER_OFFSET, y: shy + ATARI_SHIVER_OFFSET * 0.5, duration: ATARI_SHIVER_DURATION, ease: 'none' },
    0
  );

  return tl;
}

export function animateNeighborShockwave(pair: StonePair, dx: number, dy: number): gsap.core.Timeline {
  const tl = gsap.timeline();
  const origX = pair.stoneRest.x;
  const origY = pair.stoneRest.y;
  const shadowOrigX = pair.shadowRest.x;
  const shadowOrigY = pair.shadowRest.y;

  // Push stone away from source
  tl.to(
    pair.stone,
    {
      x: origX + dx * SHOCKWAVE_PUSH_DISTANCE,
      y: origY + dy * SHOCKWAVE_PUSH_DISTANCE,
      duration: SHOCKWAVE_PUSH_DURATION,
      ease: 'power2.out',
    },
    SHOCKWAVE_PUSH_DELAY
  );
  tl.to(
    pair.stone,
    {
      x: origX,
      y: origY,
      duration: SHOCKWAVE_RETURN_DURATION,
      ease: 'elastic.out(1, 0.45)',
    },
    SHOCKWAVE_RETURN_DELAY
  );

  // Shadow follows with slight lag
  tl.to(
    pair.shadow,
    {
      x: shadowOrigX + dx * SHOCKWAVE_PUSH_DISTANCE,
      y: shadowOrigY + dy * SHOCKWAVE_PUSH_DISTANCE,
      duration: SHOCKWAVE_PUSH_DURATION,
      ease: 'power2.out',
    },
    SHOCKWAVE_SHADOW_PUSH_DELAY
  );
  tl.to(
    pair.shadow,
    {
      x: shadowOrigX,
      y: shadowOrigY,
      duration: SHOCKWAVE_SHADOW_RETURN_DURATION,
      ease: 'elastic.out(1, 0.45)',
    },
    SHOCKWAVE_SHADOW_RETURN_DELAY
  );

  return tl;
}
