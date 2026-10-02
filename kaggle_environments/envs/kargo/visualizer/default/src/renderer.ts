import type { RendererOptions } from '@kaggle-environments/core';
import { paintScreen, screenRefs, type ScreenRefs } from './chrome';
import { buildShell, collectRefs, renderObservation, type CityRefs } from './renderCity';
import { buildNightShell, nightRefs, renderNight, type NightRefs } from './renderNight';
import { KARGO_STEP_DURATION } from './timing';
import { buildView } from './utils';

interface CachedShell {
  key: string;
  screen: ScreenRefs;
  city?: CityRefs;
  night?: NightRefs;
  /** Last step rendered here, and when -- how the trail's tempo is inferred. */
  lastStep?: number;
  lastAt?: number;
}

const shellCache = new WeakMap<HTMLElement, CachedShell>();

/** How long the truck trail should take to play, in ms.
 *
 * The host gives a renderer no play state and no speed, so the cadence is
 * measured: the gap between this call and the previous one IS the step
 * duration the player is currently running at, whatever the speed control says.
 *
 * Only a step that follows the one before it animates. A scrub, a restart or a
 * jump backwards is a change of position, not a trip, and sliding the trucks
 * across town to meet it would be a lie about time. The first step after a load
 * has nothing to measure against and so plays static.
 */
function trailDuration(cached: CachedShell, step: number, now: number): number {
  const consecutive = cached.lastStep === step - 1 && cached.lastAt != null;
  const gap = consecutive ? now - (cached.lastAt as number) : 0;
  cached.lastStep = step;
  cached.lastAt = now;
  // The ceiling is derived from the slowest speed the player offers (0.25x),
  // not a flat 3s: at an 800ms step that is a 3.2s tick, and a hardcoded 3s
  // would read the slowest setting as "paused" and switch the animation off --
  // turning the speed down would stop the trucks moving. A little headroom
  // over that for a slow frame. Under ~80ms the animation is shorter than a
  // couple of frames and not worth starting.
  const slowest = KARGO_STEP_DURATION * 4 * 1.25;
  if (gap < 80 || gap > slowest) return 0;
  // Stop a shade early: the marker should be parked and settled at the block's
  // end position before the next step replaces it, not still gliding.
  return gap * 0.9;
}

/** Two screens: the map while trucks run, the board while agents trade.
 *
 * Three of every eight steps are markets -- CAPEX, LABOR, CONTRACTS -- and on
 * the map those are a still frame. The decisions made in them are the game, so
 * they get their own screen and the map is not asked to stand in for one.
 *
 * The shell is built once per screen and keyed on what would invalidate it. The
 * road network is 760 static paths whose only per-step change is a stroke
 * colour; rebuilding it would reparse ~1500 SVG nodes for a recolour, and
 * scrubbing the timeline would stutter. The key includes the screen, so
 * crossing from 18:00 into CAPEX swaps the markup and nothing else does.
 */
export function renderer(options: RendererOptions): void {
  const { parent, replay, step, agents } = options;
  if (!parent || !replay) return;

  // Seat count comes from the replay, not a constant: kargo ships 2- and
  // 4-player tables and a step holds one entry per seat. Reading it off
  // `agents` instead would collapse to two whenever the host passes no names.
  const seats = Math.max(1, (replay?.steps?.[step] as unknown as unknown[] | undefined)?.length ?? 0);
  const names = Array.from({ length: seats }, (_v, i) => agents?.[i]?.name || `Player ${i + 1}`);
  const view = buildView(replay, step, names);
  if (!view) return;

  const isNight = view.phase !== 'DRIVING';
  const key = isNight
    ? `night|${names.join('|')}`
    : `city|${view.city.size}|${view.city.edges.length}|${names.join('|')}`;

  let cached = shellCache.get(parent);
  if (!cached || cached.key !== key) {
    // The tempo carries across a shell swap. Every day crosses from CAPEX back
    // to the map, and dropping the timing there would make the first driving
    // block of every day the one that does not animate.
    const { lastStep, lastAt } = cached ?? {};
    if (isNight) {
      buildNightShell(parent, names);
      cached = { key, screen: screenRefs(parent), night: nightRefs(parent), lastStep, lastAt };
    } else {
      buildShell(parent, view.city, names);
      cached = { key, screen: screenRefs(parent), city: collectRefs(parent, view.city.size), lastStep, lastAt };
    }
    shellCache.set(parent, cached);
  }

  const trailMs = trailDuration(cached, step, performance.now());

  paintScreen(cached.screen, view);
  if (cached.night) renderNight(cached.night, view);
  if (cached.city) renderObservation(cached.city, view, trailMs);
}
