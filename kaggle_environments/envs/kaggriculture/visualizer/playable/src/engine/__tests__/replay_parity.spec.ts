/**
 * Full-rollout parity check: replay Python-recorded actions through the TS
 * engine and diff the resulting state against the Python observation at every
 * step. Catches phase-ordering and field-level drift the unit specs miss.
 *
 * Fixtures are produced by running the Python env with the `random` agent at
 * a fixed seed and dumping `env.toJSON()` — see fixtures/README or rebuild
 * with:
 *
 *   python -c "
 *     import json; from kaggle_environments import make
 *     env = make('kaggriculture', debug=True,
 *                configuration={'episodeSteps': 100}, info={'seed': 42})
 *     env.run(['random', 'random'])
 *     json.dump(env.toJSON(), open('parity_seed42.json','w'))
 *   "
 */

import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { step } from '../interpreter';
import { initGameState, resolveConfig } from '../state';
import type { GameState, PlayerAction } from '../types';

interface PyStep {
  action: PlayerAction;
  observation: Record<string, unknown>;
  status: string;
  reward: number | null;
}

interface PyReplay {
  _seed: number;
  configuration: Record<string, unknown>;
  steps: PyStep[][];
}

function loadFixture(name: string): PyReplay {
  const path = join(__dirname, 'fixtures', name);
  return JSON.parse(readFileSync(path, 'utf8')) as PyReplay;
}

/**
 * Normalize a state-or-observation into the comparable public shape.
 *
 * Drops:
 *  - Presentational-only fields (`numAgents`, `scores`, `seed`, `remainingOverageTime`).
 *  - Stochastic outputs that depend on the RNG: `town.unlocked_shops` (random
 *    `rng.choice` over remaining shops) and per-tile WEED spawns (per-tile
 *    coin flips). PyRandom is plain MT19937 and not bit-identical to
 *    CPython's `random.Random` — see rng.ts.
 *  - WEED tiles are normalized to `null` so a Python-but-not-TS (or
 *    vice-versa) weed on an otherwise-empty tile doesn't cascade-fail diffs.
 *  - `market.inventory` and `market.prices`: once shops unlock with different
 *    identities in TS vs Python, town-shop consumption drifts the market on
 *    every subsequent tick. Deterministic market math (price curves, SELL/
 *    BUY/HIRE processing) is covered by unit tests in state.spec.ts and
 *    interpreter.spec.ts.
 *
 * What this still validates end-to-end against Python: movement, PLANT/
 * WATER/HARVEST/DIG/PICKUP/PLACE/BUILD/FEED/COLLECT_FERTILIZER/CARE, daily
 * decay/refresh, end-of-day inventory drop + farmer reset, hire-cost
 * accumulation, money flows, private shed/seed/inventory progression.
 *
 * Round-trips through JSON to flatten Python's int-vs-float distinction
 * (both become `number` in JS).
 */
function stripWeeds(tiles: unknown): unknown {
  if (!Array.isArray(tiles)) return tiles;
  return tiles.map((row) =>
    Array.isArray(row)
      ? row.map((t) => (t && typeof t === 'object' && (t as { kind?: string }).kind === 'WEED' ? null : t))
      : row
  );
}

function normalizePublic(snapshot: Record<string, unknown>) {
  const raw = JSON.parse(JSON.stringify(snapshot));
  const farms = Array.isArray(raw.farms)
    ? raw.farms.map((f: Record<string, unknown>) => ({ ...f, tiles: stripWeeds(f.tiles) }))
    : raw.farms;
  const town = raw.town && typeof raw.town === 'object' ? {} : raw.town; // drop unlocked_shops
  return {
    step: raw.step,
    day: raw.day,
    hour: raw.hour,
    farms,
    market: {}, // RNG cascade drifts market via shop consumption; covered by unit tests
    town,
  };
}

function normalizePrivate(priv: unknown) {
  return JSON.parse(JSON.stringify(priv));
}

function diffStep(label: string, ts: GameState, pyObs: Record<string, unknown>): void {
  const tsPublic = normalizePublic(ts as unknown as Record<string, unknown>);
  const pyPublic = normalizePublic(pyObs);
  expect(tsPublic, `${label} public state diverges`).toEqual(pyPublic);
}

function runParity(fixtureName: string): void {
  const replay = loadFixture(fixtureName);
  const cfg = resolveConfig({
    episodeSteps: replay.configuration.episodeSteps as number,
  });
  const numAgents = replay.steps[0].length;
  let state = initGameState(numAgents, cfg, replay._seed);

  // Step 0: initial state should match initial Python observation.
  diffStep('step 0 init', state, replay.steps[0][0].observation);
  for (let p = 0; p < numAgents; p++) {
    const pyPriv = (replay.steps[0][p].observation as Record<string, unknown>).private;
    expect(normalizePrivate(state.privates[p]), `step 0 private[${p}]`).toEqual(normalizePrivate(pyPriv));
  }

  // In kaggle-environments, `steps[i].action` is the action APPLIED to produce
  // `steps[i].observation` (see core.py: action is attached to action_state,
  // then interpreter runs and the result is appended). So we advance using
  // the next step's recorded actions, not the current one's.
  for (let next = 1; next < replay.steps.length; next++) {
    const actions = replay.steps[next].map((s) => s.action);
    state = step(state, actions, cfg);

    diffStep(`step ${next} public`, state, replay.steps[next][0].observation);
    for (let p = 0; p < numAgents; p++) {
      const pyPriv = (replay.steps[next][p].observation as Record<string, unknown>).private;
      expect(normalizePrivate(state.privates[p]), `step ${next} private[${p}]`).toEqual(normalizePrivate(pyPriv));
    }
  }

  // Final scores should match the rewards reported on the terminal step.
  const last = replay.steps[replay.steps.length - 1];
  expect(state.done, 'terminal step.done').toBe(true);
  expect(state.scores, 'terminal scores match Python rewards').toEqual(last.map((s) => s.reward));
}

describe('full-episode replay parity vs Python', () => {
  it('seed 42, 100 steps, random vs random', () => {
    runParity('parity_seed42.json');
  });

  it('seed 7, 100 steps, random vs random', () => {
    runParity('parity_seed7.json');
  });
});
