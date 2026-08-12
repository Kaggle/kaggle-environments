// Temporary differential-test driver: replays a Python-generated action script
// through the TS engine and prints matching snapshots for comparison.
import { readFileSync } from 'node:fs';
import { resolveConfig, initGameState } from '../state';
import { step } from '../interpreter';
import type { GameState, PlayerAction } from '../types';

const data = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const config = resolveConfig(data.config);

function snap(s: GameState) {
  return {
    step: s.step,
    day: s.day,
    hour: s.hour,
    farms: s.farms,
    privates: s.privates,
    market: s.market,
    town: s.town,
  };
}

let state = initGameState(2, config, config.seed ?? 7);
const out = [snap(state)];
for (const actions of data.scripts as PlayerAction[][]) {
  state = step(state, actions, config);
  out.push(snap(state));
}
process.stdout.write(JSON.stringify(out));
