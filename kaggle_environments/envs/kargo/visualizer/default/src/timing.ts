import { defaultGetStepRenderTime } from '@kaggle-environments/core';
import type { BaseGameStep, ReplayMode } from '@kaggle-environments/core';

/** Milliseconds a step holds at 1x.
 *
 * 260ms was set when a step was a still frame and the only cost of a slow
 * tempo was sitting through it. Now a driving step plays the block's whole
 * trip: at 260ms a truck crosses four intersections and works a territory in
 * less than a third of a second, which is a flicker, not a journey. Nothing is
 * legible at that rate and the map reads as noise.
 *
 * 800ms is paced off the thing being watched rather than off the step count --
 * long enough that a hop between intersections is a movement the eye can
 * follow and a kerb dwell reads as a pause. A 30-day episode is 241 steps, so
 * about three minutes at 1x; the speed control goes to 4x for anyone who wants
 * the old tempo back, which is the right way round -- fast is a choice, frantic
 * should not be the default.
 */
export const KARGO_STEP_DURATION = 800;

/** The overnight phases hold for less time than a driving block.
 *
 * Three of every eight steps are CAPEX, LABOR and CONTRACTS, and none of them
 * animates -- the night screen is a board that is either read or scrolled past,
 * and holding it for the same 800ms a moving truck needs is dead air three
 * times a day. Pausing is how you read a board; the timer should be paced by
 * the screen that actually moves.
 */
const NIGHT_STEP_DURATION = 420;

/** Whether this step draws the map. Mirrors `renderer`'s own screen split.
 *
 * Read defensively: the host hands the timing hook its own processed step
 * shape, and the raw kargo observation is not part of the `BaseGameStep`
 * contract. If it is not there, every step gets the driving tempo -- too slow
 * beats a night board that flicks past before it can be read.
 */
function isDriving(gameStep: BaseGameStep): boolean {
  const obs = (gameStep as unknown as { observation?: { phase?: string } }[] | undefined)?.[0]?.observation;
  return obs?.phase ? obs.phase === 'DRIVING' : true;
}

export const getKargoStepRenderTime = (gameStep: BaseGameStep, replayMode: ReplayMode, speedModifier: number): number =>
  defaultGetStepRenderTime(
    gameStep,
    replayMode,
    speedModifier,
    isDriving(gameStep) ? KARGO_STEP_DURATION : NIGHT_STEP_DURATION
  );
