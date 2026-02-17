import { ReplayData, RawStep, RawPlayerEntry } from './types';

/**
 * Validates step bounds and returns step data if valid.
 * Returns null if step is out of bounds or data is invalid.
 *
 * This function is designed for raw (untransformed) replay data where each step
 * is an array of player entries with observations.
 *
 * @param replay - The replay data object (typically raw/untransformed)
 * @param step - The step index to validate
 * @returns The step data (array of player entries) or null if invalid
 *
 * @example
 * ```ts
 * const stepData = getStepData(replay, step);
 * if (!stepData) return; // Early exit if invalid
 * const { observation } = stepData[0];
 * ```
 */
export function getStepData<TObservation = Record<string, unknown>>(
  replay: ReplayData<RawStep<TObservation>[]> | ReplayData<unknown> | undefined,
  step: number
): RawStep<TObservation> | null {
  if (!replay?.steps || !Array.isArray(replay.steps) || step < 0 || step >= replay.steps.length) {
    return null;
  }

  const stepData = replay.steps[step];
  if (!stepData || !Array.isArray(stepData) || stepData.length === 0) {
    return null;
  }

  // Validate first element has observation (common pattern for raw replays)
  const firstEntry = stepData[0] as RawPlayerEntry<TObservation>;
  if (!firstEntry || typeof firstEntry !== 'object' || !('observation' in firstEntry)) {
    return null;
  }

  return stepData as RawStep<TObservation>;
}

/**
 * Creates or retrieves a canvas element by ID.
 * Handles canvas creation, sizing, and positioning.
 *
 * @param parent - Parent element to append canvas to
 * @param id - Canvas element ID
 * @param options - Optional width/height overrides
 * @returns Tuple of [canvas, context]
 *
 * @example
 * ```ts
 * const [canvas, ctx] = getCanvas(parent, 'my-canvas', { width: 800, height: 600 });
 * ```
 */
export function getCanvas(
  parent: HTMLElement,
  id: string,
  options: { width?: number; height?: number } = {}
): [HTMLCanvasElement, CanvasRenderingContext2D | null] {
  const width = options.width ?? parent.clientWidth ?? 400;
  const height = options.height ?? parent.clientHeight ?? 400;

  let canvas = document.querySelector<HTMLCanvasElement>('#' + id);
  let isNew = false;

  if (!canvas) {
    canvas = document.createElement('canvas');
    canvas.id = id;
    canvas.style.cssText = `
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
    `;
    parent.appendChild(canvas);
    isNew = true;
  }

  // Only set dimensions on new canvases or when explicitly provided
  // (setting dimensions clears canvas content)
  if (isNew || (options.width && canvas.width !== width) || (options.height && canvas.height !== height)) {
    canvas.width = width;
    canvas.height = height;
  }

  return [canvas, canvas.getContext('2d')];
}
