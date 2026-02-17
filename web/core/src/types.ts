export interface ReplayData<TSteps = BaseGameStep[]> {
  name: string;
  version: string;
  steps: TSteps;
  configuration: Record<string, any>;
  info?: Record<string, any>;
  isTransformed?: boolean;
}

/**
 * Raw player state entry from a Kaggle environment replay.
 * Each step in a raw replay is an array of these entries (one per player).
 * The first player typically has the full observation data.
 */
export interface RawPlayerEntry<TObservation = Record<string, unknown>> {
  action?: string | number | Record<string, unknown>;
  info?: Record<string, unknown>;
  observation: TObservation;
  reward: number | null;
  status: 'ACTIVE' | 'INACTIVE' | 'DONE' | 'ERROR' | 'TIMEOUT' | 'INVALID';
}

/**
 * A single step in a raw (untransformed) replay.
 * Each step is an array of player entries.
 */
export type RawStep<TObservation = Record<string, unknown>> = RawPlayerEntry<TObservation>[];

/**
 * Raw replay data before transformation.
 * Use this type when working with replay data directly from the Kaggle API.
 */
export type RawReplayData<TObservation = Record<string, unknown>> = ReplayData<RawStep<TObservation>[]>;

/**
 * only-stream: used for recording videos that show a play-by-play with agent reasoning
 * zen: a similar streaming view but for replays with interactive controls
 * logs: a condensed view of the episode without reasoning
 */
export type ReplayMode = 'only-stream' | 'zen' | 'condensed';

export interface BaseGameStep {
  step: number;
  players: BaseGamePlayer[];
}

export interface BaseGamePlayer {
  id: number;
  name: string;
  thumbnail: string;
  isTurn: boolean;
  actionDisplayText?: string;
  thoughts?: string;
}

export interface InterestingEvent {
  step: number;
  description: string;
}

export interface EpisodeSlice {
  id: number;
  start: number;
  title: string;
  urlParamKey: string;
}
