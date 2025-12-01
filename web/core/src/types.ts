export interface ReplayData<TSteps = BaseGameStep[]> {
  name: string;
  version: string;
  steps: TSteps;
  configuration: Record<string, any>;
  info?: Record<string, any>;
  isTransformed?: boolean;
}

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
