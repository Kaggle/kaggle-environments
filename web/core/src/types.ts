export interface ReplayData {
  name: string;
  version: string;
  steps: BaseGameStep[];
  configuration: Record<string, any>;
  info?: Record<string, any>;
}

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

