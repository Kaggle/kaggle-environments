export interface ReplayStep {
  observation: Record<string, any>;
  action: Record<string, any> | null;
  reward: Record<string, number> | number | null;
  info: Record<string, any>;
  status: string;
}

export interface ReplayData {
  name: string;
  version: string;
  steps: ReplayStep[][];
  configuration: Record<string, any>;
  info?: Record<string, any>;
}

export interface Player {
  name: string;
  thumbnailUrl: string;
}


export interface BaseGameStep {
  isEndState: boolean;
  step: any;
  stateHistory: any;
}

export interface PokerGameStep extends BaseGameStep {
  hand: number;
  handConclusion?: "fold" | "showdown";
  winner?: -1 | 0 | 1; // -1 for the rare event of a tie
  bestFiveCardHands?: string[]; // e.g. ['AsJhTh2h2c', 'As9s9h2h2c'] (cards to be highlighted)
  bestHandRankType?: string[]; // e.g. ['High Card', 'Two Pair'] (human-readable string)
}

export type GameStep = PokerGameStep | BaseGameStep

