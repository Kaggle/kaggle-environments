import { BaseGamePlayer, BaseGameStep } from '../../types';

export interface CABTReplay {
  // Basic game information
  name: string;
  description: string;
  id: string;

  // Game configuration
  configuration: {
    actTimeout: number;
    episodeSteps: number;
    runTimeout: number;
    seed: number;
  };

  // Game information
  info: {
    EpisodeId: number;
    LiveVideoPath: string | null;
    TeamNames: string[];
  };

  // Game results
  rewards: number[];
  statuses: string[];
  schema_version: number;

  // Game specification
  specification: {
    action: {
      default: any[];
      description: string;
      type: string;
    };
    agents: number[];
    configuration: {
      actTimeout: {
        default: number;
        description: string;
        minimum: number;
        type: string;
      };
      episodeSteps: {
        default: number;
        description: string;
        minimum: number;
        type: string;
      };
      runTimeout: {
        default: number;
        description: string;
        minimum: number;
        type: string;
      };
    };
    info: Record<string, any>;
    observation: {
      remainingOverageTime: {
        default: number;
        description: string;
        minimum: number;
        shared: boolean;
        type: string;
      };
      step: {
        default: number;
        description: string;
        minimum: number;
        shared: boolean;
        type: string;
      };
    };
    reward: {
      default: number;
      description: string;
      enum: number[];
      type: (string | null)[];
    };
  };

  // Game steps
  steps: CABTReplayStep[][];
}

export interface CABTReplayStep {
  action: number[] | { submission: number };
  info: Record<string, any>;
  observation: {
    current: CABTGameState | null;
    logs: string[];
    remainingOverageTime: number;
    search_begin_input: any | null;
    select: any | null;
  };
  reward: number;
  status: 'ACTIVE' | 'INACTIVE' | 'DONE';
  visualize?: VisualizeStep[];
}

export interface VisualizeStep {
  current: CABTCurrentState;
  logs: any[];
  select: any;
}

export interface CABTCurrentState {
  firstPlayer: number;
  players: RawCABTPlayer[];
  result: number; // -1 ongoing, 0/1 winner, 2 draw
  stadium: RawCard[];
  turn: number;
  yourIndex: number; // Current player
}

export interface RawCard {
  id: number;
  name: string;
  playerIndex: number;
  serial: number;
  hp?: number;
  energies?: number[];
}

export interface RawCABTPlayer {
  active: RawCard[];
  asleep: boolean;
  bench: RawCard[];
  benchMax: number;
  burned: boolean;
  confused: boolean;
  deckCount: number;
  discard: RawCard[];
  hand: RawCard[];
  handCount: number;
  paralyzed: boolean;
  poisoned: boolean;
  prize: RawCard[];
}

interface CABTGameState {
  firstPlayer: number;
  looking: any | null;
  players: CABTPlayer[];
}

// Interfaces for the transformed data for the visualizer
export interface CABTPlayer extends BaseGamePlayer {
  active: RawCard[];
  bench: RawCard[];
  hand: RawCard[];
  deckCount: number;
  discardCount: number;
  prizeCount: number;
  // status effects
  asleep: boolean;
  burned: boolean;
  confused: boolean;
  paralyzed: boolean;
  poisoned: boolean;
}

export interface CABTStep extends BaseGameStep {
  players: CABTPlayer[];
  stadium: RawCard[];
  result: number;
  isTerminal: boolean;
  winner: string | null;
  turn: number;
}
