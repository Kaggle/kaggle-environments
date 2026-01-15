import { BaseGamePlayer, BaseGameStep } from '../../types';

export interface ConnectFourPlayer extends BaseGamePlayer {
  reward: number | null;
}

export interface ConnectFourBoardState {
  /** 2D array representing the connect four board (6 rows x 7 columns). Each cell contains 'x', 'o', or '.' for empty. */
  board: string[][];
  /** Current player - 'x' or 'o' */
  currentPlayer: string;
  /** Whether the game has ended */
  isTerminal: boolean;
  /** Winner - 'x', 'o', or null for no winner yet */
  winner: string | null;
}

export interface ConnectFourStep extends Omit<BaseGameStep, 'players'> {
  players: ConnectFourPlayer[];
  boardState: ConnectFourBoardState;
  isTerminal: boolean;
  winner: string | null;
}

/**
 * Everything below this point is only used in the transformer to parse
 * the replay and should not be used for game display.
 */
export interface ConnectFourReplay {
  configuration: {
    actTimeout: number;
    episodeSteps: number;
    metadata: Record<string, any>;
    openSpielGameName: string;
    openSpielGameParameters: {
      columns: number;
      rows: number;
      x_in_row: number;
    };
    openSpielGameString: string;
    runTimeout: number;
    seed: number;
  };
  description: string;
  id: string;
  info: {
    Agents?: Array<{
      Name: string;
      ThumbnailUrl: string;
    }>;
    EpisodeId: number;
    LiveVideoPath: string | null;
    TeamNames: string[];
    actionHistory: string[];
    stateHistory?: string[];
  };
  steps: Array<ConnectFourReplayStep[]>;
}

/**
 * Only used internally as part of the type for replay data,
 * do not use elsewhere.
 */
export interface ConnectFourReplayStep {
  action?: {
    actionString?: string;
    generate_returns?: string[];
    status?: string;
    submission: number;
    thoughts?: string;
  };
  info?: {
    actionApplied?: number;
    actionSubmitted?: number;
    actionSubmittedToString?: string;
    agentSelfReportedStatus?: string;
    timeTaken?: number;
  };
  observation: {
    currentPlayer: number;
    isTerminal: boolean;
    legalActionStrings: string[];
    legalActions: number[];
    observationString: string;
    playerId: number;
    remainingOverageTime: number;
    serializedGameAndState: string;
    step?: number;
  };
  reward: number | null;
  status: 'ACTIVE' | 'INACTIVE' | 'DONE';
}
