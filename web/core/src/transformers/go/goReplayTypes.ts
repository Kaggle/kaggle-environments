import { BaseGameStep } from '../../types';

export interface GoBoardState {
  board_size: number;
  komi: number;
  current_player_to_move: string;
  move_number: number;
  previous_move_a1: string | null;
  board: string[][];
}

export interface GoStep extends BaseGameStep {
  boardState: GoBoardState;
  isTerminal: boolean;
  winner: string | null;
}

/**
 * Everything below this point is only used in the transformer to parse
 * the replay and should not be used for game display.
 */
export interface GoReplay {
  configuration: {
    actTimeout: number;
    episodeSteps: number;
    metadata: Record<string, any>;
    openSpielGameName: string;
    openSpielGameParameters: {
      board_size: number;
      handicap: number;
      komi: number;
      max_game_length: number;
    };
    openSpielGameString: string;
    runTimeout: number;
    seed: number;
  };
  description: string;
  id: string;
  info: {
    EpisodeId: number;
    LiveVideoPath: string | null;
    TeamNames: string[];
    actionHistory: string[];
    stateHistory?: string[];
  };
  steps: Array<GoReplayStep[]>;
}

/**
 * Only used internally as part of the type for replay data,
 * do not use elsewhere.
 */
export interface GoReplayStep {
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
    step: number;
  };
  reward: number | null;
  status: 'ACTIVE' | 'INACTIVE' | 'DONE';
}
