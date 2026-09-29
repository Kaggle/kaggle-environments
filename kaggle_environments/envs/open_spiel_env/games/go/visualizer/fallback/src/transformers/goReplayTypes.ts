import { BaseGamePlayer, BaseGameStep } from '@kaggle-environments/core';

export interface GoBoardState {
  board_size: number;
  komi: number;
  current_player_to_move: string;
  move_number: number;
  previous_move: string | null;
  board: string[][];
}

export interface GoPlayer extends BaseGamePlayer {
  reward: number | null;
  generateReturns: string[] | null;
  /** True when this player forfeited on this step (illegal-move retries exhausted, timeout, or error). */
  forfeited?: boolean;
  /** Raw move string the parser rejected on the final attempt. Not a legal coordinate. */
  forfeitLastAttempt?: string | null;
}

export interface GoStep extends Omit<BaseGameStep, 'players'> {
  players: GoPlayer[];
  boardState: GoBoardState;
  isTerminal: boolean;
  winner: string | null;
  /** Forfeit reason category (TIMEOUT / ERROR / INVALID / TRUNCATED), or null for a natural ending. */
  status: string | null;
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
    call_details?: Array<{ response?: string; finish_reason?: string | null }>;
    /** Why the harness gave up: TRUNCATED / EMPTY / UNPARSABLE / ILLEGAL. */
    failureCategory?: string | null;
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
  status: 'ACTIVE' | 'INACTIVE' | 'DONE' | 'TIMEOUT' | 'ERROR' | 'INVALID';
}
