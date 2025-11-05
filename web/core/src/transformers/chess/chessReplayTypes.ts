import { BaseGamePlayer, BaseGameStep } from '../../types';

export interface ChessPlayer extends BaseGamePlayer {
  reward: number | null;
}

export interface FenState {
  /** 2D array representing the chess board. Each cell contains either a piece character or null for empty squares. */
  board: (string | null)[][];
  /** Active color - 'w' for White, 'b' for Black */
  activeColor: string;
  /** Castling availability - Contains 'K','Q','k','q' characters or '-' if no castling is possible */
  castling: string;
  /** En passant target square in algebraic notation or '-' if not available */
  enPassant: string;
  /** Halfmove clock: number of halfmoves since the last capture or pawn advance */
  halfmoveClock: string;
  /** Fullmove number: starts at 1 and is incremented after Black's move */
  fullmoveNumber: string;
}

export interface ChessStep extends Omit<BaseGameStep, 'players'> {
  players: ChessPlayer[];
  fenState: FenState;
  isTerminal: boolean;
  winner: string | null;
}

/**
 * Everything below this point is only used in the transformer to parse
 * the replay and should not be used for game dispay.
 */
export interface ChessReplay {
  configuration: {
    actTimeout: number;
    episodeSteps: number;
    metadata: Record<string, any>;
    openSpielGameName: string;
    openSpielGameParameters: {
      chess960: boolean;
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
  steps: Array<ChessReplayStep[]>;
}

/**
 * Only used internally as part of the type for replay data,
 * do not use elsewhere.
 */
interface ChessReplayStep {
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
