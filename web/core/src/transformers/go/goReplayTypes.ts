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
interface GoReplayStep {
  action: {
    actionString: string;
    generate_returns: string[];
    status: string;
    submission: number;
    thoughts: string;
  };
  info: {
    actionApplied: number;
    actionSubmitted: number;
    actionSubmittedToString: string;
    agentSelfReportedStatus: string;
    timeTaken: number;
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
  status: string;
}
