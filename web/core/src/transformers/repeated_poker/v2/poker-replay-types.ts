export interface PokerReplayConfiguration {
  actTimeout: number;
  episodeSteps: number;
  metadata: Record<string, unknown>;
  openSpielGameName: string;
  openSpielGameParameters: OpenSpielGameParameters;
  openSpielGameString: string;
  runTimeout: number;
  seed: number;
  setNumHands: number;
  useImage: boolean;
  useOpenings: boolean;
  validationSetNumHands: number;
}

interface OpenSpielGameParameters {
  blind_schedule: string;
  max_num_hands: number;
  reset_stacks: boolean;
  rotate_dealer: boolean;
  universal_poker_game_string: string;
}

export interface PokerReplayInfoAgent {
  Name: string;
  ThumbnailUrl: string;
}

export interface PokerReplayInfo {
  Agents: PokerReplayInfoAgent[];
  EpisodeId: number;
  LiveVideoPath: string | null;
  TeamNames: string[];
  actionHistory: number[];
  moveDuration: number[];
  stateHistory: string[];
}

interface PokerReplayStepObservation {
  currentPlayer: number;
  isTerminal: boolean;
  legalActionStrings: string[];
  legalActions: number[];
  observationString: string;
  playerId: number;
  remainingOverageTime: number;
  serializedGameAndState: string;
  step?: number;
}

interface PokerReplayStepAction {
  submission: number;
  actionString?: string;
  generate_returns?: string[];
  status?: string;
  thoughts?: string;
}

interface PokerReplayStepInfo {
  actionApplied?: number;
  actionSubmitted?: number;
  actionSubmittedToString?: string;
  agentSelfReportedStatus?: string;
  timeTaken?: number;
}

export interface PokerReplayStep {
  action: PokerReplayStepAction;
  info: PokerReplayStepInfo;
  observation: PokerReplayStepObservation;
  reward: number | null;
  status: string;
}

export interface PokerReplay {
  configuration: PokerReplayConfiguration;
  description: string;
  id: string;
  info: PokerReplayInfo;
  name: string;
  rewards: number[];
  schema_version: number;
  specification: any;
  statuses: string[];
  steps: PokerReplayStep[];
  title: string;
  version: string;
}

export interface PokerReplayUniversalPokerJson {
  acpc_state: string;
  best_five_card_hands: string[];
  best_hand_rank_types: string[];
  betting_history: string;
  blinds: number[];
  board_cards: string;
  current_player: number;
  odds: number[];
  player_contributions: number[];
  player_hands: string[];
  pot_size: number;
  starting_stacks: number[];
}

export interface PokerReplayStepHistory {
  big_blind: number;
  /**
   * A JSON string that can be parsed into a 'UniversalPokerJsonInterface' object.
   */
  current_universal_poker_json: string;
  dealer: number;
  hand_number: number;
  hand_returns: number[][];
  max_num_hands: number;
  /**
   * A JSON string that can be parsed into a 'UniversalPokerJsonInterface' object.
   */
  prev_universal_poker_json: string;
  small_blind: number;
  stacks: number[];
}

export interface PokerReplayStepHistoryParsed {
  big_blind: number;
  current_universal_poker_json: PokerReplayUniversalPokerJson;
  dealer: number;
  hand_number: number;
  hand_returns: number[][];
  max_num_hands: number;
  prev_universal_poker_json: PokerReplayUniversalPokerJson;
  small_blind: number;
  stacks: number[];
  step?: PokerReplayStep;
  stepIndex?: number;
}
