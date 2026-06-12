import { BaseGamePlayer, BaseGameStep } from '@kaggle-environments/core';

export type ItemBundle = Record<string, number>;

export interface OfferEvent {
  player: number;
  type: 'offer' | 'agree';
  items?: ItemBundle;
}

export interface BargainingObs {
  current_player: number;
  viewing_player: number;
  is_terminal: boolean;
  agreement_reached: boolean;
  max_turns: number;
  num_offers: number;
  pool: ItemBundle;
  my_values: ItemBundle;
  offer_history: OfferEvent[];
  last_offer: OfferEvent | null;
  returns: number[] | null;
  params: {
    max_turns: number;
    discount: number;
    prob_end: number;
    agree_action: number;
  };
}

export interface BargainingPlayer extends BaseGamePlayer {
  reward: number | null;
}

export interface BargainingStep extends Omit<BaseGameStep, 'players'> {
  step: number;
  players: BargainingPlayer[];
  /** Per-player parsed observation payloads: [player 0, player 1]. */
  observations: [BargainingObs | null, BargainingObs | null];
  /** Either parsed observation -- public fields are identical across players. */
  obs: BargainingObs | null;
  isTerminal: boolean;
}

/**
 * Everything below this point is only used by the transformer to parse
 * the raw replay and should not be used for game display.
 */
export interface BargainingReplayStep {
  action?: {
    actionString?: string;
    call_details?: Array<{ response?: string }>;
    status?: string;
    submission?: number;
    thoughts?: string;
  };
  observation: {
    currentPlayer: number;
    isTerminal: boolean;
    observationString: string;
    playerId: number;
  };
  reward: number | null;
  status: string;
}

export interface BargainingReplay {
  info?: {
    TeamNames?: string[];
  };
  steps: Array<BargainingReplayStep[]>;
}
