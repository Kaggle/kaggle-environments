import { BaseGamePlayer, BaseGameStep } from '../../../types';

export type RepeatedPokerStepType =
  | 'small-blind-post'
  | 'big-blind-post'
  | 'deal-player-hands'
  | 'player-action'
  | 'deal-flop'
  | 'deal-turn'
  | 'deal-river'
  | 'final'
  | 'game-over';

export interface RepeatedPokerStep extends BaseGameStep {
  stepType: RepeatedPokerStepType;
  communityCards: string;
  pot: number;
  winOdds: number[];
  bestFiveCardHands: string[];
  bestHandRankTypes: string[];
  currentPlayer: number;
  currentHandIndex: number;
}

export interface RepeatedPokerStepPlayer extends BaseGamePlayer {
  cards: string;
  chipStack: number;
  currentBet: number;
  currentBetForStreet: number;
  reward: number | null;
  isDealer: boolean;
  isWinner: boolean;
}
