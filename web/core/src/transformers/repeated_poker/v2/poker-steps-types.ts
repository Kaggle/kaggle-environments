import { BaseGamePlayer, BaseGameStep } from "../../../types";

export type RepeatedPokerStepType =
  | 'small-blind-post'
  | 'big-blind-post'
  | 'deal-player-hands'
  | 'player-action'
  | 'deal-flop'
  | 'deal-turn'
  | 'deal-river'
  | 'final';

export interface RepeatedPokerStep extends BaseGameStep {
  stepType: RepeatedPokerStepType;
  communityCards: string;
  pot: number;
  winOdds: number[];
  fiveCardBestHands: string[];
  currentPlayer: number;
}

export interface RepeatedPokerStepPlayer extends BaseGamePlayer {
  cards: string;
  chipStack: number;
  currentBet: number;
  reward: number | null;
  isDealer: boolean;
  isWinner: boolean;
}
