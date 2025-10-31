export type RepeatedPokerStepType =
  | 'small-blind-post'
  | 'big-blind-post'
  | 'deal-player-hands'
  | 'player-action'
  | 'deal-flop'
  | 'deal-turn'
  | 'deal-river'
  | 'final';

export interface RepeatedPokerStep {
  stepType: RepeatedPokerStepType;
  communityCards: string[];
  pot: number;
  step: number;
  winOdds: number[];
  fiveCardBestHands: string[];
  currentPlayer: number;
  players: RepeatedPokerStepPlayer[];
}

export interface RepeatedPokerStepPlayer {
  id: number;
  name: string;
  thumbnail: string;
  cards: string;
  chipStack: number;
  currentBet: number;
  reward: number | null;
  actionDisplayText: string;
  thoughts?: string;
  isDealer: boolean;
  isTurn: boolean;
  isWinner: boolean;
}
