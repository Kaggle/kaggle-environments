import { create } from 'zustand';
import { Chess } from 'chess.js';
import { GameRendererProps } from '@kaggle-environments/core';
import { ChessStep } from '../transformers/chessReplayTypes';

interface GameStore {
  game: Chess;
  options: GameRendererProps<ChessStep[]>;
  setState: (chess: Chess, options: GameRendererProps<ChessStep[]>) => void;
}

const useGameStore = create<GameStore>((set) => ({
  game: new Chess(),
  options: { replay: { name: '', version: '', steps: [], configuration: {} }, step: 0, agents: [] },
  setState: (game: Chess, options: GameRendererProps<ChessStep[]>) => set({ game, options }),
}));

export default useGameStore;
