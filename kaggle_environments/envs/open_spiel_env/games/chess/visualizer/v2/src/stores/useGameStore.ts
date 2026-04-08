import { create } from 'zustand';
import { Chess } from 'chess.js';
import { ChessStep, GameRendererProps } from '@kaggle-environments/core';

interface GameStore {
  game: Chess;
  options: GameRendererProps<ChessStep[]> | null;
  setState: (chess: Chess, options: GameRendererProps<ChessStep[]>) => void;
}

const useGameStore = create<GameStore>((set) => ({
  game: new Chess(),
  options: null,
  setState: (game: Chess, options: GameRendererProps<ChessStep[]>) => set({ game, options }),
}));

export default useGameStore;
