import { create } from 'zustand';
import { Chess } from 'chess.js';
import { ChessStep, GameRendererProps } from '@kaggle-environments/core';

interface ChessStore {
  chess: Chess;
  options: GameRendererProps<ChessStep[]> | null;
  setState: (chess: Chess, options: GameRendererProps<ChessStep[]>) => void;
}

const useChessStore = create<ChessStore>((set) => ({
  chess: new Chess(),
  options: null,
  setState: (chess: Chess, options: GameRendererProps<ChessStep[]>) => set({ chess, options }),
}));

export default useChessStore;
