import { create } from 'zustand';
import { Chess } from 'chess.js';

interface ChessStore {
  chess: Chess;
  setState: (chess: Chess) => void;
}

const useChessStore = create<ChessStore>((set) => ({
  chess: new Chess(),
  setState: (chess: Chess) => set({ chess }),
}));

export default useChessStore;
