import { create } from 'zustand';
import { Game } from 'tenuki';
import { GameRendererProps } from '@kaggle-environments/core';
import { GoStep } from '../transformers/goReplayTypes';

interface GameStore {
  game: Game;
  options: GameRendererProps<GoStep[]>;
  setState: (game: Game, options: GameRendererProps<GoStep[]>) => void;
}

const useGameStore = create<GameStore>((set) => ({
  game: new Game({ boardSize: 13 }),
  options: { replay: { name: '', version: '', steps: [], configuration: {} }, step: 0, agents: [] },
  setState: (game: Game, options: GameRendererProps<GoStep[]>) => set({ game, options }),
}));

export default useGameStore;
