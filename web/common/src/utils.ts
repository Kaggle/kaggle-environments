import { BaseGameStep, BaseGamePlayer } from '@kaggle-environments/core';

/**
 * Returns the player whose turn it is for the given step.
 * Falls back to the first player if no player has isTurn set.
 */
export function getPlayer(step: BaseGameStep): BaseGamePlayer | undefined {
  const players = step.players;
  if (!players || players.length === 0) {
    return undefined;
  }

  // Find the player whose turn it is
  const activePlayer = players.find(p => p.isTurn);
  if (activePlayer) {
    return activePlayer;
  }

  // Fallback to first player if no one has isTurn set
  return players[0];
}
