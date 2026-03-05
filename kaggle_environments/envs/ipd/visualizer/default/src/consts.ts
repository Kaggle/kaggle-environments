export const ACTION_NAMES = ['COOPERATE', 'DEFECT'] as const;

/**
 * Payoff matrix indexed by [player action][opponent action].
 * CC=5, DD=1, DC=10, CD=0
 */
export const PAYOFF_MATRIX: Record<number, Record<number, number>> = {
  0: { 0: 5, 1: 0 }, // COOPERATE vs COOPERATE=5, COOPERATE vs DEFECT=0
  1: { 0: 10, 1: 1 }, // DEFECT vs COOPERATE=10, DEFECT vs DEFECT=1
};
