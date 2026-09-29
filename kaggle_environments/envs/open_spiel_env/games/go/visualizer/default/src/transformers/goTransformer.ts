/* eslint-disable @typescript-eslint/no-explicit-any */
import { detectForfeit } from '@kaggle-environments/core';
import { GoPlayer, GoReplay, GoStep, GoBoardState, GoReplayStep } from './goReplayTypes';

function parseThoughts(action?: { generate_returns?: string[]; thoughts?: string }): string {
  if (action?.generate_returns?.[0]) {
    try {
      const parsed = JSON.parse(action.generate_returns[0]);
      if (parsed.main_response_and_thoughts) {
        return parsed.main_response_and_thoughts;
      }
    } catch {
      return action?.thoughts ?? '';
    }
  }
  return action?.thoughts ?? '';
}

function parseBoardState(observationString: string): GoBoardState {
  try {
    const obs = JSON.parse(observationString);
    const board: string[][] = obs.board_grid.map((row: Record<string, string>[]) =>
      row.map((cell) => Object.values(cell)[0])
    );

    return {
      board_size: obs.board_size,
      komi: obs.komi,
      current_player_to_move: obs.current_player_to_move,
      move_number: obs.move_number,
      previous_move: obs.previous_move ?? obs.previous_move_a1,
      board,
    };
  } catch {
    return {
      board_size: 13,
      komi: 7.5,
      current_player_to_move: '',
      move_number: 0,
      previous_move: null,
      board: [],
    };
  }
}

/**
 * Derive the winner from the final rewards, NOT from `observation.isTerminal`.
 *
 * A forfeit (illegal-move retry exhaustion, timeout, crash) ends the episode
 * while OpenSpiel is still mid-game, so `isTerminal` stays false even though
 * the env has already paid out +1 / -1. Gating on it here used to drop the
 * result entirely, which left the game-over screen to re-score the half-played
 * board and crown whoever happened to lead on territory -- frequently the
 * player who just forfeited.
 */
function deriveWinner(step: GoReplayStep[]): string | null {
  if (step.length < 2) return null;
  if (step[0].reward === 1) return 'black';
  if (step[1].reward === 1) return 'white';
  return null;
}

export const goTransformer = (environment: any): GoStep[] => {
  const goReplay = environment as GoReplay;
  const goSteps: GoStep[] = [];

  const firstStep = goReplay.steps[0];
  const extraStepPlayers = [0, 1].map(
    (index): GoPlayer => ({
      id: index,
      name: environment.info.TeamNames[index],
      thumbnail: '',
      isTurn: false,
      actionDisplayText: '',
      thoughts: '',
      reward: null,
      generateReturns: null,
    })
  );

  goSteps.push({
    step: goSteps.length,
    players: extraStepPlayers,
    boardState: parseBoardState(firstStep[0].observation.observationString),
    isTerminal: false,
    hasCaptures: false,
    winner: null,
    status: null,
  });

  let previousBlackStonesCount = 0;
  let previousWhiteStonesCount = 0;

  for (const step of goReplay.steps) {
    // Which seat, if any, forfeited on this step. Scoping the per-player
    // `forfeited` flag to detectForfeit's answer (rather than testing
    // `submission === -1 && action.status` inline) also filters out Go's
    // setup step, where *both* seats carry a "no legal actions" status --
    // detectForfeit treats a two-sided match as ambiguous and returns null.
    const forfeit = detectForfeit(step);
    // Keep forfeit steps even when no one produced an actionString: an
    // EMPTY / UNPARSABLE / TRUNCATED forfeit has nothing to show for a move,
    // and dropping it would erase the ending from the replay.
    if (!forfeit && step.some((p) => p.action?.actionString) === false) continue;

    const stepPlayers: GoPlayer[] = step.map((player, index): GoPlayer => {
      const forfeited = forfeit?.index === index;
      return {
        id: index,
        name: environment.info.TeamNames[index],
        thumbnail: '',
        // Forfeits count as a turn so the step survives into the timeline;
        // otherwise neither seat is "active" and the step renders blank.
        isTurn: (player.action?.submission !== undefined && player.action.submission !== -1) || forfeited,
        // Left empty on a forfeit: GameRenderer feeds this straight to the
        // board engine, and a forfeit's actionString is the raw text the
        // parser rejected, not a coordinate. Display sites use
        // `forfeitLastAttempt` instead.
        actionDisplayText: forfeited ? '' : (player.action?.actionString?.split(' ').at(1) ?? ''),
        thoughts: parseThoughts(player.action),
        reward: player.reward,
        generateReturns: player.action?.generate_returns ?? null,
        forfeited,
        forfeitLastAttempt: forfeited ? (player.action?.actionString ?? null) : null,
      };
    });

    const boardState = parseBoardState(step[0].observation.observationString);
    const stones = boardState.board.flat();
    const blackStonesCount = stones.filter((s) => s === 'B').length;
    const whiteStonesCount = stones.filter((s) => s === 'W').length;

    goSteps.push({
      step: goSteps.length,
      players: stepPlayers,
      boardState: boardState,
      isTerminal: false,
      hasCaptures: blackStonesCount < previousBlackStonesCount || whiteStonesCount < previousWhiteStonesCount,
      winner: null,
      status: null,
    });

    previousBlackStonesCount = blackStonesCount;
    previousWhiteStonesCount = whiteStonesCount;
  }

  const lastReplayStep = goReplay.steps[goReplay.steps.length - 1];

  goSteps.push({
    step: goSteps.length,
    players: extraStepPlayers,
    boardState: goSteps[goSteps.length - 1].boardState,
    isTerminal: true,
    hasCaptures: false,
    winner: deriveWinner(lastReplayStep),
    // Only the reason category is rendered -- GameOver/getStepLabel derive the
    // loser from the winner, so detectForfeit's index isn't needed here.
    status: detectForfeit(lastReplayStep)?.reasonKey ?? null,
  });

  return goSteps;
};
