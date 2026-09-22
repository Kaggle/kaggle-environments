import { detectForfeit, FORFEIT_REASONS } from '@kaggle-environments/core';
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
      board_size: 9,
      komi: 7.5,
      current_player_to_move: '',
      move_number: 0,
      previous_move: null,
      board: [],
    };
  }
}

/**
 * Derive the result from the final rewards rather than `observation.isTerminal`.
 *
 * A forfeit ends the episode while OpenSpiel is still mid-game, so `isTerminal`
 * stays false even though the env has already paid out +1 / -1. Gating on it
 * dropped the result from forfeited replays entirely.
 */
function deriveWinner(step: GoReplayStep[], forfeitReasonKey?: string | null): string {
  if (step.length < 2) return '';

  const reward0 = step[0].reward;
  const reward1 = step[1].reward;

  let result: string;
  if (reward0 === 1) {
    result = 'Black Wins!';
  } else if (reward1 === 1) {
    result = 'White Wins!';
  } else {
    return 'Draw';
  }

  const reason = forfeitReasonKey ? FORFEIT_REASONS[forfeitReasonKey] : undefined;
  return reason ? `${result} (opponent ${reason})` : result;
}

export const goTransformer = (environment: any): GoStep[] => {
  const goReplay = environment as GoReplay;
  const agents = environment.info.TeamNames;

  const goSteps: GoStep[] = [];

  goReplay.steps.forEach((step, index) => {
    // Scoping the per-player flag to detectForfeit also filters out Go's setup
    // step, where *both* seats carry a "no legal actions" status -- a two-sided
    // match is ambiguous, so detectForfeit returns null there.
    const forfeit = detectForfeit(step);

    const stepPlayers: GoPlayer[] = step.map((player, playerIndex): GoPlayer => {
      const forfeited = forfeit?.index === playerIndex;
      const actionString = player.action?.actionString ?? '';
      const [, move] = actionString.split(' ');

      return {
        id: playerIndex,
        name: agents[playerIndex],
        thumbnail: '',
        // Forfeits count as a turn so the step survives into the replay;
        // otherwise neither seat is active and the ending vanishes.
        isTurn: (player.action?.submission !== undefined && player.action.submission !== -1) || forfeited,
        actionDisplayText: forfeited ? '' : (move ?? ''),
        thoughts: parseThoughts(player.action),
        reward: player.reward,
        generateReturns: player.action?.generate_returns ?? null,
        forfeited,
        forfeitLastAttempt: forfeited ? (player.action?.actionString ?? null) : null,
      };
    });

    if (stepPlayers.some((player) => player.isTurn)) {
      // A forfeit ends the episode even though OpenSpiel never reached a
      // terminal state, so treat it as terminal for display purposes.
      const isTerminal = step[0].observation.isTerminal || !!forfeit;
      goSteps.push({
        step: index,
        players: stepPlayers,
        boardState: parseBoardState(step[0].observation.observationString),
        isTerminal,
        winner: isTerminal ? deriveWinner(step, forfeit?.reasonKey) : null,
        status: forfeit?.reasonKey ?? null,
      });
    }
  });

  return goSteps;
};
