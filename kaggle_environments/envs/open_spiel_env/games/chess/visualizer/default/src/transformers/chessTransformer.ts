import { ChessAttempt, ChessReplay, ChessPlayer, ChessStep, FenState } from './chessReplayTypes';

function parseFen(fen?: string): FenState {
  if (!fen || typeof fen !== 'string') {
    return {
      board: [],
      activeColor: '',
      castling: '',
      enPassant: '',
      halfmoveClock: '',
      fullmoveNumber: '',
    };
  }

  const [piecePlacement, activeColor, castling, enPassant, halfmoveClock, fullmoveNumber] = fen.split(' ');

  // Within the context of the replay, the active color is the color of the player that just completed their move.
  // This is the opposite of the active color in the fen string, which is the color of the player that is about to move.
  // Therefore, we need to invert the "active" color.
  const playerColor = String(activeColor).toLowerCase() === 'w' ? 'Black' : 'White';

  const board = [];
  const rows = piecePlacement.split('/');

  for (const row of rows) {
    const boardRow = [];
    for (const char of row) {
      if (isNaN(parseInt(char))) {
        boardRow.push(char);
      } else {
        for (let i = 0; i < parseInt(char); i++) {
          boardRow.push(null);
        }
      }
    }
    board.push(boardRow);
  }

  return {
    board,
    activeColor: playerColor,
    castling,
    enPassant,
    halfmoveClock,
    fullmoveNumber,
  };
}

export function getChessStepLabel(step: ChessStep) {
  if (step.isTerminal) {
    return '';
  }

  return step.players.find((player) => player.isTurn)?.actionDisplayText ?? '';
}

export function getChessStepDescription(step: ChessStep) {
  if (step.isTerminal) {
    const winner = step.winner ?? '';
    return step.forfeitReason ? `${winner}\n${step.forfeitReason}` : winner;
  }

  const player = step.players.find((p) => p.isTurn);
  if (!player) return '';
  return renderAttemptsMarkdown(player);
}

/**
 * Render a player's per-attempt LLM calls as markdown. When there's only one
 * attempt this collapses to just the response (the legacy behavior). When
 * there are retries each attempt gets a header showing its outcome:
 *   - intermediate attempts → ❌ Attempt N (illegal — retried)
 *   - final attempt on a successful turn → ✅ Attempt N (submitted)
 *   - all attempts on a forfeit → ❌ Attempt N (illegal — forfeited on last)
 *
 * Falls back to player.thoughts if call_details aren't available (older
 * replays from before the harness wrote call_details).
 */
function renderAttemptsMarkdown(player: ChessPlayer): string {
  const attempts = player.attempts ?? [];
  const fallback = player.thoughts ?? '';

  if (attempts.length === 0) return fallback;

  if (attempts.length === 1 && !player.forfeited) {
    // Single legal attempt — keep the original clean rendering.
    return attempts[0].response || fallback;
  }

  const total = attempts.length;
  const lines: string[] = [];

  if (player.forfeited) {
    const lastMove = player.forfeitLastAttempt ? ` \`${player.forfeitLastAttempt}\`` : '';
    lines.push(`> ⚠️ **Forfeited after ${total} attempt${total === 1 ? '' : 's'}.** Last attempt:${lastMove}`);
    lines.push('');
  } else {
    lines.push(`> 🔁 **Took ${total} attempts** to find a legal move.`);
    lines.push('');
  }

  attempts.forEach((attempt, i) => {
    const isLast = i === attempts.length - 1;
    const ok = isLast && !player.forfeited;
    const tag = ok
      ? `✅ **Attempt ${i + 1} of ${total}** (submitted)`
      : `❌ **Attempt ${i + 1} of ${total}** (illegal — ${isLast ? 'forfeited' : 'retried'})`;
    lines.push(`### ${tag}`);
    lines.push('');
    lines.push(attempt.response || '_(empty response)_');
    lines.push('');
  });

  return lines.join('\n').trim();
}

export function deriveWinnerFromRewards(players: ChessPlayer[]) {
  if (players.length < 2) return '';

  const player0Reward = players[0].reward;
  const player1Reward = players[1].reward;

  if (player0Reward === player1Reward) {
    return 'Draw';
  }

  const winnerPlayerIndex = player0Reward === 1 ? 0 : 1;
  const color = winnerPlayerIndex === 0 ? 'Black' : 'White';

  return `🎉 ${color} (${players[winnerPlayerIndex].name}) Wins!`;
}

/**
 * Statuses set by open_spiel_env when an agent fails to produce a valid action:
 *   TIMEOUT — exceeded the per-move / overage time budget
 *   ERROR   — agent crashed or response was unparsable / cut off
 *   INVALID — submitted an illegal move
 * In all three cases the opponent wins by default.
 *
 * Note: when illegalMoveForfeit:true and the env's INVALID branch runs, both
 * players' top-level status gets overwritten to DONE — the per-player forfeit
 * is only visible via action.submission === -1 + a non-null action.status on
 * the offender. detectForfeitLoser() below handles that case.
 */
const FORFEIT_STATUSES = new Set(['TIMEOUT', 'ERROR', 'INVALID']);

function describeForfeit(status: string): string {
  switch (status) {
    case 'TIMEOUT':
      return 'ran out of time';
    case 'INVALID':
      return 'submitted an illegal move';
    case 'ERROR':
    default:
      return 'failed to produce valid input';
  }
}

/**
 * Find the index of the player who forfeited, or -1 if there's no
 * unambiguous single forfeiter. Returns ``{loserIndex, reason}``.
 *
 * Two signals:
 *   1. Top-level player.status in FORFEIT_STATUSES — used in strict mode
 *      and for ERROR/TIMEOUT cases where the env doesn't overwrite status.
 *   2. action.submission === -1 with a non-null action.status — used for
 *      the illegalMoveForfeit:true path, where the env normalizes both
 *      top-level statuses to DONE but leaves the offender's self-reported
 *      forfeit message on action.status.
 *
 * Returns -1 / null when no detector fires OR when both players match
 * (genuinely undetermined — episode voided).
 */
function detectForfeitLoser(rawLastStep: any[]): { loserIndex: number; reason: string | null } {
  if (rawLastStep.length < 2) return { loserIndex: -1, reason: null };

  const statusForfeits = rawLastStep
    .map((player, index) => (FORFEIT_STATUSES.has(player.status) ? index : -1))
    .filter((index) => index !== -1);
  if (statusForfeits.length === 1) {
    const i = statusForfeits[0];
    return { loserIndex: i, reason: describeForfeit(rawLastStep[i].status) };
  }
  if (statusForfeits.length > 1) {
    return { loserIndex: -1, reason: null };
  }

  const actionForfeits = rawLastStep
    .map((player, index) => (player.action?.submission === -1 && player.action?.status ? index : -1))
    .filter((index) => index !== -1);
  if (actionForfeits.length === 1) {
    // Reuse INVALID phrasing — submission=-1 is the same forfeit-by-illegal-
    // move mechanism, just routed through the env's invalid_action branch
    // (which normalizes top-level status to DONE).
    return { loserIndex: actionForfeits[0], reason: describeForfeit('INVALID') };
  }

  return { loserIndex: -1, reason: null };
}

export const chessTransformer = (environment: any) => {
  const chessReplay = environment as ChessReplay;
  const agents = environment.info.TeamNames;

  const chessSteps: ChessStep[] = [];

  chessReplay.steps.forEach((step, index) => {
    // Each step contains a tuple of players, one who acted and one who's waiting
    const stepPlayers: ChessPlayer[] = step.map((player, playerIndex): ChessPlayer => {
      const attempts: ChessAttempt[] = player.action?.call_details?.map((c) => ({ response: c.response ?? '' })) ?? [];
      // A forfeit step is one where the player submitted -1 *and* the harness
      // wrote a self-reported status (action.status). Inactive turns also
      // have submission === -1 but with null action.status.
      const forfeited = player.action?.submission === -1 && !!player.action?.status;
      return {
        id: playerIndex,
        name: agents[playerIndex],
        thumbnail: '',
        // Treat forfeits as a "turn" too so the reasoning panel surfaces the
        // attempts — the player did act, they just failed every attempt.
        isTurn: player.action?.submission !== -1 || forfeited,
        actionDisplayText: forfeited
          ? `(forfeited: ${player.action?.actionString ?? 'no move'})`
          : (player.action?.actionString ?? ''),
        thoughts: player.action?.thoughts ?? '',
        reward: player.reward,
        attempts,
        forfeited,
        forfeitLastAttempt: forfeited ? (player.action?.actionString ?? null) : null,
      };
    });

    // Ignore setup steps where no one acted
    if (stepPlayers.findIndex((player) => player.isTurn) !== -1) {
      chessSteps.push({
        step: index,
        players: stepPlayers,
        // Both agents have the same observation string for the step, just grab the first one
        fenState: parseFen(step[0].observation.observationString),
        isTerminal: false,
        winner: '',
      });
    }
  });

  const lastStep = chessSteps[chessSteps.length - 1];

  // The raw terminal step is the only place rewards are populated — earlier
  // steps have reward: null. The chessSteps filter above drops any step
  // where neither player has isTurn (submission !== -1), which means the
  // terminal step (both submitted -1) gets dropped when the game ended by
  // forfeit. Always pull rewards and statuses from the raw last step.
  const rawLastStep = chessReplay.steps[chessReplay.steps.length - 1] ?? [];
  const terminalPlayers: ChessPlayer[] =
    rawLastStep.length >= 2
      ? lastStep.players.map((p, i) => ({ ...p, reward: rawLastStep[i]?.reward ?? null }))
      : lastStep.players;

  // If the loser exceeded their time budget / errored / submitted an illegal
  // move, declare the opponent the winner. Otherwise fall back to the
  // rewards-based detection (normal checkmate/stalemate paths).
  let winDescription: string;
  let forfeitReason: string | null = null;
  const { loserIndex, reason } = detectForfeitLoser(rawLastStep);

  if (loserIndex !== -1) {
    const winnerIndex = 1 - loserIndex;
    const loserName = agents[loserIndex] || `Player ${loserIndex + 1}`;
    const winnerName = agents[winnerIndex] || `Player ${winnerIndex + 1}`;
    const winnerColor = winnerIndex === 0 ? 'Black' : 'White';
    forfeitReason = `${loserName} ${reason}. ${winnerName} wins by default.`;
    winDescription = `🎉 ${winnerColor} (${winnerName}) Wins!`;
  } else {
    winDescription = deriveWinnerFromRewards(terminalPlayers);
    const multiStatusForfeit = rawLastStep.filter((p) => FORFEIT_STATUSES.has(p.status)).length > 1;
    if (multiStatusForfeit) {
      // Both players forfeited (e.g. non-strict mode where any agent error
      // marks everyone ERROR). The episode is voided rather than a draw.
      forfeitReason = 'Both players failed to produce valid input; episode voided.';
    }
  }

  // Artificially insert a step at the end to emphasize the win state
  chessSteps.push({
    players: [
      {
        id: -1,
        name: 'System',
        thumbnail: '',
        isTurn: false,
        actionDisplayText: '',
        thoughts: '',
        reward: 0,
      },
    ],
    isTerminal: true,
    fenState: lastStep.fenState,
    step: lastStep.step + 1,
    winner: winDescription,
    forfeitReason,
  });

  return chessSteps;
};
