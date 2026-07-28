import { detectForfeit, FORFEIT_REASONS, FORFEIT_STATUSES } from '@kaggle-environments/core';
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
    const outcome = isLast ? 'forfeited' : 'retried';
    const cause = attempt.finishReason === 'length' ? 'cut off at token limit' : 'illegal';
    const tag = ok
      ? `✅ **Attempt ${i + 1} of ${total}** (submitted)`
      : `❌ **Attempt ${i + 1} of ${total}** (${cause} — ${outcome})`;
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

export const chessTransformer = (environment: any) => {
  const chessReplay = environment as ChessReplay;
  const agents = environment.info.TeamNames;

  const chessSteps: ChessStep[] = [];

  chessReplay.steps.forEach((step, index) => {
    // Each step contains a tuple of players, one who acted and one who's waiting
    const stepPlayers: ChessPlayer[] = step.map((player, playerIndex): ChessPlayer => {
      const attempts: ChessAttempt[] =
        player.action?.call_details?.map((c) => ({
          response: c.response ?? '',
          finishReason: c.finish_reason ?? null,
        })) ?? [];
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
  const forfeit = detectForfeit(rawLastStep);

  if (forfeit) {
    const loserIndex = forfeit.index;
    const winnerIndex = 1 - loserIndex;
    const loserName = agents[loserIndex] || `Player ${loserIndex + 1}`;
    const winnerName = agents[winnerIndex] || `Player ${winnerIndex + 1}`;
    const winnerColor = winnerIndex === 0 ? 'Black' : 'White';
    forfeitReason = `${loserName} ${FORFEIT_REASONS[forfeit.reasonKey]}. ${winnerName} wins by default.`;
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
