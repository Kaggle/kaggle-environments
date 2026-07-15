// Nine Men's Morris replay transformer.
//
// Builds the per-step `players` array the side-panel UI needs to display
// model thoughts / action labels, and pre-parses the observation JSON into
// a `boardState` the renderer can consume directly.
//
// Nine Men's Morris is sequential: only the active player produces a real
// action in a step; the inactive player submits -1. The env also emits two
// setup steps at the start with both submissions == -1, which we drop.
//
// Forfeits (illegal-move / TIMEOUT / ERROR) end the game without OpenSpiel
// reaching a terminal state. We detect them here so the renderer can label
// the ending explicitly instead of falling back to a stale "Turn: X" line.

// Reasons the env sets on a player's top-level status when they fail to
// produce a legal action. The illegal-move path additionally routes through
// action.submission === -1 with a non-null action.status, because in that
// branch open_spiel_env normalizes both top-level statuses to DONE.
const FORFEIT_STATUSES = new Set(['TIMEOUT', 'ERROR', 'INVALID']);

const FORFEIT_REASONS: Record<string, string> = {
  TIMEOUT: 'ran out of time',
  INVALID: 'submitted an illegal move',
  ERROR: 'failed to produce valid input',
};

interface NineMensMorrisAction {
  submission?: number;
  actionString?: string | null;
  thoughts?: string | null;
  status?: string | null;
  generate_returns?: string[] | null;
}

interface NineMensMorrisObservation {
  observationString?: string;
  isTerminal?: boolean;
}

interface NineMensMorrisReplayPlayer {
  action?: NineMensMorrisAction;
  reward: number;
  observation: NineMensMorrisObservation;
  status?: string;
}

interface NineMensMorrisPlayer {
  id: number;
  name: string;
  thumbnail: string;
  isTurn: boolean;
  actionDisplayText: string;
  thoughts: string;
  reward: number;
  generateReturns: string[] | null;
  forfeited: boolean;
  forfeitLastAttempt: string | null;
}

export interface NineMensMorrisBoardState {
  board: string[]; // length 24, each 'W' | 'B' | '.'
  current_player: string;
  phase: string;
  men_to_deploy: Record<string, number>;
  num_men: Record<string, number>;
  turn_number: number;
  is_terminal: boolean;
  winner: string | null;
  last_action: string | null;
}

export interface NineMensMorrisStep {
  step: number;
  players: NineMensMorrisPlayer[];
  boardState: NineMensMorrisBoardState | null;
  isTerminal: boolean;
  winner: string | null;
  // Non-null on the terminal step when the game ended because a player
  // forfeited (illegal-move retries exhausted, timeout, or crash). The
  // renderer surfaces this instead of the normal "X wins!" line so the
  // reason for the early end is clear.
  forfeitReason: string | null;
}

// action.thoughts is the harness-curated summary and the preferred source.
// generate_returns[0].main_response_and_thoughts is the raw LLM output;
// use it only when the harness didn't populate thoughts.
function parseThoughts(action?: NineMensMorrisAction): string {
  if (action?.thoughts) return action.thoughts;
  if (action?.generate_returns?.[0]) {
    try {
      const parsed = JSON.parse(action.generate_returns[0]);
      if (parsed.main_response_and_thoughts) {
        return parsed.main_response_and_thoughts;
      }
    } catch {
      // fall through
    }
  }
  return '';
}

function parseBoardState(step: NineMensMorrisReplayPlayer[]): NineMensMorrisBoardState | null {
  const raw = step?.[0]?.observation?.observationString ?? step?.[1]?.observation?.observationString;
  if (!raw) return null;
  try {
    return JSON.parse(raw) as NineMensMorrisBoardState;
  } catch {
    return null;
  }
}

function deriveWinner(step: NineMensMorrisReplayPlayer[], teamNames: string[]): string | null {
  if (step.length < 2) return null;
  const r0 = step[0].reward ?? 0;
  const r1 = step[1].reward ?? 0;
  if (r0 === r1) return 'Draw';
  return r0 > r1 ? `${teamNames[0]} wins!` : `${teamNames[1]} wins!`;
}

// Detect if this raw step is a forfeit by a single player. Returns the index
// of the forfeiter and the reason category (TIMEOUT / ERROR / INVALID), or
// null if not a forfeit / ambiguous.
function detectForfeit(step: NineMensMorrisReplayPlayer[]): { index: number; reasonKey: string } | null {
  if (step.length < 2) return null;

  const statusForfeiters = step.map((p, i) => ({ p, i })).filter(({ p }) => p.status && FORFEIT_STATUSES.has(p.status));
  if (statusForfeiters.length === 1) {
    return { index: statusForfeiters[0].i, reasonKey: statusForfeiters[0].p.status! };
  }
  if (statusForfeiters.length > 1) return null;

  // illegalMoveForfeit path: env sets both top-level statuses to DONE, but
  // the offender's action carries submission=-1 with a self-reported status.
  const actionForfeiters = step
    .map((p, i) => ({ p, i }))
    .filter(({ p }) => p.action?.submission === -1 && !!p.action?.status);
  if (actionForfeiters.length === 1) {
    return { index: actionForfeiters[0].i, reasonKey: 'INVALID' };
  }
  return null;
}

export const nineMensMorrisTransformer = (environment: any): NineMensMorrisStep[] => {
  const teamNames: string[] = environment?.info?.TeamNames ?? ['Player W', 'Player B'];
  const rawSteps: NineMensMorrisReplayPlayer[][] = environment?.steps ?? [];
  const out: NineMensMorrisStep[] = [];

  rawSteps.forEach((step, index) => {
    const forfeit = detectForfeit(step);

    const players: NineMensMorrisPlayer[] = step.map((p, i): NineMensMorrisPlayer => {
      const submission = p.action?.submission;
      const isForfeiter = forfeit?.index === i;
      // A forfeit step's offender has submission === -1 but should still be
      // treated as "acting" so the step is retained and their thoughts /
      // last-attempt render in the side panel.
      const isTurn = (submission !== undefined && submission !== -1) || isForfeiter;
      return {
        id: i,
        name: teamNames[i] ?? (i === 0 ? 'Player W' : 'Player B'),
        thumbnail: '',
        isTurn,
        actionDisplayText: p.action?.actionString ?? '',
        thoughts: parseThoughts(p.action),
        reward: p.reward ?? 0,
        generateReturns: p.action?.generate_returns ?? null,
        forfeited: isForfeiter,
        forfeitLastAttempt: isForfeiter ? (p.action?.actionString ?? null) : null,
      };
    });

    // Skip the env's setup steps where both players submit -1 and neither
    // forfeited.
    if (!players.some((pl) => pl.isTurn)) return;

    const observationTerminal = !!step[0]?.observation?.isTerminal;
    // A forfeit ends the episode even though OpenSpiel's own state isn't
    // terminal; treat it as terminal so downstream UI shows the end state.
    const isTerminal = observationTerminal || forfeit !== null;

    let forfeitReason: string | null = null;
    if (forfeit) {
      const loser = teamNames[forfeit.index] ?? `Player ${forfeit.index + 1}`;
      const winner = teamNames[1 - forfeit.index] ?? `Player ${2 - forfeit.index}`;
      const reason = FORFEIT_REASONS[forfeit.reasonKey] ?? 'forfeited';
      forfeitReason = `${loser} ${reason}. ${winner} wins by default.`;
    }

    out.push({
      step: index,
      players,
      boardState: parseBoardState(step),
      isTerminal,
      winner: isTerminal ? deriveWinner(step, teamNames) : null,
      forfeitReason,
    });
  });

  return out;
};
