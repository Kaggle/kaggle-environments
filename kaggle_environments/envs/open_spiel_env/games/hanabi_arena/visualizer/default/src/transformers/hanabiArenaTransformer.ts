// Hanabi Arena replay transformer.
//
// Builds the per-step `players` array the side-panel Game Log needs, narrates
// each move, and resolves the head-to-head result. The renderer consumes the
// raw step separately (via `rawStep`) so it can merge all four private views
// into a two-table spectator board.
//
// Three things make this different from the base Hanabi transformer:
//
//   1. Four seats, two teams. `deriveWinnerFromRewards` and
//      `buildForfeitReason` in core are two-player helpers and would name the
//      wrong side here, so the winner and forfeit strings stay local and
//      team-aware -- the same split coin_game_arena makes.
//
//   2. The result is head-to-head, not cooperative. Both tables play the same
//      deal, so the interesting line is "Team 1 wins 16-0", with each table's
//      score and how it ended. A tie is a genuine draw, not a missing winner.
//
//   3. Narrating a move needs the state from *before* it, and only from a view
//      that is not the actor's own (a player cannot see their own cards). Step
//      N's observation is the state after step N's action, so the described
//      card comes from step N-1's merged tables.

import { detectForfeit, FORFEIT_REASONS, parseThoughts, OpenSpielRawPlayer } from '@kaggle-environments/core';

import {
  ArenaObservation,
  HanabiTable,
  actingPlayer,
  arenaFraming,
  cardText,
  colorName,
  mergedTables,
  parseMove,
  teamName,
} from '../observation';

const PLAYERS_PER_TEAM = 2;

interface ArenaPlayer {
  id: number;
  name: string;
  thumbnail: string;
  isTurn: boolean;
  actionDisplayText: string;
  thoughts: string;
  reward: number;
  generateReturns: string[] | null;
  teamId: number;
  seat: number;
  forfeited: boolean;
  forfeitLastAttempt: string | null;
}

export interface HanabiArenaStep {
  step: number;
  players: ArenaPlayer[];
  /** One fully-revealed board per team, indexed by team id. */
  tables: (HanabiTable | null)[];
  framing: ArenaObservation | null;
  isTerminal: boolean;
  /** The head-to-head result line. Null until the episode ends. */
  winner: string | null;
  /** Non-null when the episode ended early because a player forfeited. */
  forfeitReason: string | null;
  /** Plain-language description of the move made on this step, if any. */
  moveSummary: string | null;
  /** Team whose table the move on this step was played at, or null. */
  moveTeamId: number | null;
  teamTotals: number[] | null;
  winningTeam: number | string | null;
  rawStep: OpenSpielRawPlayer[];
}

function teamOf(playerId: number): number {
  return Math.floor(playerId / PLAYERS_PER_TEAM);
}

/** Describe a move using the state of its table before it and after. */
function describeMove(
  actor: number,
  actionString: string | null | undefined,
  before: HanabiTable | null,
  after: HanabiTable | null
): string | null {
  if (!before) return null;
  const seat = before.hands.find((h) => h.player_id === actor)?.player;
  if (seat === undefined) return null;
  const move = parseMove(actionString, seat, before.num_players);
  if (!move) return null;

  if (move.type === 'hint') {
    const hand = before.hands.find((h) => h.player === move.target);
    const what = 'color' in move ? colorName(move.color) : `${move.rank}s`;
    const touched = (hand?.cards ?? []).filter((c) =>
      'color' in move ? c.card?.color === move.color : c.card?.rank === move.rank
    ).length;
    const plural = touched === 1 ? 'card' : 'cards';
    const targetId = hand?.player_id;
    return `hinted P${(targetId ?? move.target) + 1}: ${what} (${touched} ${plural})`;
  }

  const card = before.hands.find((h) => h.player === seat)?.cards[move.slot]?.card;
  const label = card ? cardText(card) : `slot ${move.slot + 1}`;
  if (move.type === 'discard') return `discarded ${label}`;
  // A play either advances a firework or costs a life; both are worth calling out.
  if (after && after.fireworks_total > before.fireworks_total) return `played ${label} — fireworks advance`;
  if (after && after.life_tokens < before.life_tokens) return `misplayed ${label} — lost a life`;
  return `played ${label}`;
}

/** Human-readable label for the side panel's action line. */
function actionLabel(
  actionString: string | null | undefined,
  actor: number,
  table: HanabiTable | null,
  fallback: string
): string {
  const seat = table?.hands.find((h) => h.player_id === actor)?.player;
  if (seat === undefined || !table) return fallback;
  const move = parseMove(actionString, seat, table.num_players);
  if (!move) return fallback;
  if (move.type === 'play') return `Play slot ${move.slot + 1}`;
  if (move.type === 'discard') return `Discard slot ${move.slot + 1}`;
  const what = 'color' in move ? colorName(move.color) : `rank ${move.rank}`;
  const targetId = table.hands.find((h) => h.player === move.target)?.player_id;
  return `Hint P${(targetId ?? move.target) + 1}: ${what}`;
}

/** How a single table ended, for the result line's per-table breakdown. */
function tableNote(table: HanabiTable | null): string | null {
  if (!table) return null;
  if (table.outcome === 'lives_exhausted') return 'bombed out';
  if (table.outcome === 'perfect_score') return 'perfect';
  return null;
}

/**
 * The head-to-head result line, e.g. "Team 1 wins 16-0".
 *
 * Bombing out zeroes a table's score no matter how many fireworks it launched,
 * so the breakdown says so rather than leaving an unexplained 0.
 */
function arenaResult(framing: ArenaObservation | null, tables: (HanabiTable | null)[]): string | null {
  const totals = framing?.team_totals;
  if (!totals || totals.length < 2) return 'Game over';

  const notes = tables.map(tableNote);
  const detail = notes.some((n) => n)
    ? ` (${totals.map((t, i) => (notes[i] ? `${teamName(i)} ${notes[i]}` : `${teamName(i)} ${t}`)).join(', ')})`
    : '';

  if (framing?.winning_team === 'draw' || totals[0] === totals[1]) {
    return `Draw — both teams scored ${totals[0]} / ${framing?.max_score ?? 25}${detail}`;
  }
  const winner = totals[0] > totals[1] ? 0 : 1;
  const loser = 1 - winner;
  return `${teamName(winner)} wins ${totals[winner]}-${totals[loser]}${detail}`;
}

/** Team-aware forfeit line: the offender's whole team loses the match. */
function buildArenaForfeitReason(forfeitSeat: number, reasonKey: string, playerNames: string[]): string {
  const loser = playerNames[forfeitSeat] ?? `Player ${forfeitSeat + 1}`;
  const loserTeam = teamOf(forfeitSeat);
  const winningTeam = 1 - loserTeam;
  const seatA = winningTeam * PLAYERS_PER_TEAM;
  const winnerA = playerNames[seatA] ?? `Player ${seatA + 1}`;
  const winnerB = playerNames[seatA + 1] ?? `Player ${seatA + 2}`;
  const reason = FORFEIT_REASONS[reasonKey] ?? 'forfeited';
  return `${loser} ${reason}. ${teamName(winningTeam)} (${winnerA} & ${winnerB}) wins by default.`;
}

export const hanabiArenaTransformer = (environment: any): HanabiArenaStep[] => {
  const rawSteps: OpenSpielRawPlayer[][] = environment?.steps ?? [];
  const seatCount = rawSteps[0]?.length ?? PLAYERS_PER_TEAM * 2;
  const playerNames: string[] =
    environment?.info?.TeamNames ?? Array.from({ length: seatCount }, (_, i) => `Player ${i + 1}`);
  const out: HanabiArenaStep[] = [];

  rawSteps.forEach((step, index) => {
    const forfeit = detectForfeit(step);
    const framing = arenaFraming(step);
    const tables = mergedTables(step);
    const previous = mergedTables(rawSteps[index - 1]);
    const actor = actingPlayer(step);
    const actorTeam = actor >= 0 ? teamOf(actor) : null;

    const players: ArenaPlayer[] = step.map((p, i): ArenaPlayer => {
      const submission = p.action?.submission;
      const isForfeiter = forfeit?.index === i;
      // A forfeiter submits -1 but is still "acting", so their thoughts and
      // last attempt keep rendering in the side panel.
      const isTurn = (submission !== undefined && submission !== null && submission !== -1) || isForfeiter;
      const fallback = p.action?.actionString ?? '';
      return {
        id: i,
        name: playerNames[i] ?? `Player ${i + 1}`,
        thumbnail: '',
        isTurn,
        // Prefer a decoded label over OpenSpiel's "(Reveal player +1 color G)".
        // A forfeiter's actionString is their last *illegal* attempt, so it is
        // shown verbatim rather than dressed up as a move that was played.
        actionDisplayText:
          isTurn && !isForfeiter
            ? actionLabel(p.action?.actionString, i, tables[teamOf(i)] ?? null, fallback)
            : fallback,
        thoughts: parseThoughts(p.action),
        reward: p.reward ?? 0,
        generateReturns: p.action?.generate_returns ?? null,
        teamId: teamOf(i),
        seat: i % PLAYERS_PER_TEAM,
        forfeited: isForfeiter,
        forfeitLastAttempt: isForfeiter ? (p.action?.actionString ?? null) : null,
      };
    });

    const observationTerminal = !!step[0]?.observation?.isTerminal || !!framing?.is_terminal;
    // A forfeit ends the episode even though OpenSpiel's own state is not
    // terminal; treat it as terminal so the renderer stops showing "Turn: X".
    const isTerminal = observationTerminal || forfeit !== null;

    // On a natural terminal use the arena's own team winner; on a forfeit the
    // other team takes it by default.
    let winningTeam: number | string | null = framing?.winning_team ?? null;
    if (forfeit && (winningTeam === null || winningTeam === undefined)) {
      winningTeam = 1 - teamOf(forfeit.index);
    }

    const moveSummary =
      actor >= 0 && actorTeam !== null
        ? describeMove(actor, step[actor]?.action?.actionString, previous[actorTeam] ?? null, tables[actorTeam] ?? null)
        : null;

    // Setup steps carry no action, but the renderer reads every step through
    // `rawStep`, so they are kept rather than filtered out.
    out.push({
      step: index,
      players,
      tables: [tables[0] ?? null, tables[1] ?? null],
      framing,
      isTerminal,
      winner: isTerminal && !forfeit ? arenaResult(framing, tables) : null,
      forfeitReason: forfeit ? buildArenaForfeitReason(forfeit.index, forfeit.reasonKey, playerNames) : null,
      moveSummary,
      moveTeamId: actorTeam,
      teamTotals: framing?.team_totals ?? null,
      winningTeam,
      rawStep: step,
    });
  });

  return out;
};
