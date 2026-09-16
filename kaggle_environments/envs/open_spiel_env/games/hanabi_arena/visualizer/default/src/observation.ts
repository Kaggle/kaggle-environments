// Shared parsing for the Hanabi Arena game's JSON observations.
//
// The arena runs two independent 2-player Hanabi tables off one shuffled deck
// and ranks the teams by final score. Each of the four seats emits an
// observation that frames the arena (`your_team_id`, `active_player_id`, ...)
// and carries exactly one `table` -- its own. The opposing table is hidden
// mid-game, so a spectator board has to be assembled from all four.
//
// Two merges are needed, for two different reasons:
//
//   1. *Within* a table, Hanabi hides a player's own hand from them. Seat 0's
//      view shows seat 1's cards face-up and its own as knowledge only, and
//      vice versa. Merging the two seats of a team recovers the real hand for
//      both. (Hint knowledge is public and identical in both views, so it can
//      be read from either.)
//   2. *Across* tables, each team's board only appears in its own seats'
//      observations. Collecting one merged table per team gives the two-column
//      spectator view. At terminal the engine also publishes a `tables` array
//      with both boards fully revealed; that is preferred when present.
//
// Both the renderer and the transformer need all of this, so it lives here.

import { OpenSpielRawPlayer } from '@kaggle-environments/core';

export interface HanabiCardFace {
  color: string;
  rank: number;
}

export interface HanabiCard {
  /** The real card, or null when the reading observer is its holder. */
  card: HanabiCardFace | null;
  /** What the holder has been told out loud. */
  hinted_color: string | null;
  hinted_rank: number | null;
  /** What the holder can still deduce, including negative inference. */
  plausible_colors: string[];
  plausible_ranks: number[];
}

export interface HanabiHand {
  /** Seat within the table (0 or 1). */
  player: number;
  /** External arena player id (0-3) holding this seat. */
  player_id: number;
  is_observer: boolean;
  cards: HanabiCard[];
}

/** One team's Hanabi board. Shaped like the base game's `state_dict`. */
export interface HanabiTable {
  team_id: number;
  num_players: number;
  colors: number;
  ranks: number;
  hand_size: number;
  /** Seat whose hand is hidden in this view, or null once fully revealed. */
  observer: number | null;
  current_player: number;
  current_player_id: number | null;
  life_tokens: number;
  max_life_tokens: number;
  info_tokens: number;
  max_info_tokens: number;
  fireworks: Record<string, number>;
  /** Banked score -- zero once the last life is lost, matching the engine. */
  score: number;
  /** Stack heights regardless of lives; equals `score` until a bomb-out. */
  fireworks_total: number;
  max_score: number;
  hands: HanabiHand[];
  deck_size: number;
  deck_total: number;
  final_turns_remaining: number | null;
  discards: HanabiCardFace[];
  is_terminal: boolean;
  outcome: string;
  legal_actions: { action: number; label: string }[];
  move_history: HanabiMoveRecord[];
  move_number: number;
}

export interface HanabiMoveRecord {
  seat: number;
  player_id: number;
  action: number;
  label: string;
}

/** The arena framing around a single seat's table. */
export interface ArenaObservation {
  phase: string;
  move_number: number;
  active_player_id: number | null;
  active_team_id: number | null;
  active_seat: number | null;
  num_teams: number;
  players_per_team: number;
  max_score: number;
  is_terminal: boolean;
  your_player_id: number;
  your_team_id: number;
  your_seat: number;
  teammate_player_id: number;
  your_turn: boolean;
  your_table_finished: boolean;
  table: HanabiTable;
  /** Terminal only: both boards, fully revealed. */
  tables?: HanabiTable[];
  returns?: number[];
  team_totals?: number[];
  winning_team?: number | 'draw';
}

// OpenSpiel's color letters, in firework-display order.
export const COLOR_ORDER = ['R', 'Y', 'G', 'W', 'B'];

export const COLOR_NAMES: Record<string, string> = {
  R: 'red',
  Y: 'yellow',
  G: 'green',
  W: 'white',
  B: 'blue',
};

// Ink colors for each suit. White is drawn as a mid gray so it stays legible
// against the white card stock.
export const COLOR_INK: Record<string, string> = {
  R: '#c8384f',
  Y: '#c99700',
  G: '#3f8f4f',
  W: '#8b8b8b',
  B: '#2b6cb0',
};

export const COLOR_TINT: Record<string, string> = {
  R: '#fbe3e7',
  Y: '#fbf1cf',
  G: '#e0f2e2',
  W: '#efefef',
  B: '#dfeaf8',
};

// One color per team, matching the other arena visualizers.
export const TEAM_COLORS = ['#1f77b4', '#d62728'];

export function colorName(letter: string): string {
  return COLOR_NAMES[letter] ?? letter;
}

export function teamName(teamId: number): string {
  return `Team ${teamId + 1}`;
}

export function parseObservation(step: OpenSpielRawPlayer[] | undefined, playerIdx: number): ArenaObservation | null {
  const raw = step?.[playerIdx]?.observation?.observationString;
  if (!raw) return null;
  try {
    return JSON.parse(raw) as ArenaObservation;
  } catch {
    return null;
  }
}

/** Every seat's observation for this step, in player-id order, nulls dropped. */
export function parseAllObservations(step: OpenSpielRawPlayer[] | undefined): ArenaObservation[] {
  if (!Array.isArray(step)) return [];
  const views: ArenaObservation[] = [];
  for (let i = 0; i < step.length; i++) {
    const obs = parseObservation(step, i);
    if (obs) views.push(obs);
  }
  return views;
}

/** Fill in the hidden hand of each view's own observer from a teammate's view. */
function mergeTableViews(views: HanabiTable[]): HanabiTable {
  const merged: HanabiTable = JSON.parse(JSON.stringify(views[0]));
  for (const hand of merged.hands) {
    if (!hand.is_observer) continue;
    // These cards are hidden from views[0]'s observer; find a view that sees them.
    const revealing = views.find((v) => v.observer !== hand.player);
    const revealed = revealing?.hands.find((h) => h.player === hand.player);
    if (!revealed) continue;
    hand.cards.forEach((card, i) => {
      card.card = revealed.cards[i]?.card ?? null;
    });
  }
  merged.observer = null;
  return merged;
}

/**
 * A spectator view: one fully-revealed board per team, indexed by team id.
 *
 * Returns an empty array before the deal, when no observation has been emitted.
 */
export function mergedTables(step: OpenSpielRawPlayer[] | undefined): HanabiTable[] {
  const views = parseAllObservations(step);
  if (!views.length) return [];

  // At terminal the engine publishes both boards outright; no merge needed.
  const revealed = views.find((v) => Array.isArray(v.tables) && v.tables.length);
  if (revealed?.tables) {
    const tables = [...revealed.tables].sort((a, b) => a.team_id - b.team_id);
    return tables.map((t) => ({ ...t, observer: null }));
  }

  const byTeam = new Map<number, HanabiTable[]>();
  for (const view of views) {
    if (!view.table) continue;
    const bucket = byTeam.get(view.table.team_id);
    if (bucket) bucket.push(view.table);
    else byTeam.set(view.table.team_id, [view.table]);
  }
  return [...byTeam.entries()].sort(([a], [b]) => a - b).map(([, group]) => mergeTableViews(group));
}

/** The arena framing, which is identical in every seat's view bar the "your_*" fields. */
export function arenaFraming(step: OpenSpielRawPlayer[] | undefined): ArenaObservation | null {
  const views = parseAllObservations(step);
  return views[0] ?? null;
}

export type HanabiMove =
  | { type: 'play'; slot: number }
  | { type: 'discard'; slot: number }
  | { type: 'hint'; target: number; color: string }
  | { type: 'hint'; target: number; rank: number };

const PLAY_RE = /^\(Play (\d+)\)$/;
const DISCARD_RE = /^\(Discard (\d+)\)$/;
const HINT_COLOR_RE = /^\(Reveal player \+(\d+) color (\S+)\)$/;
const HINT_RANK_RE = /^\(Reveal player \+(\d+) rank (\d+)\)$/;

/**
 * Decode OpenSpiel's action string into a structured move.
 *
 * Hint actions name their target *relative* to the actor's seat at its own
 * table ("player +1"), not by arena player id -- so `actor` here is a seat and
 * `numPlayers` the seats per table, never the four arena players.
 */
export function parseMove(
  actionString: string | null | undefined,
  actor: number,
  numPlayers: number
): HanabiMove | null {
  if (!actionString) return null;
  let m = PLAY_RE.exec(actionString);
  if (m) return { type: 'play', slot: Number(m[1]) };
  m = DISCARD_RE.exec(actionString);
  if (m) return { type: 'discard', slot: Number(m[1]) };
  m = HINT_COLOR_RE.exec(actionString);
  if (m) return { type: 'hint', target: (actor + Number(m[1])) % numPlayers, color: m[2] };
  m = HINT_RANK_RE.exec(actionString);
  if (m) return { type: 'hint', target: (actor + Number(m[1])) % numPlayers, rank: Number(m[2]) };
  return null;
}

/** Arena player id who acted on this step, or -1 for setup / forfeit steps. */
export function actingPlayer(step: OpenSpielRawPlayer[] | undefined): number {
  if (!Array.isArray(step)) return -1;
  return step.findIndex((p) => {
    const submission = p.action?.submission;
    return submission !== undefined && submission !== null && submission >= 0;
  });
}

export function cardText(card: HanabiCardFace): string {
  return `${card.color}${card.rank}`;
}
