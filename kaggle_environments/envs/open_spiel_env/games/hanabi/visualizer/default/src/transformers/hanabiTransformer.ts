// Hanabi replay transformer.
//
// Builds the per-step `players` array the side-panel Game Log needs, and
// narrates each move. The renderer consumes the raw step separately (via
// `rawStep`) so it can merge both players' private views.
//
// Two things make Hanabi different from the other OpenSpiel transformers:
//
//   1. It is fully cooperative. Both seats always receive the same reward, so
//      the shared `deriveWinnerFromRewards` helper would label every natural
//      ending a "Draw". Instead we report the team result: the fireworks score
//      out of the maximum, plus why the game ended. A forfeit is the only case
//      that produces asymmetric rewards, and it keeps the shared wording.
//
//   2. Narrating a move needs the state from *before* it. Step N's observation
//      is the state after step N's action, so the played/discarded card and the
//      set of cards a hint touched are read from step N-1 -- and only from an
//      observation belonging to some other player, since the actor's own cards
//      are hidden in their own view. `mergedObservation` handles that.

import {
  detectForfeit,
  buildForfeitReason,
  parseThoughts,
  OpenSpielRawPlayer,
  ForfeitInfo,
} from '@kaggle-environments/core';

import { HanabiObservation, actingPlayer, cardText, colorName, mergedObservation, parseMove } from '../observation';

interface HanabiPlayer {
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

export interface HanabiStep {
  step: number;
  players: HanabiPlayer[];
  isTerminal: boolean;
  /**
   * The cooperative result line, e.g. "Team scored 18 / 25". Null until the
   * game ends. Named `winner` only because the side panel expects that key --
   * Hanabi has no winner.
   */
  winner: string | null;
  /** Non-null when the game ended early because a player forfeited. */
  forfeitReason: string | null;
  /** Plain-language description of the move made on this step, if any. */
  moveSummary: string | null;
  rawStep: OpenSpielRawPlayer[];
}

/** Describe a move using the state from before it and the state after. */
function describeMove(
  actor: number,
  actionString: string | null | undefined,
  before: HanabiObservation | null,
  after: HanabiObservation | null
): string | null {
  if (!before) return null;
  const move = parseMove(actionString, actor, before.num_players);
  if (!move) return null;

  if (move.type === 'hint') {
    const hand = before.hands.find((h) => h.player === move.target);
    const what = 'color' in move ? colorName(move.color) : `${move.rank}s`;
    const touched = (hand?.cards ?? []).filter((c) =>
      'color' in move ? c.card?.color === move.color : c.card?.rank === move.rank
    ).length;
    const plural = touched === 1 ? 'card' : 'cards';
    return `hinted P${move.target + 1}: ${what} (${touched} ${plural})`;
  }

  const card = before.hands.find((h) => h.player === actor)?.cards[move.slot]?.card;
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
  numPlayers: number,
  fallback: string
): string {
  const move = parseMove(actionString, actor, numPlayers);
  if (!move) return fallback;
  if (move.type === 'play') return `Play slot ${move.slot + 1}`;
  if (move.type === 'discard') return `Discard slot ${move.slot + 1}`;
  const what = 'color' in move ? colorName(move.color) : `rank ${move.rank}`;
  return `Hint P${move.target + 1}: ${what}`;
}

/**
 * The cooperative result line. Bombing out zeroes the score no matter how many
 * fireworks were launched, so that case says so explicitly rather than
 * reporting a number the team did not actually earn.
 */
function teamResult(observation: HanabiObservation | null, forfeit: ForfeitInfo | null): string | null {
  if (forfeit) return null;
  if (!observation) return 'Game over';
  if (observation.outcome === 'lives_exhausted') {
    // `score` is already 0 here, so the count of what was lost comes from the
    // stack heights.
    const lost = observation.fireworks_total;
    return `Out of lives — team scores 0 (${lost} firework${lost === 1 ? '' : 's'} lost)`;
  }
  if (observation.outcome === 'perfect_score') {
    return `Perfect score! Team scored ${observation.score} / ${observation.max_score}`;
  }
  return `Team scored ${observation.score} / ${observation.max_score}`;
}

export const hanabiTransformer = (environment: any): HanabiStep[] => {
  const rawSteps: OpenSpielRawPlayer[][] = environment?.steps ?? [];
  // Hanabi seats 2-5 players, so the fallback is sized from the replay rather
  // than being a two-name literal.
  const seatCount = rawSteps[0]?.length ?? 2;
  const teamNames: string[] =
    environment?.info?.TeamNames ?? Array.from({ length: seatCount }, (_, i) => `Player ${i + 1}`);
  const out: HanabiStep[] = [];

  rawSteps.forEach((step, index) => {
    const forfeit = detectForfeit(step);
    const observation = mergedObservation(step);
    const previous = mergedObservation(rawSteps[index - 1]);
    const actor = actingPlayer(step);
    const numPlayers = observation?.num_players ?? previous?.num_players ?? teamNames.length;

    const players: HanabiPlayer[] = step.map((p, i): HanabiPlayer => {
      const submission = p.action?.submission;
      const isForfeiter = forfeit?.index === i;
      // A forfeiter submits -1 but is still "acting", so their thoughts and
      // last attempt keep rendering in the side panel.
      const isTurn = (submission !== undefined && submission !== null && submission !== -1) || isForfeiter;
      const fallback = p.action?.actionString ?? '';
      return {
        id: i,
        name: teamNames[i] ?? `Player ${i + 1}`,
        thumbnail: '',
        isTurn,
        // Prefer a decoded label over OpenSpiel's "(Reveal player +1 color G)".
        // A forfeiter's actionString is their last *illegal* attempt, so it is
        // shown verbatim rather than dressed up as a move that was played.
        actionDisplayText:
          isTurn && !isForfeiter ? actionLabel(p.action?.actionString, i, numPlayers, fallback) : fallback,
        thoughts: parseThoughts(p.action),
        reward: p.reward ?? 0,
        generateReturns: p.action?.generate_returns ?? null,
        forfeited: isForfeiter,
        forfeitLastAttempt: isForfeiter ? (p.action?.actionString ?? null) : null,
      };
    });

    const observationTerminal = !!step[0]?.observation?.isTerminal;
    // A forfeit ends the episode even though OpenSpiel's own state is not
    // terminal; treat it as terminal so the renderer stops showing "Turn: X".
    const isTerminal = observationTerminal || forfeit !== null;

    const moveSummary =
      actor >= 0 ? describeMove(actor, step[actor]?.action?.actionString, previous, observation) : null;

    // Setup and chance steps carry no action, but the renderer reads every step
    // through `rawStep`, so they are kept rather than filtered out.
    out.push({
      step: index,
      players,
      isTerminal,
      winner: isTerminal ? teamResult(observation, forfeit) : null,
      forfeitReason: forfeit ? buildForfeitReason(forfeit, teamNames) : null,
      moveSummary,
      rawStep: step,
    });
  });

  return out;
};
