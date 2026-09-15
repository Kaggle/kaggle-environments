// Hanabi Arena renderer.
//
// Two teams play the *same deal* at two separate tables, so the screen is two
// mirrored boards side by side and the interesting read is the divergence
// between them: identical opening hands, different scores.
//
// Within a table, the whole game is the gap between what a card *is* and what
// its holder *knows*, so every card is drawn twice over: the face (visible to
// the spectator, hidden from its owner) and, beneath it, the knowledge strip --
// what the holder has been told and what they can still deduce.
//
// Only one table is live at a time; the active seat is highlighted and the
// idle table dims. A table that has already finished stays on screen with its
// final board, since the comparison is the point.

import { escapeHtml, type RendererOptions } from '@kaggle-environments/core';

import {
  COLOR_INK,
  COLOR_ORDER,
  COLOR_TINT,
  HanabiCard,
  HanabiTable,
  TEAM_COLORS,
  actingPlayer,
  cardText,
  mergedTables,
  parseMove,
  teamName,
} from './observation';
import type { HanabiArenaStep } from './transformers/hanabiArenaTransformer';

const PLAYERS_PER_TEAM = 2;

// Two shades per team so the seats of a table are distinguishable while still
// reading as one side.
const SEAT_COLORS = ['#1f77b4', '#3a9af0', '#d62728', '#ff7f6e'];

function teamColor(teamId: number): string {
  return TEAM_COLORS[teamId % TEAM_COLORS.length];
}

function playerColor(playerId: number): string {
  return SEAT_COLORS[playerId % SEAT_COLORS.length];
}

function getPlayerName(replay: any, idx: number): string {
  return replay?.info?.TeamNames?.[idx] ?? replay?.agents?.[idx]?.name ?? `Player ${idx + 1}`;
}

function getPlayerNames(replay: any, numPlayers: number): string[] {
  return Array.from({ length: numPlayers }, (_, i) => getPlayerName(replay, i));
}

/** Colors present in this game, in display order, plus any unexpected extras. */
function colorsOf(table: HanabiTable): string[] {
  const present = Object.keys(table.fireworks);
  const ordered = COLOR_ORDER.filter((c) => present.includes(c));
  return [...ordered, ...present.filter((c) => !ordered.includes(c))];
}

/** What changed at a table on this step, so the renderer can point at it. */
interface MoveHighlight {
  /** Seat whose hand a hint touched, and which of its slots. */
  hintTarget: number | null;
  hintSlots: Set<number>;
  /** Firework that just advanced. */
  playedColor: string | null;
  /** The discard pile grew (by a discard or a misplay). */
  discardAdded: boolean;
  /** Seat that drew a replacement, and the slot it landed in. */
  drew: { seat: number; slot: number } | null;
}

const NO_HIGHLIGHT: MoveHighlight = {
  hintTarget: null,
  hintSlots: new Set(),
  playedColor: null,
  discardAdded: false,
  drew: null,
};

function computeHighlight(
  actorId: number,
  actionString: string | null | undefined,
  before: HanabiTable | null,
  after: HanabiTable | null
): MoveHighlight {
  if (actorId < 0 || !before || !after) return NO_HIGHLIGHT;
  const seat = before.hands.find((h) => h.player_id === actorId)?.player;
  if (seat === undefined) return NO_HIGHLIGHT;
  const move = parseMove(actionString, seat, before.num_players);
  if (!move) return NO_HIGHLIGHT;

  if (move.type === 'hint') {
    // A hint shifts nothing, so the touched slots are the same before and after.
    const hand = after.hands.find((h) => h.player === move.target);
    const slots = new Set<number>();
    (hand?.cards ?? []).forEach((card, i) => {
      const hit = 'color' in move ? card.card?.color === move.color : card.card?.rank === move.rank;
      if (hit) slots.add(i);
    });
    return { ...NO_HIGHLIGHT, hintSlots: slots, hintTarget: move.target };
  }

  // Playing or discarding removes a card, the rest shift left, and the
  // replacement (if the deck still has one) lands in the last slot.
  const handAfter = after.hands.find((h) => h.player === seat)?.cards.length ?? 0;
  const handBefore = before.hands.find((h) => h.player === seat)?.cards.length ?? 0;
  const played = before.hands.find((h) => h.player === seat)?.cards[move.slot]?.card ?? null;
  return {
    hintTarget: null,
    hintSlots: new Set(),
    // Stack heights, not banked score: the score drops to 0 on a bomb-out and
    // would read as "no firework advanced" on an unrelated later comparison.
    playedColor: after.fireworks_total > before.fireworks_total ? (played?.color ?? null) : null,
    discardAdded: after.discards.length > before.discards.length,
    // The hand only stays full if the deck had a card left to replace with.
    drew: handAfter === handBefore && handAfter > 0 ? { seat, slot: handAfter - 1 } : null,
  };
}

function delta(before: number | undefined, after: number): string {
  if (before === undefined || before === after) return '';
  const diff = after - before;
  const sign = diff > 0 ? '+' : '';
  return `<span class="delta ${diff > 0 ? 'up' : 'down'}">${sign}${diff}</span>`;
}

function buildTokens(table: HanabiTable, previous: HanabiTable | null): string {
  const lives = Array.from(
    { length: table.max_life_tokens },
    (_, i) => `<span class="token life ${i < table.life_tokens ? '' : 'spent'}">&#9829;</span>`
  ).join('');
  const info = Array.from(
    { length: table.max_info_tokens },
    (_, i) => `<span class="token info ${i < table.info_tokens ? '' : 'spent'}">&#9679;</span>`
  ).join('');
  return `
    <span class="gauge sketched-border">
      <span class="gauge-label">Lives</span>${lives}${delta(previous?.life_tokens, table.life_tokens)}
    </span>
    <span class="gauge sketched-border">
      <span class="gauge-label">Info</span>${info}${delta(previous?.info_tokens, table.info_tokens)}
    </span>
    <span class="gauge sketched-border">
      <span class="gauge-label">Deck</span><span class="gauge-value">${table.deck_size}</span>
    </span>
  `;
}

function buildFireworks(table: HanabiTable, highlight: MoveHighlight): string {
  const piles = colorsOf(table)
    .map((color) => {
      const height = table.fireworks[color] ?? 0;
      const ink = COLOR_INK[color] ?? '#444343';
      const rungs = Array.from({ length: table.ranks }, (_, i) => {
        const rank = i + 1;
        const lit = rank <= height;
        return `<span class="rung ${lit ? 'lit' : ''}" style="${lit ? `background:${ink};` : ''}">${rank}</span>`;
      })
        .reverse()
        .join('');
      const advanced = highlight.playedColor === color ? ' advanced' : '';
      return `
        <div class="firework${advanced}" style="--ink:${ink};">
          <div class="rungs">${rungs}</div>
          <div class="firework-label" style="color:${ink};">${escapeHtml(color)}</div>
        </div>`;
    })
    .join('');
  return `<div class="fireworks">${piles}</div>`;
}

/** The knowledge strip: what the holder has been told, and what is left open. */
function buildKnowledge(card: HanabiCard, table: HanabiTable): string {
  const pips = colorsOf(table)
    .map((color) => {
      const open = card.plausible_colors.includes(color);
      const told = card.hinted_color === color;
      const ink = COLOR_INK[color] ?? '#444343';
      return `<span class="pip ${open ? 'open' : ''} ${told ? 'told' : ''}" style="--ink:${ink};"></span>`;
    })
    .join('');
  const ranks = Array.from({ length: table.ranks }, (_, i) => {
    const rank = i + 1;
    const open = card.plausible_ranks.includes(rank);
    const told = card.hinted_rank === rank;
    return `<span class="rank-mark ${open ? 'open' : ''} ${told ? 'told' : ''}">${rank}</span>`;
  }).join('');
  return `<div class="knowledge"><div class="pips">${pips}</div><div class="rank-marks">${ranks}</div></div>`;
}

function buildCard(card: HanabiCard, table: HanabiTable, classes: string[]): string {
  const face = card.card;
  const ink = face ? (COLOR_INK[face.color] ?? '#444343') : '#8b8b8b';
  const tint = face ? (COLOR_TINT[face.color] ?? '#ffffff') : '#ffffff';
  const faceHtml = face
    ? `<span class="card-rank" style="color:${ink};">${face.rank}</span>
       <span class="card-color" style="color:${ink};">${escapeHtml(face.color)}</span>`
    : `<span class="card-rank unknown">?</span>`;
  return `
    <div class="hb-card ${classes.join(' ')}" style="--ink:${ink};background-color:${tint};">
      <div class="card-face">${faceHtml}</div>
      ${buildKnowledge(card, table)}
    </div>`;
}

function buildHand(
  table: HanabiTable,
  seat: number,
  name: string,
  isActive: boolean,
  highlight: MoveHighlight
): string {
  const hand = table.hands.find((h) => h.player === seat);
  const cards = (hand?.cards ?? [])
    .map((card, slot) => {
      const classes: string[] = [];
      if (highlight.hintTarget === seat && highlight.hintSlots.has(slot)) classes.push('hinted');
      if (highlight.drew?.seat === seat && highlight.drew.slot === slot) classes.push('drawn');
      return buildCard(card, table, classes);
    })
    .join('');
  const body = cards || '<div class="hb-empty">no cards</div>';
  const ink = playerColor(hand?.player_id ?? seat);
  return `
    <div class="hand-panel sketched-border ${isActive ? 'active' : ''}">
      <div class="hand-header" style="color:${ink};">
        ${escapeHtml(name)}${isActive ? ' <span class="turn-arrow">&#9654;</span>' : ''}
      </div>
      <div class="hb-hand">${body}</div>
    </div>`;
}

function buildDiscards(table: HanabiTable, highlight: MoveHighlight): string {
  if (!table.discards.length) {
    return `<div class="discards"><span class="discards-label">Discards</span><span class="hb-empty">none</span></div>`;
  }
  const chips = table.discards
    .map((card, i) => {
      const ink = COLOR_INK[card.color] ?? '#444343';
      const fresh = highlight.discardAdded && i === table.discards.length - 1 ? ' fresh' : '';
      return `<span class="discard-chip${fresh}" style="color:${ink};border-color:${ink};">${escapeHtml(cardText(card))}</span>`;
    })
    .join('');
  return `<div class="discards"><span class="discards-label">Discards</span>${chips}</div>`;
}

/** How a table ended, shown under its score once it is finished. */
function outcomeNote(table: HanabiTable): string {
  if (!table.is_terminal) return '';
  if (table.outcome === 'lives_exhausted') return 'out of lives — scores 0';
  if (table.outcome === 'perfect_score') return 'perfect score';
  if (table.outcome === 'deck_exhausted') return 'deck exhausted';
  return table.outcome.replace(/_/g, ' ');
}

function buildTableColumn(
  table: HanabiTable | null,
  teamId: number,
  playerNames: string[],
  activePlayerId: number,
  highlight: MoveHighlight,
  previous: HanabiTable | null
): string {
  const banner =
    `<div class="team-banner sketched-border" style="border-color:${teamColor(teamId)};color:${teamColor(teamId)};">` +
    `${escapeHtml(teamName(teamId))}</div>`;
  if (!table) {
    return `<div class="table-column" data-team="${teamId}">${banner}<div class="hb-empty">no data</div></div>`;
  }

  const isActiveTable = table.hands.some((h) => h.player_id === activePlayerId);
  const tableHighlight = isActiveTable ? highlight : NO_HIGHLIGHT;
  const note = outcomeNote(table);

  const hands = table.hands
    .slice()
    .sort((a, b) => a.player - b.player)
    .map((hand) =>
      buildHand(
        table,
        hand.player,
        playerNames[hand.player_id] ?? `Player ${hand.player_id + 1}`,
        hand.player_id === activePlayerId,
        tableHighlight
      )
    )
    .join('');

  return `
    <div class="table-column ${table.is_terminal ? 'finished' : ''}" data-team="${teamId}">
      ${banner}
      <div class="table-score sketched-border">
        <span class="gauge-label">Score</span>
        <span class="gauge-value">${table.score} / ${table.max_score}</span>
        ${delta(previous?.score, table.score)}
        ${note ? `<span class="outcome-note">${escapeHtml(note)}</span>` : ''}
      </div>
      <div class="gauges">${buildTokens(table, previous)}</div>
      ${buildFireworks(table, tableHighlight)}
      <div class="hands">${hands}</div>
      ${buildDiscards(table, tableHighlight)}
    </div>`;
}

function buildStatus(stepData: HanabiArenaStep | undefined, playerNames: string[], activePlayerId: number): string {
  const parts: string[] = [];
  if (stepData?.moveSummary) {
    const teamId = stepData.moveTeamId ?? 0;
    parts.push(
      `<span class="annotation" style="color:${teamColor(teamId)};">` +
        `${escapeHtml(teamName(teamId))}: ${escapeHtml(stepData.moveSummary)}</span>`
    );
  }
  if (stepData?.isTerminal) {
    const result = stepData.forfeitReason ? 'Game over' : (stepData.winner ?? 'Game over');
    parts.push(`<p class="result">${escapeHtml(result)}</p>`);
    if (stepData.forfeitReason) {
      parts.push(`<span class="annotation forfeit-reason">${escapeHtml(stepData.forfeitReason)}</span>`);
    }
  } else if (activePlayerId >= 0) {
    const name = playerNames[activePlayerId] ?? `Player ${activePlayerId + 1}`;
    parts.push(
      `<span>Turn: <span style="color:${playerColor(activePlayerId)};font-weight:700;">` +
        `${escapeHtml(name)}</span></span>`
    );
  }
  return parts.join(' ');
}

export function renderer(options: RendererOptions<HanabiArenaStep[]>) {
  const { parent, replay, step } = options;
  const steps = (replay?.steps ?? []) as HanabiArenaStep[];

  parent.innerHTML = `
    <div class="renderer-container">
      <div class="header"></div>
      <div class="boards-row"></div>
      <div class="status-container sketched-border"></div>
    </div>
  `;
  const header = parent.querySelector('.header') as HTMLDivElement;
  const boardsRow = parent.querySelector('.boards-row') as HTMLDivElement;
  const statusContainer = parent.querySelector('.status-container') as HTMLDivElement;
  if (!steps.length) {
    statusContainer.textContent = 'No replay data.';
    return;
  }

  const stepData = steps[step];
  const currentStep = stepData?.rawStep;
  const tables = mergedTables(currentStep);
  const previous = mergedTables(steps[step - 1]?.rawStep);
  const numPlayers = currentStep?.length || PLAYERS_PER_TEAM * 2;
  const playerNames = getPlayerNames(replay, numPlayers);

  // Team pills: two seats per side, with an explicit "vs" between the teams.
  const numTeams = Math.max(2, Math.ceil(numPlayers / PLAYERS_PER_TEAM));
  header.innerHTML = Array.from({ length: numTeams }, (_, teamId) => {
    const seats = playerNames.slice(teamId * PLAYERS_PER_TEAM, (teamId + 1) * PLAYERS_PER_TEAM);
    const names = seats
      .map((name, i) => {
        const pid = teamId * PLAYERS_PER_TEAM + i;
        return `<span class="seat-name" style="color:${playerColor(pid)};">${escapeHtml(name)}</span>`;
      })
      .join('<span class="amp">&amp;</span>');
    return (
      `<span class="team-pill sketched-border" style="border-color:${teamColor(teamId)};">` +
      `<span class="team-label" style="color:${teamColor(teamId)};">${escapeHtml(teamName(teamId))}</span>${names}</span>`
    );
  }).join('<span class="vs">vs</span>');

  if (!tables.length) {
    // The first step precedes the deal, so there is no board to draw yet.
    statusContainer.textContent = 'Dealing...';
    return;
  }

  // Prefer the transformer's isTerminal: it also fires on forfeits, which the
  // raw OpenSpiel observation does not mark terminal.
  const isTerminal = !!stepData?.isTerminal;
  const activePlayerId = isTerminal ? -1 : (stepData?.framing?.active_player_id ?? -1);

  const actor = actingPlayer(currentStep);
  const actorTeam = stepData?.moveTeamId ?? (actor >= 0 ? Math.floor(actor / PLAYERS_PER_TEAM) : null);
  const highlight =
    actorTeam === null
      ? NO_HIGHLIGHT
      : computeHighlight(
          actor,
          currentStep?.[actor]?.action?.actionString,
          previous[actorTeam] ?? null,
          tables[actorTeam] ?? null
        );

  boardsRow.innerHTML = Array.from({ length: numTeams }, (_, teamId) =>
    buildTableColumn(
      tables[teamId] ?? null,
      teamId,
      playerNames,
      // Only the table holding the active seat gets a turn marker.
      activePlayerId,
      highlight,
      previous[teamId] ?? null
    )
  ).join('');

  statusContainer.innerHTML = buildStatus(stepData, playerNames, activePlayerId);
}
