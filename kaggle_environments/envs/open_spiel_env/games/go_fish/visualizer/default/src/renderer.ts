import { escapeHtml, type RendererOptions } from '@kaggle-environments/core';
import type { GoFishStep } from './transformers/goFishTransformer';

interface GoFishPlayerInfo {
  player: number;
  cards: number;
  books: number;
}

interface GoFishEvent {
  type: 'ask' | 'draw';
  player: number;
  target?: number;
  rank: number;
  rank_label: string;
  received?: number;
  booked: boolean;
}

interface GoFishObservation {
  phase: string | null;
  current_player: number;
  observer: number;
  is_terminal: boolean;
  winner: number | string | null;
  returns: number[];
  hand: Record<string, number>;
  players: GoFishPlayerInfo[];
  recent_events: GoFishEvent[];
}

// A spectator view that merges both players' private hands onto the shared
// public state.
interface MergedGoFishObservation extends GoFishObservation {
  hands: { 0: Record<string, number>; 1: Record<string, number> };
}

const PLAYER_0_COLOR = '#1f77b4';
const PLAYER_1_COLOR = '#d62728';

// Column order for laying out a hand: standard-deck ranks first, then any extra
// letter-labelled ranks (non-standard ranks<13 or ranks>13 variants).
const STANDARD_RANKS = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K'];

function rankSortKey(label: string): number {
  const idx = STANDARD_RANKS.indexOf(label);
  if (idx >= 0) return idx;
  // Letter-labelled ranks (a, b, c, ...) sort after the standard set.
  return 100 + (label.charCodeAt(0) - 'a'.charCodeAt(0));
}

function parseObservation(step: any, playerIdx: number): GoFishObservation | null {
  const raw = step?.[playerIdx]?.observation?.observationString;
  if (!raw) return null;
  try {
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

function mergedObservation(step: any): MergedGoFishObservation | null {
  // Each player's observation only reveals their own hand contents; merge so
  // the spectator view shows both hands at once. Public fields (players, phase,
  // current_player, ...) are identical across both observations.
  const o0 = parseObservation(step, 0);
  const o1 = parseObservation(step, 1);
  const base = o0 ?? o1;
  if (!base) return null;
  const merged: MergedGoFishObservation = {
    ...(JSON.parse(JSON.stringify(base)) as GoFishObservation),
    hands: {
      0: o0 && o0.observer === 0 ? o0.hand : {},
      1: o1 && o1.observer === 1 ? o1.hand : {},
    },
  };
  return merged;
}

function getPlayerName(replay: any, idx: number): string {
  return replay?.info?.TeamNames?.[idx] ?? replay?.agents?.[idx]?.name ?? (idx === 0 ? 'Player 1' : 'Player 2');
}

function gameParams(replay: any): { ranks: number; suits: number } {
  const p = replay?.configuration?.openSpielGameParameters ?? {};
  return { ranks: p.ranks ?? 13, suits: p.suits ?? 4 };
}

function poolSize(players: GoFishPlayerInfo[], ranks: number, suits: number): number {
  // The observation doesn't expose the pool directly. Total cards = ranks*suits;
  // subtract cards held in hands and cards removed as completed books.
  const inHands = players.reduce((acc, p) => acc + (p.cards ?? 0), 0);
  const booked = players.reduce((acc, p) => acc + (p.books ?? 0), 0) * suits;
  return Math.max(0, ranks * suits - inHands - booked);
}

function buildRankCard(rankLabel: string, count: number, suits: number, highlight: boolean): HTMLDivElement {
  const el = document.createElement('div');
  el.className = 'gf-card' + (highlight ? ' highlight' : '') + (count >= suits ? ' complete' : '');
  el.innerHTML = `<span class="gf-rank">${escapeHtml(rankLabel)}</span>` + `<span class="gf-count">×${count}</span>`;
  return el;
}

function buildFaceDownHand(cardCount: number): HTMLDivElement {
  const el = document.createElement('div');
  el.className = 'gf-hand gf-hand--facedown';
  const count = Math.max(0, cardCount);
  if (count === 0) {
    const empty = document.createElement('div');
    empty.className = 'gf-empty';
    empty.textContent = 'no cards';
    el.appendChild(empty);
    return el;
  }
  for (let i = 0; i < count; i++) {
    const c = document.createElement('div');
    c.className = 'gf-card gf-card--facedown';
    el.appendChild(c);
  }
  return el;
}

function buildOwnHand(hand: Record<string, number>, suits: number, highlightRank: string | null): HTMLDivElement {
  const el = document.createElement('div');
  el.className = 'gf-hand';
  const labels = Object.keys(hand).sort((a, b) => rankSortKey(a) - rankSortKey(b));
  if (labels.length === 0) {
    const empty = document.createElement('div');
    empty.className = 'gf-empty';
    empty.textContent = 'no cards';
    el.appendChild(empty);
    return el;
  }
  for (const label of labels) {
    el.appendChild(buildRankCard(label, hand[label], suits, label === highlightRank));
  }
  return el;
}

function renderPlayerRow(
  container: HTMLDivElement,
  name: string,
  info: GoFishPlayerInfo | undefined,
  hand: Record<string, number> | null,
  isActive: boolean,
  isWinner: boolean,
  color: string,
  showFaceDown: boolean,
  highlightRank: string | null
) {
  container.innerHTML = '';
  const meta = document.createElement('div');
  meta.className = 'player-meta';
  const cards = info?.cards ?? 0;
  const books = info?.books ?? 0;
  meta.innerHTML =
    `<span class="sketched-border name-pill" style="color:${color};">` +
    `${escapeHtml(name)}${isActive ? ' ▶' : ''}${isWinner ? ' ★' : ''}</span>` +
    `<span class="stat sketched-border">Cards: ${cards}</span>` +
    `<span class="stat sketched-border">Books: ${books}</span>`;
  container.appendChild(meta);
  container.appendChild(showFaceDown ? buildFaceDownHand(cards) : buildOwnHand(hand ?? {}, 4, highlightRank));
}

function buildDeckPile(pool: number): HTMLDivElement {
  const pile = document.createElement('div');
  pile.className = 'gf-pile';
  const stack = document.createElement('div');
  stack.className = 'gf-pile-stack';
  if (pool === 0) {
    const slot = document.createElement('div');
    slot.className = 'gf-card--slot';
    stack.appendChild(slot);
  } else {
    for (let i = 0; i < Math.min(pool, 4); i++) {
      const c = document.createElement('div');
      c.className = 'gf-card gf-card--facedown';
      stack.appendChild(c);
    }
  }
  pile.appendChild(stack);
  const label = document.createElement('div');
  label.className = 'gf-pile-label';
  label.textContent = `Pool (${pool})`;
  pile.appendChild(label);
  return pile;
}

function phaseLabel(phase: string | null): string {
  if (!phase) return '';
  // Split camelCase (e.g. "EmptyDraw" -> "Empty Draw").
  return phase.replace(/([a-z])([A-Z])/g, '$1 $2');
}

// The most recent event describes what the last mover did and its outcome.
// recent_events in a player's observation are the opponent's actions since that
// player last moved, so we scan both players' observations and take the newest.
function findLastEvent(step: any): GoFishEvent | null {
  const events: GoFishEvent[] = [];
  for (const idx of [0, 1]) {
    const obs = parseObservation(step, idx);
    if (obs?.recent_events?.length) events.push(...obs.recent_events);
  }
  return events.length ? events[events.length - 1] : null;
}

function describeEvent(ev: GoFishEvent, playerNames: string[]): string {
  const who = playerNames[ev.player] ?? `Player ${ev.player + 1}`;
  if (ev.type === 'ask') {
    const target = playerNames[ev.target ?? 0] ?? `Player ${(ev.target ?? 0) + 1}`;
    if (ev.received && ev.received > 0) {
      let msg = `${who} asked ${target} for ${ev.rank_label} and took ${ev.received}`;
      if (ev.booked) msg += ` — booked ${ev.rank_label}!`;
      return msg;
    }
    return `${who} asked ${target} for ${ev.rank_label} — Go Fish!`;
  }
  // draw (fishing from the pool)
  let msg = `${who} drew from the pool`;
  if (ev.booked) msg += ` — booked ${ev.rank_label}!`;
  return msg;
}

function buildStatus(
  observation: GoFishObservation,
  lastEvent: string | null,
  playerNames: string[],
  activeIdx: number,
  winnerIdx: number,
  isTerminal: boolean,
  forfeitReason: string | null
): string {
  const parts: string[] = [];
  if (observation.phase) {
    parts.push(`<span class="phase-pill">${escapeHtml(phaseLabel(observation.phase))}</span>`);
  }
  if (lastEvent) {
    parts.push(`<span class="annotation">${escapeHtml(lastEvent)}</span>`);
  } else if (!isTerminal && activeIdx >= 0) {
    const color = activeIdx === 0 ? PLAYER_0_COLOR : PLAYER_1_COLOR;
    parts.push(
      `<span>Turn: <span style="color:${color};font-weight:700;">${escapeHtml(playerNames[activeIdx])}</span></span>`
    );
  }
  if (isTerminal) {
    let html: string;
    if (winnerIdx === 0) {
      html = `<span style="color:${PLAYER_0_COLOR};font-weight:700;">${escapeHtml(playerNames[0])} wins</span>`;
    } else if (winnerIdx === 1) {
      html = `<span style="color:${PLAYER_1_COLOR};font-weight:700;">${escapeHtml(playerNames[1])} wins</span>`;
    } else {
      html = `<span>Game over: draw</span>`;
    }
    parts.push(html);
    if (forfeitReason) {
      parts.push(`<span class="annotation forfeit-reason">${escapeHtml(forfeitReason)}</span>`);
    }
  }
  return parts.join(' ');
}

export function renderer(options: RendererOptions<GoFishStep[]>) {
  const { parent, replay, step } = options;
  const steps = (replay?.steps ?? []) as GoFishStep[];
  if (!steps.length) return;

  parent.innerHTML = `
    <div class="renderer-container">
      <div class="header"></div>
      <div class="table">
        <div class="player-row top-row"></div>
        <div class="center-row"></div>
        <div class="player-row bottom-row"></div>
      </div>
      <div class="status-container sketched-border"></div>
    </div>
  `;
  const header = parent.querySelector('.header') as HTMLDivElement;
  const topRow = parent.querySelector('.top-row') as HTMLDivElement;
  const centerRow = parent.querySelector('.center-row') as HTMLDivElement;
  const bottomRow = parent.querySelector('.bottom-row') as HTMLDivElement;
  const statusContainer = parent.querySelector('.status-container') as HTMLDivElement;

  const stepData = steps[step];
  const currentStep = stepData?.rawStep;
  const observation = mergedObservation(currentStep);
  if (!observation) {
    statusContainer.textContent = 'Waiting for first observation...';
    return;
  }

  const { ranks, suits } = gameParams(replay);
  const playerNames = [getPlayerName(replay, 0), getPlayerName(replay, 1)];
  // Prefer the transformer-supplied isTerminal -- it also fires on forfeits,
  // which the raw OpenSpiel observation.is_terminal does not.
  const isTerminal = !!stepData?.isTerminal || observation.is_terminal;
  const forfeitReason = stepData?.forfeitReason ?? null;
  const forfeiterIdx = stepData?.players?.findIndex((p) => p.forfeited) ?? -1;
  const activeIdx = isTerminal ? -1 : observation.current_player;
  let winnerIdx = typeof observation.winner === 'number' ? observation.winner : -1;
  if (winnerIdx < 0 && forfeitReason && forfeiterIdx >= 0) {
    winnerIdx = 1 - forfeiterIdx;
  }

  header.innerHTML = `
    <span class="player sketched-border ${activeIdx === 0 ? 'active' : ''}" style="color: ${PLAYER_0_COLOR};">
      ${escapeHtml(playerNames[0])}
    </span>
    <span class="vs">vs</span>
    <span class="player sketched-border ${activeIdx === 1 ? 'active' : ''}" style="color: ${PLAYER_1_COLOR};">
      ${escapeHtml(playerNames[1])}
    </span>
  `;

  const hands = observation.hands;
  const info0 = observation.players?.find((p) => p.player === 0);
  const info1 = observation.players?.find((p) => p.player === 1);

  // Highlight the rank involved in the most recent event, in the mover's own
  // hand, so the highlighted card matches the narrated move in the status bar.
  const lastEvent = findLastEvent(currentStep);
  let highlightRank0: string | null = null;
  let highlightRank1: string | null = null;
  if (lastEvent) {
    if (lastEvent.player === 0) highlightRank0 = lastEvent.rank_label;
    if (lastEvent.player === 1) highlightRank1 = lastEvent.rank_label;
  }

  renderPlayerRow(
    topRow,
    playerNames[1],
    info1,
    hands?.[1] ?? {},
    activeIdx === 1,
    winnerIdx === 1,
    PLAYER_1_COLOR,
    false,
    highlightRank1
  );

  centerRow.innerHTML = '';
  centerRow.appendChild(buildDeckPile(poolSize(observation.players ?? [], ranks, suits)));

  renderPlayerRow(
    bottomRow,
    playerNames[0],
    info0,
    hands?.[0] ?? {},
    activeIdx === 0,
    winnerIdx === 0,
    PLAYER_0_COLOR,
    false,
    highlightRank0
  );

  statusContainer.innerHTML = buildStatus(
    observation,
    lastEvent ? describeEvent(lastEvent, playerNames) : null,
    playerNames,
    activeIdx,
    winnerIdx,
    isTerminal,
    forfeitReason
  );
}
