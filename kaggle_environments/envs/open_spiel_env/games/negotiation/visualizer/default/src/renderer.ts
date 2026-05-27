import type { RendererOptions } from '@kaggle-environments/core';

const ITEM_COLORS = ['#1f4f8b', '#9a3324', '#3c6e3c', '#7a5a1f', '#5c3a73', '#b6862c'];
const ITEM_NAMES = ['apples', 'pears', 'grapes', 'plums', 'figs', 'limes'];

interface NegotiationObs {
  current_player: number;
  viewing_player: number;
  turn_type: 'proposal' | 'utterance' | null;
  max_steps: number;
  item_pool: number[];
  my_utilities: number[];
  proposals: Array<{ player: number; items?: number[]; accept: boolean }>;
  utterances: Array<{ player: number; symbols: number[] }>;
  most_recent_proposal: number[] | null;
  most_recent_utterance: number[] | null;
  agreement_reached: boolean;
  is_terminal: boolean;
  winner: number | 'draw' | null;
  rewards: number[] | null;
  params: {
    num_items: number;
    num_symbols: number;
    utterance_dim: number;
    enable_utterances: boolean;
  };
}

function parseObservation(step: any, playerIdx: number): NegotiationObs | null {
  const raw = step?.[playerIdx]?.observation?.observationString;
  if (!raw) return null;
  try {
    return JSON.parse(raw) as NegotiationObs;
  } catch {
    return null;
  }
}

function itemSwatch(itemIdx: number, size: number): string {
  const color = ITEM_COLORS[itemIdx % ITEM_COLORS.length];
  return `<span class="neg-mini-item" style="background-color:${color};width:${size}px;height:${size}px;"></span>`;
}

function utilityChips(utils: number[] | null, hidden: boolean): string {
  if (hidden) {
    const placeholder = Array(3).fill(0);
    return placeholder
      .map(
        (_, i) =>
          `<span class="neg-util-chip hidden"><span class="neg-util-swatch" style="background-color:${ITEM_COLORS[i % ITEM_COLORS.length]}"></span>?</span>`
      )
      .join('');
  }
  if (!utils) return '';
  return utils
    .map(
      (v, i) =>
        `<span class="neg-util-chip"><span class="neg-util-swatch" style="background-color:${ITEM_COLORS[i % ITEM_COLORS.length]}"></span>${v}</span>`
    )
    .join('');
}

function renderPool(pool: number[]): string {
  return pool
    .map((qty, i) => {
      const color = ITEM_COLORS[i % ITEM_COLORS.length];
      const name = ITEM_NAMES[i % ITEM_NAMES.length];
      const items = Array(qty)
        .fill(0)
        .map(() => `<span class="neg-pool-item" style="background-color:${color}"></span>`)
        .join('');
      return `
        <div class="neg-pool-row">
          <div class="neg-pool-label"><span class="neg-pool-item" style="background-color:${color}"></span>${name}</div>
          <div class="neg-pool-items">${items || '<span style="opacity:0.4">none</span>'}</div>
          <div class="neg-pool-count">×${qty}</div>
        </div>`;
    })
    .join('');
}

function renderSplit(proposerItems: number[], pool: number[]): string {
  const proposerSide = proposerItems
    .map((q, i) => `<span class="neg-split-side">${itemSwatch(i, 12)}<span class="neg-mini-count">${q}</span></span>`)
    .join(' ');
  const otherSide = pool
    .map((p, i) => Math.max(0, p - (proposerItems[i] ?? 0)))
    .map((q, i) => `<span class="neg-split-side">${itemSwatch(i, 12)}<span class="neg-mini-count">${q}</span></span>`)
    .join(' ');
  return `
    <div class="neg-bubble-split">
      <span><b>keeps:</b> ${proposerSide}</span>
      <span class="neg-split-arrow">|</span>
      <span><b>offers:</b> ${otherSide}</span>
    </div>`;
}

function renderConversation(obs: NegotiationObs): string {
  const messages: string[] = [];
  const utterances = obs.utterances;
  const enableUtt = obs.params.enable_utterances;
  const lastNonAcceptByPlayer: Record<number, number[]> = {};

  for (let i = 0; i < obs.proposals.length; i++) {
    const p = obs.proposals[i];
    const isLatest = i === obs.proposals.length - 1;
    const sideClass = p.player === 0 ? 'from-p0' : 'from-p1';
    const utt = enableUtt && utterances[i] ? utterances[i] : null;
    if (p.accept) {
      // The acceptance references the most recent non-accept proposal from
      // the OTHER player. Show it as a clear "ACCEPTED" badge.
      const otherPlayer = 1 - p.player;
      const acceptedSplit = lastNonAcceptByPlayer[otherPlayer];
      messages.push(`
        <div class="neg-msg ${sideClass}">
          <div class="neg-bubble accept ${isLatest ? 'latest' : ''}">
            <div class="neg-bubble-who"><span class="neg-glyph" style="background-color:${p.player === 0 ? '#1f4f8b' : '#9a3324'}"></span>Player ${p.player + 1} ACCEPTS</div>
            ${acceptedSplit ? renderSplit(acceptedSplit, obs.item_pool) : ''}
            <span class="neg-utterance">(proposer keeps; other receives the remainder)</span>
          </div>
        </div>`);
    } else if (p.items) {
      lastNonAcceptByPlayer[p.player] = p.items;
      const uttHtml = utt ? `<span class="neg-utterance">utterance: [${utt.symbols.join(', ')}]</span>` : '';
      messages.push(`
        <div class="neg-msg ${sideClass}">
          <div class="neg-bubble ${isLatest ? 'latest' : ''}">
            <div class="neg-bubble-who"><span class="neg-glyph" style="background-color:${p.player === 0 ? '#1f4f8b' : '#9a3324'}"></span>Player ${p.player + 1} proposes</div>
            ${renderSplit(p.items, obs.item_pool)}
            ${uttHtml}
          </div>
        </div>`);
    }
  }

  if (messages.length === 0) {
    return `<div class="neg-log-empty">No proposals yet — Player 1 to open.</div>`;
  }
  return messages.join('');
}

function turnBadge(obs: NegotiationObs, step: number, totalSteps: number): string {
  if (obs.is_terminal) {
    return `<div class="neg-turn-badge">step ${step} of ${totalSteps - 1}</div>`;
  }
  const who = obs.current_player >= 0 ? `Player ${obs.current_player + 1}` : '—';
  const turn = obs.turn_type ?? 'proposal';
  return `<div class="neg-turn-badge">${who}'s ${turn}<br/>step ${step} of ${totalSteps - 1}</div>`;
}

function statusText(obs: NegotiationObs): string {
  if (!obs.is_terminal) {
    const proposalsSoFar = obs.proposals.filter((p) => !p.accept).length;
    return `<div>Round ${proposalsSoFar + 1} of up to ${obs.max_steps} · max quantity per item is ${5}</div>
      <div class="neg-status-sub">utterances are private symbols — they carry no game effect</div>`;
  }
  const r = obs.rewards ?? [0, 0];
  const result =
    obs.winner === 'draw'
      ? 'Tied utility'
      : obs.winner === 0
        ? 'Player 1 wins on utility'
        : obs.winner === 1
          ? 'Player 2 wins on utility'
          : 'Game over';
  const reason = obs.agreement_reached ? 'Deal accepted' : 'No agreement reached';
  return `<div><b>${reason}</b> — ${result}</div>
    <div class="neg-status-sub">final utility: Player 1 = ${r[0]} · Player 2 = ${r[1]}</div>`;
}

export function renderer(options: RendererOptions) {
  const { step, replay, parent } = options;
  const steps = (replay.steps as unknown as any[]) ?? [];
  const safeStep = Math.max(0, Math.min(step, steps.length - 1));
  const current = steps[safeStep];

  parent.innerHTML = `
    <div class="renderer-container">
      <div class="neg-header"></div>
      <div class="neg-pool sketched-border"></div>
      <div class="neg-log sketched-border"></div>
      <div class="neg-status sketched-border"></div>
    </div>
  `;
  const header = parent.querySelector('.neg-header') as HTMLDivElement;
  const pool = parent.querySelector('.neg-pool') as HTMLDivElement;
  const log = parent.querySelector('.neg-log') as HTMLDivElement;
  const status = parent.querySelector('.neg-status') as HTMLDivElement;
  if (!header || !pool || !log || !status) return;

  const obs0 = parseObservation(current, 0);
  const obs1 = parseObservation(current, 1);
  const obs = obs0 ?? obs1;
  if (!obs) {
    status.textContent = 'Waiting for replay…';
    return;
  }

  const isTerm = obs.is_terminal;
  const activeP0 = !isTerm && obs.current_player === 0;
  const activeP1 = !isTerm && obs.current_player === 1;

  // Reveal both utility vectors at terminal so the final score makes sense;
  // during play each player only sees their own.
  const p0Utils = obs0?.my_utilities ?? null;
  const p1Utils = obs1?.my_utilities ?? null;
  const r = obs.rewards ?? [null, null];

  const playerCard = (pid: 0 | 1, utils: number[] | null, active: boolean, reward: number | null): string => {
    const accepted =
      isTerm &&
      obs.agreement_reached &&
      obs.proposals.length > 0 &&
      obs.proposals[obs.proposals.length - 1].player === pid;
    const cls = `neg-player-card sketched-border p${pid}${active ? ' active' : ''}${accepted ? ' accepted' : ''}`;
    const scoreLine =
      reward !== null ? `<span class="neg-score">utility ${reward}</span>` : `<span class="neg-score">utility —</span>`;
    return `
      <div class="${cls}">
        <div class="neg-player-name">
          <span class="neg-player-tag"><span class="neg-glyph"></span>Player ${pid + 1}</span>
          ${scoreLine}
        </div>
        <div class="neg-util-row">${utilityChips(utils, false)}</div>
      </div>`;
  };

  header.innerHTML = `
    ${playerCard(0, p0Utils, activeP0, r[0])}
    ${turnBadge(obs, safeStep, steps.length)}
    ${playerCard(1, p1Utils, activeP1, r[1])}
  `;

  pool.innerHTML = `
    <div class="neg-pool-title"><span>Item pool</span><span class="neg-pool-sub">split it between the two players</span></div>
    ${renderPool(obs.item_pool)}
  `;

  log.innerHTML = renderConversation(obs);
  // Keep the most recent message in view.
  log.scrollTop = log.scrollHeight;

  status.innerHTML = statusText(obs);
}
