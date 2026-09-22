import { ASSET_STATES, THERAPEUTIC_AREAS, TRIAL_PHASES, type MarketAlert, type StepView } from './types';
import { formatMoney } from './utils';

export interface PanelRefs {
  root: HTMLElement;
  canvas: HTMLCanvasElement;
  playerCards: HTMLElement[];
  bd: HTMLElement;
  markets: HTMLElement;
  alerts: HTMLElement;
  spark: HTMLCanvasElement;
  status: HTMLElement;
}

const SPARK_COLORS = ['#1f6f9c', '#9c5a1f'];

function el(tag: string, className?: string, text?: string): HTMLElement {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function playerCard(index: number): HTMLElement {
  const card = el('div', `player-card player-${index}`);
  card.append(
    el('div', 'player-name'),
    el('div', 'player-stats'),
    el('div', 'player-sites'),
    el('div', 'player-badge')
  );
  return card;
}

export function buildShell(parent: HTMLElement): PanelRefs {
  parent.textContent = '';
  const root = el('div', 'renderer-container');

  const header = el('div', 'header');
  const cards = [playerCard(0), playerCard(1)];
  header.append(...cards);

  const board = el('div', 'board');
  const canvas = document.createElement('canvas');
  canvas.className = 'pipeline';
  board.append(canvas);

  const lower = el('div', 'lower');
  const bd = el('div', 'panel bd-panel');
  const markets = el('div', 'panel market-panel');
  const alerts = el('div', 'panel alert-panel');
  lower.append(bd, markets, alerts);

  const footer = el('div', 'footer');
  const spark = document.createElement('canvas');
  spark.className = 'spark';
  const status = el('div', 'status-container');
  footer.append(spark, status);

  root.append(header, board, lower, footer);
  parent.append(root);

  return { root, canvas, playerCards: cards, bd, markets, alerts, spark, status };
}

function renderPlayerCards(refs: PanelRefs, view: StepView) {
  const leader = view.players.reduce((best, p, i, all) => (p.enpv > all[best].enpv ? i : best), 0);
  view.players.forEach((player, i) => {
    const card = refs.playerCards[i];
    if (!card) return;
    card.classList.toggle('leading', view.players.length > 1 && i === leader && !player.bankrupt);
    card.classList.toggle('bankrupt', player.bankrupt);

    (card.children[0] as HTMLElement).textContent = player.name;
    (card.children[1] as HTMLElement).textContent =
      `cash ${formatMoney(player.cash)}  ·  eNPV ${formatMoney(player.enpv)}  ·  eROI ${player.eroi.toFixed(2)}`;

    const onMarket = player.assets.filter((a) => ASSET_STATES[a.state] === 'On Market').length;
    const inDev = player.assets.filter((a) => ASSET_STATES[a.state] === 'In Development').length;
    (card.children[2] as HTMLElement).textContent =
      `${inDev} in trials  ·  ${onMarket} on market  ·  ${player.operationalSites} sites` +
      (player.buildingSites ? ` (+${player.buildingSites} building)` : '') +
      (player.failedCount ? `  ·  ${player.failedCount} failed` : '');

    const badge = card.children[3] as HTMLElement;
    badge.textContent = player.bankrupt ? 'BANKRUPT' : i === leader ? 'LEADING' : '';
  });
}

function renderBd(refs: PanelRefs, view: StepView) {
  refs.bd.textContent = '';
  refs.bd.append(el('div', 'panel-title', 'Business development'));
  if (view.bdOffers.length === 0) {
    refs.bd.append(el('div', 'panel-empty', 'No assets on offer'));
    return;
  }
  for (const offer of view.bdOffers) {
    const row = el('div', 'panel-row');
    row.append(
      el('span', 'panel-key', offer.name),
      el('span', 'panel-val', `${TRIAL_PHASES[offer.phase] ?? '—'} · ${formatMoney(offer.maxRevenue)} peak`)
    );
    row.title = THERAPEUTIC_AREAS[offer.therapeuticArea] ?? '';
    refs.bd.append(row);
  }
}

function renderMarkets(refs: PanelRefs, view: StepView) {
  refs.markets.textContent = '';
  refs.markets.append(el('div', 'panel-title', 'Indication markets'));
  // Only contested or boosted indications are worth the space.
  const active = view.indicationMarkets.filter(
    ([, , firstMover, demand, drugs]) => drugs > 0 || firstMover || demand !== 1
  );
  if (active.length === 0) {
    refs.markets.append(el('div', 'panel-empty', 'No drugs on market yet'));
    return;
  }
  for (const [, name, firstMover, demand, drugs] of active.slice(0, 6)) {
    const row = el('div', 'panel-row');
    const parts = [`${drugs} drug${drugs === 1 ? '' : 's'}`];
    if (demand !== 1) parts.push(`demand ×${demand.toFixed(2)}`);
    if (firstMover) parts.push(`first: ${firstMover}`);
    row.append(el('span', 'panel-key', name), el('span', 'panel-val', parts.join(' · ')));
    refs.markets.append(row);
  }
}

function alertText(alert: MarketAlert): string {
  const area = THERAPEUTIC_AREAS[alert.therapeuticArea] ?? '';
  const who = alert.agentId;
  const d = alert.details ?? {};
  switch (alert.eventType) {
    case 'drug_release':
      return `${who} launched a drug in ${area} (${formatMoney(Number(d.max_revenue ?? 0))} peak)`;
    case 'bd_deal':
      return `${who} acquired ${d.asset_name ?? 'an asset'} for ${formatMoney(Number(d.price ?? 0))}`;
    case 'pipeline_leak':
      return `${who}'s ${area} programme reached ${d.new_phase ?? 'a new phase'}`;
    case 'clinical_site_deal':
      return `${who} won a clinical site for ${formatMoney(Number(d.price ?? 0))}`;
    case 'be_spend':
      return `${who} spent on brand equity in ${area}`;
    case 'dc_spend':
      return `${who} spent on demand creation in ${area}`;
    default:
      return `${who}: ${alert.eventType}`;
  }
}

function renderAlerts(refs: PanelRefs, view: StepView) {
  refs.alerts.textContent = '';
  refs.alerts.append(el('div', 'panel-title', 'Intelligence'));
  if (view.alerts.length === 0) {
    refs.alerts.append(el('div', 'panel-empty', 'Quiet on the market'));
    return;
  }
  const recent = [...view.alerts].sort((a, b) => b.step - a.step).slice(0, 5);
  for (const alert of recent) {
    const row = el('div', 'panel-row alert-row');
    row.append(el('span', 'panel-key', `t${alert.step}`), el('span', 'panel-val', alertText(alert)));
    refs.alerts.append(row);
  }
}

function renderSpark(refs: PanelRefs, view: StepView) {
  const canvas = refs.spark;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  if (w <= 0 || h <= 0) return;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);

  const series = view.players.map((p) => p.enpvSeries);
  const all = series.flat();
  if (all.length < 2) return;
  const min = Math.min(...all, 0);
  const max = Math.max(...all);
  const span = max - min || 1;
  const n = Math.max(...series.map((s) => s.length));

  // Zero line, so a swing into negative eNPV is legible.
  const zeroY = h - 2 - ((0 - min) / span) * (h - 4);
  ctx.save();
  ctx.setLineDash([3, 4]);
  ctx.strokeStyle = '#bbb8ae';
  ctx.beginPath();
  ctx.moveTo(0, zeroY);
  ctx.lineTo(w, zeroY);
  ctx.stroke();
  ctx.restore();

  series.forEach((values, i) => {
    if (values.length < 2) return;
    ctx.beginPath();
    values.forEach((value, x) => {
      const px = (x / Math.max(1, n - 1)) * (w - 2) + 1;
      const py = h - 2 - ((value - min) / span) * (h - 4);
      if (x === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });
    ctx.strokeStyle = SPARK_COLORS[i % SPARK_COLORS.length];
    ctx.lineWidth = 1.5;
    ctx.stroke();
  });
}

function statusText(view: StepView): string {
  const over = view.gameOver;
  if (!over) return `Step ${view.time} · eNPV leader ${view.players.reduce((a, b) => (b.enpv > a.enpv ? b : a)).name}`;

  const name = (i: number | null) => (i === null ? null : (view.players[i]?.name ?? `Player ${i + 1}`));
  if (over.kind === 'forfeit') {
    const offender = name(over.offender) ?? 'A player';
    const winner = name(over.winner);
    return winner
      ? `${offender} submitted an illegal action. ${winner} wins by default.`
      : `${offender} submitted an illegal action.`;
  }
  const winner = name(over.winner);
  const suffix = over.kind === 'bankruptcy' ? ' by bankruptcy' : '';
  return winner ? `${winner} wins${suffix}` : 'Draw';
}

export function renderPanels(refs: PanelRefs, view: StepView) {
  renderPlayerCards(refs, view);
  renderBd(refs, view);
  renderMarkets(refs, view);
  renderAlerts(refs, view);
  renderSpark(refs, view);
  refs.status.textContent = statusText(view);
  refs.status.classList.toggle('terminal', view.gameOver !== null);
}
