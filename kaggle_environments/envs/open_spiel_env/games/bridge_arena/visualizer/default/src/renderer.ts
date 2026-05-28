import type { RendererOptions } from '@kaggle-environments/core';

// External AABB ids: 0,1 = team A (N,S); 2,3 = team B (E,W).
const TEAM_A_PIDS = [0, 1] as const;
const TEAM_B_PIDS = [2, 3] as const;
const TEAM_COLORS = ['#1f77b4', '#d62728'] as const;
const TABLE_ORDER = ['W', 'N', 'E', 'S'] as const; // auction display order

type AuctionEntry = {
  internal_seat: number;
  table_position: string;
  player_id: number;
  team_id: number;
  call: string;
  action: number;
};

type PlayEntry = {
  card: string;
  action: number;
};

type Observation = {
  phase?: string;
  is_terminal?: boolean;
  dealer_table_position?: string;
  dealer_player_id?: number;
  current_player_id?: number | null;
  current_table_position?: string | null;
  current_team_id?: number | null;
  auction?: AuctionEntry[];
  plays?: PlayEntry[];
  table_seating?: Record<string, string>;
  teams?: Record<string, number[]>;
  returns?: number[];
  team_totals?: number[];
  winning_team?: number | string;
};

function getObservation(step: any, playerIdx: number): Observation | null {
  const raw = step?.[playerIdx]?.observation?.observationString;
  if (!raw) return null;
  try {
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

function getSharedObservation(step: any): Observation | null {
  if (!Array.isArray(step)) return null;
  for (let i = 0; i < step.length; i++) {
    const obs = getObservation(step, i);
    if (obs) return obs;
  }
  return null;
}

function getPlayerName(replay: any, idx: number): string {
  const info = replay?.info?.TeamNames?.[idx];
  if (info) return info;
  const fromAgent = replay?.agents?.[idx]?.name;
  if (fromAgent) return fromAgent;
  return `Player ${idx}`;
}

function teamColor(team: number): string {
  return TEAM_COLORS[team % TEAM_COLORS.length];
}

function teamLabel(team: number): string {
  return team === 0 ? 'A' : 'B';
}

function escapeHTML(s: string): string {
  return s.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
}

function isRedSuit(call: string): boolean {
  return call.includes('♥') || call.includes('♦');
}

function renderTeamBlock(
  team: number,
  pids: readonly number[],
  playerNames: string[],
  tableSeating: Record<string, string>,
  activePid: number | null
): string {
  const seats = pids
    .map((pid) => {
      const pos = tableSeating[String(pid)] ?? '?';
      const isActive = activePid === pid;
      return `
        <span class="seat ${isActive ? 'active' : ''}" style="color: ${teamColor(team)};">
          <span class="pos">${pos}</span>
          ${escapeHTML(playerNames[pid])}
        </span>
      `;
    })
    .join('');
  const seatPositions = pids.map((p) => tableSeating[String(p)] ?? '?').join('/');
  return `
    <span class="team-block sketched-border">
      <span class="team-label" style="color:${teamColor(team)};">Team ${teamLabel(team)} (${seatPositions})</span>
      <span class="seats">${seats}</span>
    </span>
  `;
}

function renderAuction(auction: AuctionEntry[], dealerPos: string | undefined, activePid: number | null): string {
  // Display columns are W,N,E,S (left to right). Each row holds the
  // next 4 calls in clockwise order. The dealer's column gets the very
  // first call; preceding columns in the first row are blank.
  const dealerCol = dealerPos ? TABLE_ORDER.indexOf(dealerPos as (typeof TABLE_ORDER)[number]) : 0;
  const offset = Math.max(0, dealerCol);
  const totalCells = offset + auction.length;
  const numRows = Math.max(1, Math.ceil(totalCells / 4));

  const headerRow = TABLE_ORDER.map((pos) => `<th>${pos}</th>`).join('');
  const bodyRows: string[] = [];
  for (let r = 0; r < numRows; r++) {
    const cells: string[] = [];
    for (let c = 0; c < 4; c++) {
      const cellIdx = r * 4 + c;
      if (cellIdx < offset) {
        cells.push('<td></td>');
        continue;
      }
      const auctionIdx = cellIdx - offset;
      const entry = auction[auctionIdx];
      if (!entry) {
        cells.push('<td></td>');
        continue;
      }
      const teamCls = entry.team_id === 0 ? 'team-a' : 'team-b';
      const isCurrent = auctionIdx === auction.length - 1 && activePid === null;
      const callHTML = isRedSuit(entry.call) ? `<span style="color:#d62728;">${entry.call}</span>` : entry.call;
      cells.push(`<td class="call ${teamCls} ${isCurrent ? 'current' : ''}">${callHTML}</td>`);
    }
    bodyRows.push(`<tr>${cells.join('')}</tr>`);
  }
  return `
    <table class="auction sketched-border">
      <thead><tr>${headerRow}</tr></thead>
      <tbody>${bodyRows.join('')}</tbody>
    </table>
  `;
}

function deriveContract(auction: AuctionEntry[]): {
  level?: number;
  denom?: string;
  declarerPid?: number;
  declarerPos?: string;
  doubled?: 'X' | 'XX' | null;
} {
  let lastBid: AuctionEntry | null = null;
  let doubled: 'X' | 'XX' | null = null;
  for (const entry of auction) {
    const call = entry.call;
    if (call === 'Pass') continue;
    if (call === 'Dbl') {
      doubled = 'X';
      continue;
    }
    if (call === 'RDbl') {
      doubled = 'XX';
      continue;
    }
    // It's a bid -- format "<level><denom>".
    lastBid = entry;
    doubled = null;
  }
  if (!lastBid) return {};

  const match = /^(\d)(♣|♦|♥|♠|NT)$/.exec(lastBid.call);
  if (!match) return {};
  const level = parseInt(match[1], 10);
  const denom = match[2];
  // Declarer: on the bidding side, first player to name the denomination.
  const winningSide = lastBid.team_id;
  let declarer: AuctionEntry | null = null;
  for (const entry of auction) {
    if (entry.team_id !== winningSide) continue;
    const m = /^\d(♣|♦|♥|♠|NT)$/.exec(entry.call);
    if (m && m[1] === denom) {
      declarer = entry;
      break;
    }
  }
  return {
    level,
    denom,
    declarerPid: declarer?.player_id,
    declarerPos: declarer?.table_position,
    doubled,
  };
}

function renderContract(
  auction: AuctionEntry[],
  phase: string | undefined,
  dealerPos: string | undefined,
  dealerPid: number | undefined,
  playerNames: string[]
): string {
  const contract = deriveContract(auction);
  const rows: string[] = [];
  rows.push(
    `<span class="row"><span class="label">Dealer</span><span class="value">${dealerPos ?? '?'}` +
      (dealerPid !== undefined ? ` &mdash; ${escapeHTML(playerNames[dealerPid])}` : '') +
      `</span></span>`
  );
  rows.push(`<span class="row"><span class="label">Phase</span><span class="value">${phase ?? '?'}</span></span>`);
  if (contract.level && contract.denom) {
    const dbl = contract.doubled ? ` ${contract.doubled}` : '';
    const denom = isRedSuit(contract.denom) ? `<span style="color:#d62728;">${contract.denom}</span>` : contract.denom;
    rows.push(
      `<span class="row"><span class="label">Contract</span><span class="value">${contract.level}${denom}${dbl}</span></span>`
    );
    if (contract.declarerPos !== undefined && contract.declarerPid !== undefined) {
      rows.push(
        `<span class="row"><span class="label">Declarer</span><span class="value">${contract.declarerPos} &mdash; ${escapeHTML(playerNames[contract.declarerPid])}</span></span>`
      );
    }
  } else if (auction.length > 0) {
    rows.push(`<span class="row"><span class="label">Contract</span><span class="value">— (passed out)</span></span>`);
  }
  return `
    <div class="panel contract sketched-border">
      <div class="panel-title">Contract</div>
      ${rows.join('')}
    </div>
  `;
}

function renderPlays(plays: PlayEntry[]): string {
  if (!plays.length) {
    return '';
  }
  const chips = plays
    .map((p) => {
      const red = isRedSuit(p.card);
      return `<span class="card-chip ${red ? 'red' : ''}">${p.card}</span>`;
    })
    .join('');
  return `
    <div class="panel sketched-border">
      <div class="panel-title">Cards played (${plays.length})</div>
      <div class="plays">${chips}</div>
    </div>
  `;
}

function renderStatus(obs: Observation, activePid: number | null, playerNames: string[]): string {
  if (obs.is_terminal) {
    const totals = obs.team_totals ?? [];
    const wt = obs.winning_team;
    if (wt === 'draw') {
      return `<span>Draw</span><span class="annotation">A ${totals[0] ?? '?'} &middot; B ${totals[1] ?? '?'}</span>`;
    }
    if (wt === 0 || wt === 1) {
      const team = Number(wt);
      return (
        `<span style="color:${teamColor(team)};">Team ${teamLabel(team)} wins</span>` +
        `<span class="annotation">${totals[team] ?? '?'} vs ${totals[1 - team] ?? '?'}</span>`
      );
    }
    return `<span>Game over</span>`;
  }
  if (activePid !== null && activePid !== undefined) {
    const pos = obs.current_table_position ?? '?';
    const team = obs.current_team_id ?? 0;
    return (
      `<span style="color:${teamColor(team)};">${pos} (player ${activePid}, ${escapeHTML(playerNames[activePid])}) to act</span>` +
      `<span class="annotation">phase: ${obs.phase ?? '?'}</span>`
    );
  }
  return `<span>Setting up...</span><span class="annotation">phase: ${obs.phase ?? '?'}</span>`;
}

export function renderer(options: RendererOptions) {
  const { parent, replay, step } = options;
  const steps = (replay?.steps ?? []) as any[];
  if (!steps.length) {
    parent.innerHTML = '';
    return;
  }

  const currentStep = steps[Math.min(step, steps.length - 1)];
  // The setup step (step 0) has no observationString. Fall back to the
  // next step that does so the table layout is visible immediately.
  let obs = getSharedObservation(currentStep);
  if (!obs) {
    for (let i = step + 1; i < steps.length; i++) {
      const next = getSharedObservation(steps[i]);
      if (next) {
        obs = next;
        break;
      }
    }
  }
  if (!obs) {
    parent.innerHTML = '<div class="renderer-container">Waiting for first observation...</div>';
    return;
  }

  const playerNames = Array.from({ length: 4 }, (_, i) => getPlayerName(replay, i));
  const tableSeating = obs.table_seating ?? { '0': 'N', '1': 'S', '2': 'E', '3': 'W' };
  const auction = obs.auction ?? [];
  const plays = obs.plays ?? [];
  const activePid =
    obs.is_terminal || obs.current_player_id === null || obs.current_player_id === undefined
      ? null
      : Number(obs.current_player_id);

  const statusCls = obs.is_terminal
    ? obs.winning_team === 0
      ? 'team-a-wins'
      : obs.winning_team === 1
        ? 'team-b-wins'
        : 'draw'
    : '';

  parent.innerHTML = `
    <div class="renderer-container">
      <div class="header">
        ${renderTeamBlock(0, TEAM_A_PIDS, playerNames, tableSeating, activePid)}
        <span class="vs">vs</span>
        ${renderTeamBlock(1, TEAM_B_PIDS, playerNames, tableSeating, activePid)}
      </div>
      <div class="table-row">
        ${renderContract(auction, obs.phase, obs.dealer_table_position, obs.dealer_player_id, playerNames)}
        <div class="panel sketched-border">
          <div class="panel-title">Auction</div>
          ${
            auction.length
              ? renderAuction(auction, obs.dealer_table_position, activePid)
              : '<div style="text-align:center;color:#444343;font-size:0.85rem;">(no calls yet)</div>'
          }
        </div>
      </div>
      ${renderPlays(plays)}
      <div class="status-container sketched-border ${statusCls}">${renderStatus(obs, activePid, playerNames)}</div>
    </div>
  `;
}
