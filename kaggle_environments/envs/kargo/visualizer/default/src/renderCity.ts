/** The city: one SVG for the network, HTML sprites on top of it.
 *
 * Split that way on purpose. The road network is 760 edges whose only per-step
 * change is a stroke colour, so it is built once as SVG and recoloured in
 * place. The sprites move every step and need the CSS status dots, which are
 * pseudo-elements -- those do not exist in SVG.
 */

import { esc, flankMarkup, topbarMarkup } from './chrome';
import { CLOSED_STROKE, CONGESTION, CONGESTION_ORDER, DISTRICT_GROUND, PLAYER_INK, ROAD_CASING } from './theme';
import type { Crumb, StopState, Truck, View } from './types';
import { clockLabel, money, spriteSrc, truckSprite } from './utils';

/** viewBox units per grid cell. Arbitrary; keeps stroke widths whole-ish. */
const CELL = 50;

/** `EDGE_KM` in constants.py: one grid cell is 1.5 km of arterial. */
const KM_PER_CELL = 1.5;

const ROAD_WIDTH: Record<string, number> = { ARTERIAL: 7, COLLECTOR: 4.5, LOCAL: 2.5 };

export interface CityRefs {
  size: number;
  roads: SVGPathElement[];
  closures: SVGGElement;
  sprites: HTMLElement;
  feed: HTMLElement;
}

function nodeXY(node: number, size: number): [number, number] {
  const r = Math.floor(node / size);
  const c = node % size;
  return [(c + 0.5) * CELL, (r + 0.5) * CELL];
}

/** Percent-of-map coordinates, for the HTML sprite layer. */
function nodePct(node: number, size: number, dx = 0, dy = 0): [number, number] {
  const [x, y] = nodeXY(node, size);
  return [((x + dx) / (size * CELL)) * 100, ((y + dy) / (size * CELL)) * 100];
}

function ground(city: View['city']): string {
  const { size, node_district: nd } = city;
  const rects: string[] = [];
  for (let i = 0; i < size * size; i++) {
    const r = Math.floor(i / size);
    const c = i % size;
    const fill = DISTRICT_GROUND[nd[i]] ?? '#d0ccc4';
    rects.push(`<rect x="${c * CELL}" y="${r * CELL}" width="${CELL}" height="${CELL}" fill="${fill}" />`);
  }
  return `<g class="ground">${rects.join('')}</g>`;
}

function roads(city: View['city']): string {
  const casings: string[] = [];
  const surfaces: string[] = [];
  city.edges.forEach(([u, v, klass], i) => {
    const [x1, y1] = nodeXY(u, city.size);
    const [x2, y2] = nodeXY(v, city.size);
    const w = ROAD_WIDTH[klass] ?? 2.5;
    const d = `M${x1} ${y1}L${x2} ${y2}`;
    // The casing is what makes a road legible, not its fill: the ramp's pale
    // end is only 1.04:1 against the district ground it sits on, but the
    // casing is 8.7:1 against the lightest ground. +3.4 units, not +2, because
    // the whole 1000-unit board renders into ~670 px -- a 1-unit casing is
    // 0.67 px per side and disappears.
    casings.push(`<path d="${d}" stroke="${ROAD_CASING}" stroke-width="${w + 3.4}" stroke-linecap="round" />`);
    surfaces.push(`<path class="road" data-edge="${i}" d="${d}" stroke-width="${w}" stroke-linecap="round" />`);
  });
  return `<g class="casings" fill="none">${casings.join('')}</g><g class="surfaces" fill="none">${surfaces.join('')}</g>`;
}

function congestionLegend(): string {
  // Direct labels are not optional here: every step below HEAVY is under 3:1
  // against the surface, and the validator's contrast WARN obliges relief.
  const swatches = CONGESTION_ORDER.map(
    (lvl) =>
      `<span class="ramp-step"><i style="background:${CONGESTION[lvl]}"></i>${esc(lvl[0] + lvl.slice(1).toLowerCase())}</span>`
  ).join('');
  const states: [StopState, string][] = [
    ['delivered', 'Delivered'],
    ['late', 'Late'],
    ['refused', 'Refused'],
    ['failed', 'Missed'],
    ['abandoned', 'Abandoned'],
    ['pending', 'Pending'],
  ];
  const dots = states
    .map(([k, label]) => `<span class="key-item"><span class="dot dot-${k} dot-key"></span>${label}</span>`)
    .join('');
  return `
    <div class="legend">
      <div class="legend-row">
        <span class="legend-title">Congestion</span>
        <span class="ramp">${swatches}</span>
      </div>
      <div class="legend-row">
        <span class="legend-title">Stop outcome</span>
        <span class="key">${dots}</span>
        <span class="key-item"><span class="closed-swatch"></span>Road closed</span>
      </div>
    </div>`;
}

export function buildShell(parent: HTMLElement, city: View['city'], names: string[]): void {
  const side = city.size * CELL;
  parent.innerHTML = `
    <div class="kargo">
      ${topbarMarkup()}
      <div class="main">
        ${flankMarkup(names, 'left')}
        <div class="map-wrap">
          <div class="map">
            <svg viewBox="0 0 ${side} ${side}" preserveAspectRatio="none" aria-label="City road network">
              ${ground(city)}
              ${roads(city)}
              <g class="closures" data-closures fill="none"></g>
            </svg>
            <div class="sprite-layer" data-sprites></div>
          </div>
        </div>
        ${flankMarkup(names, 'right')}
      </div>
      ${congestionLegend()}
      <div class="feed" data-feed></div>
    </div>`;
}

export function collectRefs(parent: HTMLElement, size: number): CityRefs {
  const q = <T extends Element>(sel: string) => parent.querySelector(sel) as T;
  return {
    size,
    roads: Array.from(parent.querySelectorAll<SVGPathElement>('path.road')),
    closures: q('[data-closures]'),
    sprites: q('[data-sprites]'),
    feed: q('[data-feed]'),
  };
}

/** Worst outcome wins at a node, same rule as buildView uses per segment. */
const SEVERITY: Record<StopState, number> = {
  pending: 0,
  delivered: 1,
  late: 2,
  abandoned: 3,
  failed: 4,
  refused: 5,
};

interface NodeStop {
  node: number;
  state: StopState;
  faces: number;
  /** Faces with parcels still on a truck. */
  open: number;
  parcels: number;
  counts: Partial<Record<StopState, number>>;
}

/** One marker per node, not per segment.
 *
 * Measured on a real replay: 93 segments landed on 3 nodes, 54 of them on one.
 * That is the engine being literal -- a lot's segments are scattered within a
 * kilometre of its anchor -- but 54 kerb sprites inside a 33 px cell is a
 * smear, so they collapse to the block they share. The per-segment `pos`
 * offset goes with them: sub-cell placement is meaningless once the marker
 * stands for the whole block.
 *
 * Openness is READ, not inferred. A segment is open while `pending` still
 * lists addresses on a truck. Inferring it from the event stream was wrong:
 * one DELIVER marks the segment, but a segment has several doors, so a block
 * reported "0 open, 2 parcels" -- resolved and unresolved at the same time.
 *
 * The summary is worst-once-settled, not worst-wins. With 54 faces behind one
 * dot, a single late door repainting all 54 is a lie by aggregation; the dot
 * shows progress while work remains and a verdict after, and the tooltip
 * breaks the verdict down by outcome.
 */
function nodeStops(view: View, pid: number): NodeStop[] {
  const by = new Map<number, NodeStop>();
  for (const seg of view.segments[pid] ?? []) {
    const parcels = seg.pending?.length ?? 0;
    let cur = by.get(seg.node);
    if (!cur) {
      cur = { node: seg.node, state: 'delivered', faces: 0, open: 0, parcels: 0, counts: {} };
      by.set(seg.node, cur);
    }
    cur.faces += 1;
    cur.parcels += parcels;
    if (parcels > 0) {
      cur.open += 1;
      continue;
    }
    const state = view.stops[pid]?.get(seg.id) ?? 'delivered';
    cur.counts[state] = (cur.counts[state] ?? 0) + 1;
    if (SEVERITY[state] > SEVERITY[cur.state]) cur.state = state;
  }
  for (const s of by.values()) if (s.open > 0) s.state = 'pending';
  return [...by.values()];
}

function stopMarkup(view: View, pid: number): string {
  return nodeStops(view, pid)
    .map((s) => {
      const [x, y] = nodePct(s.node, view.city.size);
      const who = view.names[pid] ?? `P${pid + 1}`;
      const breakdown = Object.entries(s.counts)
        .sort((a, b) => SEVERITY[b[0] as StopState] - SEVERITY[a[0] as StopState])
        .map(([k, v]) => `${v} ${k}`)
        .join(', ');
      const head = `${who} -- ${s.faces} block face${s.faces > 1 ? 's' : ''}`;
      const body = s.open > 0 ? `${s.open} open, ${s.parcels} parcels left` : breakdown || s.state;
      const count = s.open > 0 ? `<span class="face-count">${s.open}</span>` : '';
      // The seat tint is a hairline under the kerb, never the outcome colour:
      // whose stop it is and how it went are two facts and get two encodings.
      return `<span class="stop marker" data-seat="${pid}" style="left:${x.toFixed(2)}%;top:${y.toFixed(2)}%;--seat:${PLAYER_INK[pid]}" title="${esc(`${head} -- ${body}`)}">
        <img src="${spriteSrc('stop_kerb')}" alt="" />
        <span class="dot dot-${s.state}"></span>${count}
      </span>`;
    })
    .join('');
}

/** Where a crumb actually is, in percent of the map.
 *
 * The node fixes the intersection; `off` walks in from it. A lot's segments sit
 * within a kilometre or so of their anchor, which is a fraction of a cell, so
 * the offset is what separates one kerb from the next -- without it every stop
 * in a territory renders on the same pixel.
 */
function crumbPct(c: Crumb, size: number, nudge: [number, number]): [number, number] {
  const dx = c.off ? (c.off[0] / KM_PER_CELL) * CELL : 0;
  const dy = c.off ? (c.off[1] / KM_PER_CELL) * CELL : 0;
  return nodePct(c.node, size, dx + nudge[0], dy + nudge[1] - CELL * 0.18);
}

/** Fleets are nudged off the intersection so seats parked on one node do not
 * hide each other.
 *
 * Four seats need two axes: at a four-player table the whole fleet can converge
 * on one warehouse, and offsetting along x alone would leave seat 3 sitting on
 * seat 1. Each seat takes its own corner. The offset is a function of seat id
 * only, so it never moves between steps -- a marker that drifts because the
 * neighbouring seat went bankrupt would read as the truck relocating.
 */
const SEAT_NUDGE: [number, number][] = [
  [-0.22, -0.1],
  [0.22, -0.1],
  [-0.22, 0.12],
  [0.22, 0.12],
];

function seatNudge(pid: number): [number, number] {
  const [x, y] = SEAT_NUDGE[pid] ?? [0, 0];
  return [x * CELL, y * CELL];
}

function truckMarkup(view: View, pid: number): string {
  const nudge = seatNudge(pid);
  return (view.trucks[pid] ?? [])
    .filter((t) => t.status === 'ACTIVE' || t.status === 'IDLE' || t.status === 'DISABLED')
    .map((t) => {
      // The marker is parked at the END of the block, because that is the
      // position every other panel agrees on and the one it must hold once the
      // animation finishes or is skipped. The trail animates it away from here
      // and back; a replay without a trail simply never moves.
      const last = t.trail?.length ? t.trail[t.trail.length - 1] : null;
      const [x, y] = last
        ? crumbPct(last, view.city.size, nudge)
        : nodePct(t.node, view.city.size, nudge[0], nudge[1] - CELL * 0.18);
      const broken = t.status === 'DISABLED';
      const src = broken ? spriteSrc('truck_disabled') : truckSprite(t.type, pid);
      const title = `${t.id} (${t.type}) ${t.status} -- ${t.carrying?.length ?? 0} parcels, ${clockLabel(t.clock)}`;
      // Keyed by truck id, not fleet index: this list is filtered by status and
      // `animateTrails` walks the unfiltered fleet, so one truck in the shop
      // would shift every index after it and drive the wrong sprite.
      return `<span class="truck marker" data-truck="${pid}:${esc(t.id)}" style="left:${x.toFixed(2)}%;top:${y.toFixed(2)}%" title="${esc(title)}">
        <img src="${esc(src)}" alt="" />
      </span>`;
    })
    .join('');
}

/** Drive the block's trail as one keyframe animation per truck.
 *
 * Keyframes rather than a per-frame rAF loop: the browser owns the timing, a
 * scrub mid-flight cancels cleanly, and an offscreen tab does not accumulate
 * a backlog of positions to catch up on.
 *
 * Offsets are keyed to SIMULATED time, not spaced evenly -- a truck that spends
 * ninety minutes at one kerb and six crossing town should look like it. The
 * sprite is positioned by `left`/`top` and animated by `translate` so the two
 * never fight; the block's start is expressed as a delta back from the parked
 * end position, which is what `renderObservation` already wrote.
 */
function animateTrails(refs: CityRefs, view: View, durationMs: number): void {
  for (const a of refs.sprites.getAnimations?.({ subtree: true }) ?? []) a.cancel();
  if (durationMs <= 0) return;

  view.trucks.forEach((fleet: Truck[], pid: number) => {
    const nudge = seatNudge(pid);
    fleet.forEach((t) => {
      const trail = t.trail;
      if (!trail || trail.length < 2) return;
      const el = refs.sprites.querySelector<HTMLElement>(`[data-truck="${pid}:${t.id.replace(/["\\]/g, '\\$&')}"]`);
      if (!el) return;

      const pts = trail.map((c) => crumbPct(c, view.city.size, nudge));
      const [endX, endY] = pts[pts.length - 1];
      const t0 = trail[0].t;
      const span = trail[trail.length - 1].t - t0;
      if (!(span > 0)) return;

      // Percent on `translate` resolves against the ELEMENT's own box, not the
      // map's, so the delta is converted to map-percent via the sprite layer's
      // pixel size. Falling back to no animation beats animating in the wrong
      // units on a zero-width layer.
      const w = refs.sprites.clientWidth;
      const h = refs.sprites.clientHeight;
      if (!w || !h) return;

      const keyframes = pts.map(([x, y], k) => ({
        offset: Math.min(1, Math.max(0, (trail[k].t - t0) / span)),
        translate: `${(((x - endX) / 100) * w).toFixed(2)}px ${(((y - endY) / 100) * h).toFixed(2)}px`,
      }));
      // Linear between crumbs, and the dwell at a kerb comes from the data:
      // the engine stamps both arrival and departure, so a long service window
      // is two keyframes at one position and the truck genuinely sits still.
      el.animate(keyframes, { duration: durationMs, easing: 'linear', fill: 'none' });
    });
  });
}

function warehouseMarkup(view: View): string {
  // A warehouse is "active" when anyone is staged there this block.
  const staged = new Set<string>();
  for (const fleet of view.trucks) for (const t of fleet) if (t.staged) staged.add(t.staged);
  return Object.entries(view.city.warehouses)
    .map(([wid, node]) => {
      const [x, y] = nodePct(node, view.city.size);
      const src = spriteSrc(staged.has(wid) ? 'warehouse_active' : 'warehouse');
      return `<span class="warehouse marker" style="left:${x.toFixed(2)}%;top:${y.toFixed(2)}%" title="${esc(wid)}">
        <img src="${src}" alt="${esc(wid)}" />
      </span>`;
    })
    .join('');
}

function edgeIndex(id: string): number {
  return Number(String(id).replace(/^e_/, ''));
}

function incidentMarkup(view: View): string {
  return view.traffic.incidents
    .map((inc) => {
      const edge = view.city.edges[edgeIndex(inc.edge)];
      if (!edge) return '';
      const [x1, y1] = nodeXY(edge[0], view.city.size);
      const [x2, y2] = nodeXY(edge[1], view.city.size);
      const side = view.city.size * CELL;
      const x = ((x1 + x2) / 2 / side) * 100;
      const y = ((y1 + y2) / 2 / side) * 100;
      const sprite =
        inc.kind === 'ACCIDENT'
          ? 'incident_accident'
          : inc.kind === 'CONSTRUCTION'
            ? 'incident_construction'
            : 'incident_closure';
      const until = inc.until == null ? 'all day' : `until ${clockLabel(inc.until)}`;
      return `<span class="incident marker" style="left:${x.toFixed(2)}%;top:${y.toFixed(2)}%" title="${esc(`${inc.kind} -- ${until}`)}">
        <img src="${spriteSrc(sprite)}" alt="${esc(inc.kind)}" />
      </span>`;
    })
    .join('');
}

/** Closed roads get their own overstroke: a state, not a point on the ramp. */
function paintClosures(refs: CityRefs, view: View): void {
  const parts: string[] = [];
  for (const inc of view.traffic.incidents) {
    if (inc.kind === 'ACCIDENT') continue; // slows the road, does not shut it
    const edge = view.city.edges[edgeIndex(inc.edge)];
    if (!edge) continue;
    const [x1, y1] = nodeXY(edge[0], view.city.size);
    const [x2, y2] = nodeXY(edge[1], view.city.size);
    const w = (ROAD_WIDTH[edge[2]] ?? 2.5) + 0.5;
    parts.push(
      `<path d="M${x1} ${y1}L${x2} ${y2}" stroke="${CLOSED_STROKE}" stroke-width="${w}" stroke-dasharray="5 4" stroke-linecap="butt" />`
    );
  }
  refs.closures.innerHTML = parts.join('');
}

function paintRoads(refs: CityRefs, view: View): void {
  const levels = view.traffic.congestion ?? [];
  refs.roads.forEach((path, i) => {
    path.setAttribute('fill-opacity', '1');
    path.style.stroke = CONGESTION[levels[i]] ?? CONGESTION.FREE;
  });
}

const FEED_DOT: Record<string, StopState> = {
  REFUSED: 'refused',
  UNDELIVERED: 'failed',
  ABANDONED: 'abandoned',
};

function paintFeed(refs: CityRefs, view: View): void {
  // Deliveries are the overwhelming majority (11,672 against 406 in a 30-day
  // run), so listing them would bury every event that costs money. The feed
  // shows the exceptions and counts the rest.
  const delivered = view.events.filter((e) => e.kind === 'DELIVER');
  const loaded = view.events.filter((e) => e.kind === 'LOADED');
  const notable = view.events.filter((e) => e.kind !== 'DELIVER' && e.kind !== 'LOADED');
  const lateCount = delivered.filter((e) => e.late).length;

  const summary =
    (loaded.length > 0
      ? `<span class="feed-item"><span class="dot dot-pending dot-key"></span>${loaded.length} lot${
          loaded.length === 1 ? '' : 's'
        } loaded</span>`
      : '') +
    (delivered.length > 0
      ? `<span class="feed-item"><span class="dot dot-delivered dot-key"></span>${delivered.length} delivered${
          lateCount ? `, ${lateCount} late` : ''
        }</span>`
      : '');
  const items = notable
    .slice(0, 12)
    .map((e) => {
      const state = FEED_DOT[e.kind];
      const chip = state
        ? `<span class="dot dot-${state} dot-key"></span>`
        : '<span class="dot dot-failed dot-key"></span>';
      const who = view.names[e.player] ?? `P${e.player + 1}`;
      const what =
        e.kind === 'SERVICE_DUE'
          ? `${e.truck} due for service`
          : e.kind === 'RAN_DRY'
            ? `${e.truck} out of fuel`
            : e.kind === 'LOAD_REFUSED'
              ? `${e.truck} could not load ${e.lot ?? ''} (${(e.reason ?? '').toLowerCase().replace('_', ' ')})`
              : `${e.packages ?? 0} pkg ${e.kind.toLowerCase()}${e.truck ? '' : ' at the dock'} (${money(e.cost ?? 0)})`;
      return `<span class="feed-item">${chip}<b style="color:${PLAYER_INK[e.player]}">${esc(who)}</b> ${esc(what)} <em>${clockLabel(e.minute)}</em></span>`;
    })
    .join('');
  const more = notable.length > 12 ? `<span class="feed-item feed-more">+${notable.length - 12} more</span>` : '';
  refs.feed.innerHTML = summary + items + more || '<span class="feed-item feed-quiet">No events this block</span>';
}

export function renderObservation(refs: CityRefs, view: View, trailMs = 0): void {
  paintRoads(refs, view);
  paintClosures(refs, view);
  // Seats in order, kerbs under incidents under trucks: the truck is the thing
  // being watched and must never end up behind a stop it is serving.
  const seats = view.names.map((_n, pid) => pid);
  refs.sprites.innerHTML =
    warehouseMarkup(view) +
    seats.map((pid) => stopMarkup(view, pid)).join('') +
    incidentMarkup(view) +
    seats.map((pid) => truckMarkup(view, pid)).join('');
  animateTrails(refs, view, trailMs);
  paintFeed(refs, view);
}
