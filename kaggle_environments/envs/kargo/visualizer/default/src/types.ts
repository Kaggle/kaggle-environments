/** The slice of the kargo observation the visualizer actually reads.
 *
 * Deliberately partial. The observation is large (a 33 MB replay for 30 days)
 * and most of it -- market boards, driver resumes, bid books -- belongs to the
 * agent, not the picture.
 */

export type Phase = 'CAPEX' | 'LABOR' | 'CONTRACTS' | 'DRIVING';
export type Weather = 'CLEAR' | 'RAIN' | 'SNOW';
export type RoadClass = 'ARTERIAL' | 'COLLECTOR' | 'LOCAL';
export type VehicleType = 'VAN' | 'BOX';
export type TruckStatus = 'IDLE' | 'ACTIVE' | 'DISABLED' | 'ORDERED';

/** The six outcomes a block face can be in. Rendered as a CSS dot, not art. */
export type StopState = 'pending' | 'delivered' | 'late' | 'refused' | 'failed' | 'abandoned';

/** An edge is [u, v, class]. Free-flow time and capacity stay in the engine:
 * `City.static_view` publishes only what a player is allowed to know. */
export type Edge = [number, number, RoadClass];

export interface City {
  size: number;
  edges: Edge[];
  node_district: string[];
  warehouses: Record<string, number>;
}

export interface Traffic {
  /** One level per edge, index-aligned with `city.edges`. 0 free .. 5 gridlock. */
  congestion: number[];
  incidents: Incident[];
  weather: Weather;
  forecast: Weather;
}

export interface Incident {
  /** `e_<index>` into `city.edges`, not a bare index. */
  edge: string;
  kind: 'ACCIDENT' | 'CONSTRUCTION' | 'EMERGENCY';
  /** Minutes from 08:00, or null for "shut all day". */
  until: number | null;
}

/** One stamp on a truck's path through the block just run.
 *
 * `off` is the offset in km from `node`, and is null when the truck is on the
 * intersection itself rather than inside a territory hanging off it.
 */
export interface Crumb {
  t: number;
  node: number;
  kind: 'START' | 'DRIVE' | 'STOP' | 'DEPART';
  off: [number, number] | null;
}

export interface Truck {
  id: string;
  type: VehicleType;
  node: number;
  status: TruckStatus;
  clock: number;
  fuel: number;
  km_since_service: number;
  driver: string | null;
  staged: string | null;
  carrying: string[];
  /** Lots on board. */
  lots?: string[];
  route: unknown[];
  /** The block's path, for animation. Absent on replays from before it existed. */
  trail?: Crumb[];
}

export interface Segment {
  id: string;
  node: number;
  /** Offset within the node's block, in km. Lets stops on one node separate. */
  pos: [number, number];
  district: string;
  pending: string[];
}

/** Emitted by the engine per step, per player. See dispatch._event. */
export interface KargoEvent {
  kind: 'DELIVER' | 'REFUSED' | 'UNDELIVERED' | 'ABANDONED' | 'SERVICE_DUE' | 'RAN_DRY' | 'LOADED' | 'LOAD_REFUSED';
  player: number;
  truck: string;
  node: number;
  day: number;
  minute: number;
  address?: string;
  segment?: string;
  packages?: number;
  late?: boolean;
  premium?: boolean;
  cost?: number;
  lot?: string;
  reason?: string;
}

export interface PublicPlayer {
  cash: number;
  debt: number;
  net_worth: number;
  fleet: unknown[];
  drivers: unknown[];
  standing: unknown[];
  results: DayResult[];
}

export interface DayResult {
  day: number;
  delivered: number;
  late: number;
  failed: number;
  refused: number;
  revenue: number;
  cost: number;
  net_worth: number;
}

export interface Driver {
  id: string;
  name: string;
  wage: number;
  resume: number;
  truck: string | null;
  notice: boolean;
}

export interface Private {
  trucks: Truck[];
  drivers: Driver[];
  segments: Segment[];
  events: KargoEvent[];
  day_report: Omit<DayResult, 'day' | 'net_worth'>;
}

/** A lot on the freight board. SPOT is one day's work; STANDING is a term. */
export interface Listing {
  id: string;
  warehouse: string;
  district: string;
  packages: number;
  stops: number;
  truck_days: number;
  parcel_units: number;
  reserve: number;
  kind: 'SPOT' | 'STANDING';
  deadline: number;
  payout_per_package: number;
  dock_packages: number;
  promised_packages: number;
  /** Palletised freight only a BOX holds. */
  bulk?: boolean;
  /** Standing accounts only: the terms, in days, a bid may name. */
  term_options?: number[];
}

export interface Candidate {
  id: string;
  name: string;
  resume: number;
  asking: number;
}

export interface UsedTruck {
  id: string;
  type: VehicleType;
  price: number;
  odometer: number;
  age_days: number;
}

/** Tonight's board. Everything an agent may act on overnight. */
export interface Market {
  listings: Listing[];
  accounts: Listing[];
  used: UsedTruck[];
  /** Units of each type left in the rental pool. */
  rentals: Record<string, number>;
  candidates: Candidate[];
  fill_ceiling: number;
}

/** Who won what, at what ask. `kind` distinguishes a lot from an account. */
export interface AuctionRow {
  lot: string;
  player: number;
  ask: number;
  reserve: number;
  kind: 'SPOT' | 'STANDING';
}

/** Every admissible bid on a lot, cheapest first. Published after the auction. */
export interface BidBookRow {
  lot: string;
  bids: [number, number][];
}

/** The engine's own record of how the night resolved. Shapes vary by op. */
export interface History {
  auction: AuctionRow[];
  capex: Record<string, any>[];
  labor: Record<string, any>[];
  bids: BidBookRow[];
  standing: Record<string, any>[];
}

export interface Observation {
  step: number;
  day: number;
  block: number;
  minute: number;
  phase: Phase;
  player: number;
  city: City;
  traffic: Traffic;
  market: Market;
  history: History;
  public: PublicPlayer[];
  private?: Private;
}

/** What the renderer works from: both players' views of one step, merged. */
export interface View {
  step: number;
  day: number;
  block: number;
  minute: number;
  phase: Phase;
  city: City;
  traffic: Traffic;
  players: PublicPlayer[];
  /** Per player. Index is the player id. */
  trucks: Truck[][];
  drivers: Driver[][];
  segments: Segment[][];
  /** Cumulative outcome per segment id, per player -- see buildView. */
  stops: Map<string, StopState>[];
  /** The live running tally for the day in progress, per player.
   *
   * `public[].results` only gains a row at close of day, so mid-day it has
   * nothing for today and a scoreboard read from it sits at zero until 18:00. */
  dayReports: Private['day_report'][];
  /** This step's events only, both players. */
  events: KargoEvent[];
  names: string[];
  /** Tonight's board, as the agents saw it when they chose. */
  market: Market;
  /** Per player, and only during the overnight phases -- see `night`. */
  night: Night;
}

/** One overnight decision, paired with the offer it answered.
 *
 * The three parts come from three different steps and it matters which:
 *
 *   step N      the OFFER -- `market`, the board the agent was shown
 *   step N+1    the ACTION -- the reply that offer drew, and the OUTCOME,
 *               because the interpreter resolves the action and then publishes
 *
 * A replay records an action on the step it produced, not the step it answered,
 * so reading the action off step N would show the previous phase's decision
 * under this phase's heading -- staging listed under LABOR, hiring under
 * CONTRACTS. Everything here is shifted to undo that.
 */
export interface Night {
  /** Per player, the action that answered this step's board. */
  actions: Record<string, any>[];
  /** How the engine resolved it. Null on the last step: nothing answered yet. */
  outcome: History | null;
  /** True when the replay ends before the reply -- show the board, not a lie. */
  pending: boolean;
}
