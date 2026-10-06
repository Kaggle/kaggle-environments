"""The shipper: the external market that wants parcels moved.

Demand follows a logistic trend under a boom / normal / slump regime, with
weekly seasonality, drifting territory weights and daily noise. Price rises
when demand exceeds the field's capacity and falls when it does not. Unsold or
undelivered packages are reposted at a markup until the shipper's patience
runs out, which lowers that territory's demand.

Every parameter is drawn per episode and none is published.
"""

import math

from .constants import (
    DAYS,
    DEMAND_AR1,
    DEMAND_CEILING,
    DEMAND_MIDPOINT,
    DEMAND_START,
    DEMAND_STEEPNESS,
    DEMAND_WEEK,
    INDEX_DOWN,
    INDEX_IDLE_REVERT,
    INDEX_NOISE,
    INDEX_RANGE,
    INDEX_SPILLOVER,
    INDEX_UP,
    LISTING_NOISE,
    LOT_FRACTION,
    MAX_LISTINGS,
    PAIR_DEVIATION,
    PAIR_WEIGHT_DRIFT,
    PAIR_WEIGHT_REVERT,
    PAIR_WEIGHT_SPREAD,
    PATIENCE_MAX,
    PRICE_ELASTICITY,
    PRICE_SPEED,
    PRICE_THROUGHPUT,
    REGIME_STAY,
    REGIMES,
    REPUTATION_HIT,
    RETRY_MARKUP,
    STANDING_DISCOUNT,
    STANDING_SHARE,
    STANDING_TERMS,
)

_REGIME_ORDER = list(REGIMES)


class Shipper:
    """The market's hidden state, advanced once a night."""

    def __init__(self, rng, pairs, capacity):
        self.rng = rng
        self.pairs = list(pairs)
        # The field's starting fleet, not its current one.
        self.capacity = max(1, capacity)

        self.lo = rng.uniform(*DEMAND_START)
        self.hi = rng.uniform(*DEMAND_CEILING)
        self.mid = rng.uniform(*DEMAND_MIDPOINT) * DAYS
        self.steep = rng.uniform(*DEMAND_STEEPNESS)
        week = [w * rng.uniform(0.9, 1.1) for w in DEMAND_WEEK]
        self.week = [w * len(week) / sum(week) for w in week]
        self.regime = "NORMAL"
        self.stay = rng.uniform(*REGIME_STAY)
        self.regime_mult = {name: m * rng.uniform(0.95, 1.05) for name, m in REGIMES.items()}
        self.noise = 0.0

        self.home = {p: rng.gauss(0.0, PAIR_WEIGHT_SPREAD) for p in self.pairs}
        self.weight = dict(self.home)

        self.market = 0.0  # log of the market price level
        self.dev = dict.fromkeys(self.pairs, 0.0)  # log deviation per pair
        self.throughput = rng.uniform(*PRICE_THROUGHPUT)
        self.elasticity = rng.uniform(*PRICE_ELASTICITY)
        self.speed = rng.uniform(*PRICE_SPEED)
        self.eta_up = rng.uniform(*INDEX_UP)
        self.eta_down = rng.uniform(*INDEX_DOWN)
        self.spill = rng.uniform(*INDEX_SPILLOVER)

        self.markup = rng.uniform(*RETRY_MARKUP)
        self.patience_max = rng.randint(*PATIENCE_MAX)
        self.reputation = rng.uniform(*REPUTATION_HIT)

        self.backlog = []  # (lot, packages still to move)
        self.stats = {}  # pair -> tonight's outcome tallies
        self.lost = 0  # packages given up on, all episode
        self.day = -1

    # --- demand ---------------------------------------------------------------

    def _advance(self, day):
        """Step the regime, the noise and the territory weights to `day`."""
        rng = self.rng
        while self.day < day:
            self.day += 1
            if self.day > 0:
                if rng.random() > self.stay:
                    i = _REGIME_ORDER.index(self.regime)
                    step = rng.choice([-1, 1])
                    self.regime = _REGIME_ORDER[max(0, min(len(_REGIME_ORDER) - 1, i + step))]
                rho, sigma = DEMAND_AR1
                self.noise = rho * self.noise + rng.gauss(0.0, sigma)
                for p in self.pairs:
                    pull = PAIR_WEIGHT_REVERT * (self.home[p] - self.weight[p])
                    self.weight[p] += pull + rng.gauss(0.0, PAIR_WEIGHT_DRIFT)

    def demand(self, day):
        """Fresh truck-days the shipper wants moved tomorrow."""
        self._advance(day)
        trend = self.lo + (self.hi - self.lo) / (1.0 + math.exp(-self.steep * (day - self.mid)))
        level = trend * self.regime_mult[self.regime] * self.week[day % 7] * math.exp(self.noise)
        return self.capacity * level

    # --- price ----------------------------------------------------------------

    def index(self, pair):
        lo, hi = INDEX_RANGE
        return max(lo, min(hi, math.exp(self.market + self.dev[pair])))

    def reserve(self, board, pair, fraction, retry=0):
        base = board.reserves[pair] * fraction
        jitter = 1.0 + self.rng.uniform(-LISTING_NOISE, LISTING_NOISE)
        return base * self.index(pair) * (1.0 + self.markup) ** retry * jitter

    def _tally(self, pair):
        return self.stats.setdefault(pair, {"posted": 0, "unserved": 0, "lots": 0, "bids": 0, "clear": []})

    def observe_auction(self, posted, bid_book, awards):
        """What cleared tonight, and how hard carriers fought for it."""
        depth = {row["lot"]: len(row["bids"]) for row in bid_book}
        won = {a["lot"]: a for a in awards}
        for lot in posted:
            pair = (lot["warehouse"], lot["district"])
            t = self._tally(pair)
            t["posted"] += lot["packages"]
            t["lots"] += 1
            t["bids"] += depth.get(lot["id"], 0)
            award = won.get(lot["id"])
            if award is None:
                t["unserved"] += lot["packages"]
                self.backlog.append((lot, lot["packages"]))
            elif lot["reserve"] > 0:
                t["clear"].append(award["ask"] / lot["reserve"])

    def observe_failed(self, lot, packages):
        """Packages a carrier won and did not deliver: the shipper still needs them moved."""
        if packages <= 0:
            return
        pair = (lot["warehouse"], lot["district"])
        self._tally(pair)["unserved"] += packages
        self.backlog.append((lot, packages))

    def _reprice(self, wanted, supply):
        """Set tonight's prices: the market level off demand against the field,
        then each territory's deviation off its own outcomes."""
        rng = self.rng
        ratio = max(wanted, 0.1) / max(0.5, supply * self.throughput)
        target = self.elasticity * math.log(ratio)
        self.market += self.speed * (target - self.market) + rng.gauss(0.0, INDEX_NOISE)

        posted = sum(t["posted"] for t in self.stats.values())
        market_short = min(1.0, sum(t["unserved"] for t in self.stats.values()) / posted) if posted else 0.0
        for pair in self.pairs:
            t = self.stats.get(pair)
            if not t or not t["posted"]:
                self.dev[pair] *= 1.0 - INDEX_IDLE_REVERT
                continue
            short = min(1.0, t["unserved"] / t["posted"])
            depth = t["bids"] / max(1, t["lots"])
            clear = sum(t["clear"]) / len(t["clear"]) if t["clear"] else 1.0
            # Several bids under reserve lower the deviation, scaled by market slack.
            glut = (1.0 - market_short) * min(1.0, max(0.0, depth - 1.0) / 2.0) * (0.3 + (1.0 - clear))
            self.dev[pair] += self.eta_up * short - self.eta_down * glut + rng.gauss(0.0, INDEX_NOISE)
        for pair in self.pairs:
            v = (1.0 - self.spill) * self.dev[pair]
            self.dev[pair] = max(-PAIR_DEVIATION, min(PAIR_DEVIATION, v))
        self.stats = {}

    # --- the board ------------------------------------------------------------

    def post(self, board, day, committed, supply):
        """Tonight's listings: yesterday's backlog first, then fresh demand.

        `supply` is the field's trucks that can roll, off the public roster.
        The board is laid out before it is priced.
        """
        rng = self.rng
        fresh = max(0.0, self.demand(day) - committed)
        plan = []  # (pair, packages, fraction, retry, patience)

        for lot, left in self.backlog:
            pair = (lot["warehouse"], lot["district"])
            retry = lot.get("retry", 0) + 1
            patience = lot.get("_patience", self.patience_max)
            if retry > patience or len(plan) >= MAX_LISTINGS:
                self.lost += left
                self.weight[pair] += math.log(1.0 - self.reputation)
                continue
            plan.append((pair, left, lot["truck_days"] * left / max(1, lot["packages"]), retry, patience))
        self.backlog = []

        # Tonight's fresh freight lands on a handful of territories, weighted
        # by where the shipper's demand currently sits.
        count = max(2, min(len(self.pairs), int(fresh / 1.5 + 0.5)))
        live = self._draw_pairs(count)
        weights = [math.exp(self.weight[p]) for p in live]
        posted = 0.0
        while posted < fresh and len(plan) < MAX_LISTINGS:
            pair = rng.choices(live, weights=weights)[0]
            fraction = rng.uniform(*LOT_FRACTION)
            packages = max(1, int(round(board.truck_days[pair]["packages"] * fraction)))
            plan.append((pair, packages, fraction, 0, rng.randint(1, self.patience_max)))
            posted += fraction

        # The market level prices fresh demand only; retries carry their own
        # markup and deviation.
        self._reprice(posted + committed, supply)

        listings, accounts = [], []
        fresh_lots = 0
        for pair, packages, fraction, retry, patience in plan:
            lot = board._listing(pair[0], pair[1], packages, fraction, self.reserve(board, pair, fraction, retry), rng)  # noqa: SLF001
            lot["retry"] = retry
            lot["_patience"] = patience
            # Only fresh freight posts as a standing account, a share of it.
            fresh_lots += 0 if retry else 1
            if not retry and len(accounts) < int(STANDING_SHARE * fresh_lots + 0.5):
                lot["id"] = f"acct_{board._next_account}"  # noqa: SLF001
                board._next_account += 1  # noqa: SLF001
                lot["kind"] = "STANDING"
                lot["term_options"] = list(STANDING_TERMS)
                lot["reserve"] = round(lot["reserve"] * (1.0 - rng.uniform(*STANDING_DISCOUNT)), 2)
                lot["payout_per_package"] = round(lot["reserve"] / lot["packages"], 2)
                accounts.append(lot)
            else:
                listings.append(lot)
        return listings, accounts

    def _draw_pairs(self, count):
        """`count` distinct pairs, each drawn in proportion to its weight."""
        pool = list(self.pairs)
        out = []
        while pool and len(out) < count:
            pick = self.rng.choices(pool, weights=[math.exp(self.weight[p]) for p in pool])[0]
            pool.remove(pick)
            out.append(pick)
        return out
