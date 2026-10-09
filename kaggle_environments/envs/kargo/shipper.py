"""The shipper: the external market that wants parcels and bulk freight moved.

Parcel demand follows a logistic trend under a boom / normal / slump regime,
with weekly seasonality, drifting territory weights and daily noise. Price
rises when demand exceeds the field's capacity and falls when it does not.
Unsold or undelivered packages are reposted at a markup until the shipper's
patience runs out, which lowers that territory's demand.

Bulk freight runs the same loop on its own demand path and index, priced
against the field's `BOX` trucks.

Every parameter is drawn per episode and none is published.
"""

import math
import random

from .constants import (
    BULK_DEMAND_AR1,
    BULK_DEMAND_CEILING,
    BULK_DEMAND_MIDPOINT,
    BULK_DEMAND_START,
    BULK_DEMAND_STEEPNESS,
    BULK_DEMAND_WEEK,
    BULK_DISTRICTS,
    BULK_ELASTICITY,
    BULK_INDEX_RANGE,
    BULK_INDEX_UP,
    BULK_MAX_LISTINGS,
    BULK_PAIR_WEIGHT_SPREAD,
    BULK_PATIENCE,
    BULK_RETRY_MARKUP,
    BULK_SPEED,
    BULK_STOPS,
    BULK_THROUGHPUT,
    BULK_UNITS,
    BULK_UNITS_PER_PACKAGE,
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


class _Index:
    """A price index: a market level off demand against supply, and a deviation
    per territory off its own outcomes."""

    index_range = INDEX_RANGE

    def index(self, pair):
        lo, hi = self.index_range
        return max(lo, min(hi, math.exp(self.market + self.dev[pair])))

    def _stream(self, *key):
        """A draw stream fixed by the episode and `key`, so no other draw can shift it."""
        return random.Random(":".join(map(str, (self.key, *key))))

    def _tally(self, pair):
        return self.stats.setdefault(pair, {"posted": 0, "unserved": 0, "lots": 0, "bids": 0, "clear": []})

    def _observe(self, lot, award, depth):
        pair = (lot["warehouse"], lot["district"])
        t = self._tally(pair)
        t["posted"] += lot["packages"]
        t["lots"] += 1
        t["bids"] += depth
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

    def _reprice(self, wanted, supply, rng):
        """Set tonight's prices: the market level off demand against the field,
        then each territory's deviation off its own outcomes."""
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


class Shipper(_Index):
    """The market's hidden state, advanced once a night."""

    def __init__(self, rng, pairs, capacity, bulk_capacity=0):
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
        self.key = rng.getrandbits(64)
        bulk_pairs = [p for p in self.pairs if p[1] in BULK_DISTRICTS]
        self.bulk = BulkMarket(random.Random(rng.getrandbits(64)), bulk_pairs, bulk_capacity)

        self.backlog = []  # (lot, packages still to move)
        self.stats = {}  # pair -> tonight's outcome tallies
        self.lost = 0  # packages given up on, all episode
        self.day = -1

    # --- demand ---------------------------------------------------------------

    def _advance(self, day):
        """Step the regime, the noise and the territory weights to `day`."""
        while self.day < day:
            self.day += 1
            rng = self._stream("demand", self.day)
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

    def reserve(self, board, pair, fraction, rng, retry=0):
        base = board.reserves[pair] * fraction
        jitter = 1.0 + rng.uniform(-LISTING_NOISE, LISTING_NOISE)
        return base * self.index(pair) * (1.0 + self.markup) ** retry * jitter

    def observe_auction(self, posted, bid_book, awards):
        """What cleared tonight, and how hard carriers fought for it."""
        depth = {row["lot"]: len(row["bids"]) for row in bid_book}
        won = {a["lot"]: a for a in awards}
        for lot in posted:
            market = self.bulk if lot.get("bulk") else self
            market._observe(lot, won.get(lot["id"]), depth.get(lot["id"], 0))  # noqa: SLF001

    def observe_failed(self, lot, packages):
        """Packages a carrier won and did not deliver: the shipper still needs them moved."""
        if lot.get("bulk"):
            self.bulk.observe_failed(lot, packages)
        else:
            super().observe_failed(lot, packages)

    # --- the board ------------------------------------------------------------

    def post(self, board, day, committed, supply, bulk_supply=0):
        """Tonight's listings: yesterday's backlog first, then fresh demand.

        `supply` is the field's trucks that can roll, off the public roster;
        `bulk_supply` its `BOX` trucks. Parcels are priced against the trucks
        left after tonight's bulk board takes its `BOX` truck-days. The board is
        laid out before it is priced.
        """
        bulk = self.bulk.post(board, day, bulk_supply)
        supply -= min(bulk_supply, sum(lot["truck_days"] for lot in bulk))
        fresh = max(0.0, self.demand(day) - committed)
        plan = []  # (pair, packages, fraction, retry, patience, key)

        for lot, left in self.backlog:
            pair = (lot["warehouse"], lot["district"])
            retry = lot.get("retry", 0) + 1
            patience = lot.get("_patience", self.patience_max)
            if retry > patience or len(plan) >= MAX_LISTINGS:
                self.lost += left
                self.weight[pair] += math.log(1.0 - self.reputation)
                continue
            key = f"{lot.get('_key', lot['id'])}/r{retry}"
            plan.append((pair, left, lot["truck_days"] * left / max(1, lot["packages"]), retry, patience, key))
        self.backlog = []

        # Tonight's fresh freight lands on a handful of territories, weighted
        # by where the shipper's demand currently sits.
        rng = self._stream("fresh", day)
        count = max(2, min(len(self.pairs), int(fresh / 1.5 + 0.5)))
        live = self._draw_pairs(count, rng)
        weights = [math.exp(self.weight[p]) for p in live]
        posted = 0.0
        n_fresh = 0
        while posted < fresh and len(plan) < MAX_LISTINGS:
            pair = rng.choices(live, weights=weights)[0]
            fraction = rng.uniform(*LOT_FRACTION)
            packages = max(1, int(round(board.truck_days[pair]["packages"] * fraction)))
            plan.append((pair, packages, fraction, 0, rng.randint(1, self.patience_max), f"d{day}f{n_fresh}"))
            posted += fraction
            n_fresh += 1

        # The market level prices fresh demand only; retries carry their own
        # markup and deviation.
        self._reprice(posted + committed, supply, self._stream("reprice", day))

        listings, accounts = [], []
        fresh_lots = 0
        for pair, packages, fraction, retry, patience, key in plan:
            rng = self._stream("lot", key)
            reserve = self.reserve(board, pair, fraction, rng, retry)
            lot = board._listing(pair[0], pair[1], packages, fraction, reserve, rng)  # noqa: SLF001
            lot["retry"] = retry
            lot["_patience"] = patience
            lot["_key"] = key
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
        return listings + bulk, accounts

    def _draw_pairs(self, count, rng):
        """`count` distinct pairs, each drawn in proportion to its weight."""
        pool = list(self.pairs)
        out = []
        while pool and len(out) < count:
            pick = rng.choices(pool, weights=[math.exp(self.weight[p]) for p in pool])[0]
            pool.remove(pick)
            out.append(pick)
        return out


class BulkMarket(_Index):
    """Palletised freight only a `BOX` holds: its own demand path and index.

    Demand is sized off the field's starting `BOX` trucks. Unserved lots come
    back for longer and at a steeper markup than parcels, and each territory's
    deviation reacts faster to its unserved share. Bulk is spot only.
    """

    index_range = BULK_INDEX_RANGE

    def __init__(self, rng, pairs, capacity):
        self.pairs = list(pairs)
        self.capacity = max(1, capacity)
        self.lo = rng.uniform(*BULK_DEMAND_START)
        self.hi = rng.uniform(*BULK_DEMAND_CEILING)
        self.mid = rng.uniform(*BULK_DEMAND_MIDPOINT) * DAYS
        self.steep = rng.uniform(*BULK_DEMAND_STEEPNESS)
        week = [w * rng.uniform(0.9, 1.1) for w in BULK_DEMAND_WEEK]
        self.week = [w * len(week) / sum(week) for w in week]
        self.noise = 0.0
        self.weight = {p: math.log(BULK_DISTRICTS[p[1]]) + rng.gauss(0.0, BULK_PAIR_WEIGHT_SPREAD) for p in self.pairs}

        self.market = 0.0
        self.dev = dict.fromkeys(self.pairs, 0.0)
        self.throughput = rng.uniform(*BULK_THROUGHPUT)
        self.elasticity = rng.uniform(*BULK_ELASTICITY)
        self.speed = rng.uniform(*BULK_SPEED)
        self.eta_up = rng.uniform(*BULK_INDEX_UP)
        self.eta_down = rng.uniform(*INDEX_DOWN)
        self.spill = rng.uniform(*INDEX_SPILLOVER)
        self.markup = rng.uniform(*BULK_RETRY_MARKUP)
        self.reputation = rng.uniform(*REPUTATION_HIT)
        self.key = rng.getrandbits(64)

        self.backlog = []
        self.stats = {}
        self.lost = 0
        self.day = -1

    def _advance(self, day):
        while self.day < day:
            self.day += 1
            if self.day > 0:
                rho, sigma = BULK_DEMAND_AR1
                self.noise = rho * self.noise + self._stream("demand", self.day).gauss(0.0, sigma)

    def demand(self, day):
        """Fresh `BOX` truck-days of bulk the shipper wants moved tomorrow."""
        self._advance(day)
        trend = self.lo + (self.hi - self.lo) / (1.0 + math.exp(-self.steep * (day - self.mid)))
        return self.capacity * trend * self.week[day % 7] * math.exp(self.noise)

    def post(self, board, day, supply):
        """Tonight's bulk listings: the backlog first, then fresh demand."""
        fresh = self.demand(day)
        plan = []  # (pair, packages, stops, retry, patience, key)
        for lot, left in self.backlog:
            pair = (lot["warehouse"], lot["district"])
            retry = lot.get("retry", 0) + 1
            patience = lot.get("_patience", BULK_PATIENCE[1])
            if retry > patience or len(plan) >= BULK_MAX_LISTINGS:
                self.lost += left
                self.weight[pair] += math.log(1.0 - self.reputation)
                continue
            key = f"{lot.get('_key', lot['id'])}/r{retry}"
            plan.append((pair, left, min(lot["stops"], left), retry, patience, key))
        self.backlog = []

        rng = self._stream("fresh", day)
        weights = [math.exp(self.weight[p]) for p in self.pairs]
        lo, hi = (int(round(u / BULK_UNITS_PER_PACKAGE)) for u in BULK_UNITS)
        posted, n_fresh = 0.0, 0
        while posted < fresh and len(plan) < BULK_MAX_LISTINGS:
            pair = rng.choices(self.pairs, weights=weights)[0]
            packages = rng.randint(lo, hi)
            stops = rng.randint(*BULK_STOPS)
            plan.append((pair, packages, stops, 0, rng.randint(*BULK_PATIENCE), f"d{day}b{n_fresh}"))
            posted += board.bulk_terms(pair[0], pair[1], packages, stops)[0]
            n_fresh += 1

        self._reprice(posted, supply, self._stream("reprice", day))

        listings = []
        for pair, packages, stops, retry, patience, key in plan:
            rng = self._stream("lot", key)
            base = board.bulk_terms(pair[0], pair[1], packages, stops)[2]
            jitter = 1.0 + rng.uniform(-LISTING_NOISE, LISTING_NOISE)
            reserve = base * self.index(pair) * (1.0 + self.markup) ** retry * jitter
            lot = board._bulk_listing(pair[0], pair[1], packages, stops, reserve, rng)  # noqa: SLF001
            lot["retry"] = retry
            lot["_patience"] = patience
            lot["_key"] = key
            listings.append(lot)
        return listings
