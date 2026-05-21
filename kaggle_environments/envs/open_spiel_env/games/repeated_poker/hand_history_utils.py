# Copyright 2025 The game_arena Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ACPC-Pluribus hand history converter.

Initially based on https://github.com/VitamintK/pluribus-hand-parser.

The ACPC Pluribus-style one-line format is:
STATE:<hand_num>:<actions_by_street>:<cards_by_street>:<profits_by_player>:<player_ids>
- actions_by_street: "<street0>/<street1>/<street2>/<street3>"
  * Each street is a compact token string with 'c', 'f', or 'r<digits>'.
- cards_by_street: "P0P1...P{N-1}/flop/turn/river"
  * Each Pk is 2 chars per card.
- profits_by_player: "-50|100|..." (integers in chips)
- player_ids: "A|B|C|D|E|F"

"""
import dataclasses
import datetime
import enum
import re
import textwrap

# ------------------------------------------------------------------------------
# Configuration & core enums
# ------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Config:
  """Configuration for parsing ACPC logs."""
  seats: int
  small_blind: int
  big_blind: int
  starting_stacks: list[int]
  ante: int = 0
  timestamp: datetime.datetime | None = None
  table_name: str = ""


class Street(enum.Enum):
  BLINDS = 0
  PREFLOP = 1
  FLOP = 2
  TURN = 3
  RIVER = 4


class ActionKind(enum.Enum):
  ANTE = enum.auto()
  SB = enum.auto()
  BB = enum.auto()
  BET = enum.auto()
  RAISE = enum.auto()
  CALL = enum.auto()
  CHECK = enum.auto()
  FOLD = enum.auto()


@dataclasses.dataclass(frozen=True)
class Player:
  id: str
  seat: int
  stack_start: int


@dataclasses.dataclass(frozen=True)
class Card:
  text: str  # "Ah", "Td", etc.

  def __post_init__(self):
    if len(self.text) != 2:
      raise ValueError(f"Invalid card: {self.text}")

  def __str__(self):
    return self.text


@dataclasses.dataclass(frozen=True)
class Event:
  """A single event in a hand."""

  street: Street
  actor: int  # player index 0..seats-1
  kind: ActionKind
  to_amount: int | None = None  # total contributed on this street after action
  delta: int | None = None  # marginal paid now
  all_in: bool = False

  def ps_text(self, player_name: str) -> str:
    """Returns a poker site text representation of this event.

    Args:
      player_name: The name of the player who performed the action.
    """
    if self.kind in (
        ActionKind.SB,
        ActionKind.BB,
        ActionKind.BET,
        ActionKind.RAISE,
        ActionKind.CALL,
        ActionKind.ANTE,
    ):
      if self.kind is ActionKind.RAISE:
        assert self.delta is not None and self.to_amount is not None
        line = f"{player_name}: raises {self.delta} to {self.to_amount}"
      elif self.kind is ActionKind.BET:
        assert self.to_amount is not None and self.delta is not None
        # bet's printed amount is the *delta* posted now
        line = f"{player_name}: bets {self.delta}"
      elif self.kind is ActionKind.CALL:
        assert self.delta is not None
        line = f"{player_name}: calls {self.delta}"
      elif self.kind is ActionKind.SB:
        assert self.to_amount is not None
        line = f"{player_name}: posts small blind {self.to_amount}"
      elif self.kind is ActionKind.BB:
        assert self.to_amount is not None
        line = f"{player_name}: posts big blind {self.to_amount}"
      else:  # ANTE
        assert self.to_amount is not None
        line = f"{player_name}: posts ante {self.to_amount}"
    elif self.kind is ActionKind.CHECK:
      line = f"{player_name}: checks"
    elif self.kind is ActionKind.FOLD:
      line = f"{player_name}: folds"
    else:
      line = f"{player_name}: {self.kind.name.lower()}"
    if self.all_in:
      line += " and is all-in"
    return line


@dataclasses.dataclass(frozen=True)
class Hand:
  """Data class representing a parsed ACPC hand."""
  hand_id: str
  config: Config
  players: list[Player]
  button_index: int | None  # None for legacy
  hole_cards: list[tuple[Card, Card]]
  community: list[list[Card]]  # [flop3, turn1, river1] (some may be empty)
  events: list[Event]
  winners: list[int]     # player indices
  profits: list[int]     # signed chips by player

  # Derived at parse time
  uncalled_amount: int = 0
  uncalled_receiver: int | None = None  # last aggressor when uncalled exists
  summary_folded: list[bool] = dataclasses.field(default_factory=list)


# ------------------------------------------------------------------------------
# Policies for seating / action order
# ------------------------------------------------------------------------------


class SeatingPolicy:
  """Abstract base class for defining seating and action order policies.

  Subclasses must implement `blind_indices` and `action_order`.
  """

  def blind_indices(
      self,
      cfg: Config,
      button_index: int
  ) -> tuple[int, int]:
    raise NotImplementedError

  def action_order(
      self,
      cfg: Config,
      button_index: int,
      street: Street
  ) -> list[int]:
    raise NotImplementedError


class LegacyACPCPolicy(SeatingPolicy):
  """Matches the original script behavior.

  - SB=seat 0, BB=seat 1 - Preflop order begins at seat 2 - Postflop order
  begins at seat 0
  """

  def blind_indices(
      self,
      cfg: Config,
      button_index: int
  ) -> tuple[int, int]:
    del button_index  # button_index ignored
    if cfg.seats == 2:
      return (1, 0)
    else:
      return (0, 1)

  def action_order(
      self, cfg: Config, button_index: int, street: Street
  ) -> list[int]:
    del button_index  # button_index ignored
    if street == Street.PREFLOP:
      start = 2 % cfg.seats
    elif street in (Street.FLOP, Street.TURN, Street.RIVER):
      start = 0
    else:
      return []
    return [(start + i) % cfg.seats for i in range(cfg.seats)]


class ButtonPolicy(SeatingPolicy):
  """Explicit button-aware policy; works for HU and ring games.

  HU rules:
    - SB = button, BB = next seat. Preflop starts at button (SB).
    - Postflop starts at BB.
  Ring rules:
    - SB = next seat after button, BB = next after SB.
    - Preflop starts at UTG = after BB.
    - Postflop starts left of button (SB).
  """

  def blind_indices(self, cfg: Config, button_index: int) -> tuple[int, int]:
    b = button_index
    if cfg.seats == 2:  # heads-up
      return (b, (b + 1) % cfg.seats)
    sb = (b + 1) % cfg.seats
    bb = (b + 2) % cfg.seats
    return (sb, bb)

  def action_order(
      self, cfg: Config, button_index: int, street: Street
  ) -> list[int]:
    b = button_index
    if street == Street.PREFLOP:
      if cfg.seats == 2:
        start = b  # button acts first preflop in HU
      else:
        _, bb = self.blind_indices(cfg, b)
        start = (bb + 1) % cfg.seats  # UTG
    elif street in (Street.FLOP, Street.TURN, Street.RIVER):
      if cfg.seats == 2:
        _, bb = self.blind_indices(cfg, b)
        start = bb
      else:
        start = (b + 1) % cfg.seats  # left of button
    else:
      return []
    return [(start + i) % cfg.seats for i in range(cfg.seats)]


# ------------------------------------------------------------------------------
# Tokenizer
# ------------------------------------------------------------------------------


_token_re = re.compile(r"c|f|r\d+")


def tokenize_actions(street_text: str) -> tuple[tuple[str, int | None], ...]:
  """Return tokens as ('c', None), ('f', None), ('r', amount).

  Args:
    street_text: The string representing actions on a single street.

  Raises ValueError with position info on invalid input.
  """
  if not street_text:
    return tuple()
  tokens = _token_re.findall(street_text)
  if "".join(tokens) != street_text:
    # find first bad position
    i = 0
    j = 0
    while i < len(street_text) and j < len(tokens):
      t = tokens[j]
      if street_text.startswith(t, i):
        i += len(t)
        j += 1
      else:
        break
    raise ValueError(f"Invalid action string at pos {i} in {street_text!r}")
  parsed: list[tuple[str, int | None]] = []
  k = 0
  while k < len(street_text):
    ch = street_text[k]
    if ch in ("c", "f"):
      parsed.append((ch, None))
      k += 1
    elif ch == "r":
      k += 1
      start = k
      while k < len(street_text) and street_text[k].isdigit():
        k += 1
      if k == start:
        raise ValueError(
            f"raise without amount at pos {start} in {street_text!r}"
        )
      amt = int(street_text[start:k])
      parsed.append(("r", amt))
    else:
      raise ValueError(f"invalid char {ch!r} at pos {k} in {street_text!r}")
  return tuple(parsed)


# ------------------------------------------------------------------------------
# Parsing utilities
# ------------------------------------------------------------------------------


@dataclasses.dataclass
class ParseState:
  """Internal state used while parsing an ACPC line."""

  # Per-street tracking
  street: Street = Street.BLINDS
  table_max: int = 0  # max committed this street
  prev_street_max: int = 0
  contrib_street: list[int] = dataclasses.field(default_factory=list)

  # Across-street totals
  contrib_total: list[int] = dataclasses.field(default_factory=list)
  active: list[bool] = dataclasses.field(default_factory=list)
  all_in: list[bool] = dataclasses.field(default_factory=list)
  last_aggressor: int | None = None
  uncalled_amount: int = 0


def _apply_delta(ps: ParseState, actor: int, delta: int, cfg: Config) -> None:
  ps.contrib_street[actor] += delta
  ps.contrib_total[actor] += delta
  if ps.contrib_total[actor] >= cfg.starting_stacks[actor]:
    ps.contrib_total[actor] = cfg.starting_stacks[actor]
    ps.all_in[actor] = True


def _advance_street(ps: ParseState) -> None:
  ps.prev_street_max = ps.table_max
  ps.table_max = 0
  ps.contrib_street = [0 for _ in ps.contrib_street]


# ------------------------------------------------------------------------------
# ACPC line parser → Hand
# ------------------------------------------------------------------------------


def parse_acpc_line(
    line: str,
    cfg: Config,
    policy: SeatingPolicy,
    button_index: int | None = None,
    hand_id_override: str | None = None,
) -> tuple[Hand, ParseState]:
  """Parses a single line from an ACPC hand history log.

  Args:
      line: The single line string from the log, starting with "STATE:".
      cfg: The Config object specifying game parameters.
      policy: The SeatingPolicy to use for blind and action order.
      button_index: The index of the player with the button. Required for
          `ButtonPolicy`, ignored for `LegacyACPCPolicy`.
      hand_id_override: If provided, this hand ID will be used instead of the
          one found in the ACPC line.

  Returns:
      A Hand object containing the parsed information.

  Raises:
      ValueError: If the line format is invalid or inconsistent with the config.
  """
  parts = line.strip().split(":")
  if len(parts) != 6 or parts[0] != "STATE":
    raise ValueError(f"Line is not a valid ACPC STATE with 6 parts: {line}")
  _, hand_num, actions_blob, cards_blob, profits_blob, players_blob = parts

  player_ids = players_blob.split("|")
  if len(player_ids) != cfg.seats:
    raise ValueError(
        f"Seat count mismatch: config seats={cfg.seats}, log has"
        f" {len(player_ids)} players"
    )

  # Parse cards
  cards_by_street = cards_blob.split("/")
  hole = cards_by_street[0].split("|")
  if len(hole) != cfg.seats:
    raise ValueError("Hole cards count does not match seats")
  hole_cards: list[tuple[Card, Card]] = []
  for h in hole:
    cs = [Card(x) for x in textwrap.wrap(h, 2) if x]
    if len(cs) != 2:
      raise ValueError("Each player must have exactly two hole cards")
    hole_cards.append((cs[0], cs[1]))

  community: list[list[Card]] = []
  for chunk in cards_by_street[1:]:
    cards = [Card(x) for x in textwrap.wrap(chunk, 2) if x]
    community.append(cards)

  # Profits / winners
  if profits_blob:
    profits = [int(float(p)) for p in profits_blob.split("|")]
    winners = [i for i, v in enumerate(profits) if v > 0]
  else:
    profits = []
    winners = []

  # Players
  players = [
      Player(id=pid, seat=i, stack_start=cfg.starting_stacks[i])
      for i, pid in enumerate(player_ids)
  ]

  # Seating policy selection
  if isinstance(policy, LegacyACPCPolicy):
    # Legacy ignores button
    b_idx = 0
  else:
    if button_index is None:
      raise ValueError("Button index is required for non-legacy policies")
    b_idx = button_index % cfg.seats

  # Prepare parse state
  ps = ParseState(
      street=Street.BLINDS,
      table_max=0,
      prev_street_max=0,
      contrib_street=[0] * cfg.seats,
      contrib_total=[0] * cfg.seats,
      active=[True] * cfg.seats,
      all_in=[False] * cfg.seats,
      last_aggressor=None,
      uncalled_amount=0,
  )

  events: list[Event] = []

  # Antes first (if any)
  if cfg.ante > 0:
    for p in range(cfg.seats):
      _apply_delta(ps, p, cfg.ante, cfg)
      events.append(Event(
          Street.BLINDS, p, ActionKind.ANTE, to_amount=cfg.ante, delta=cfg.ante
      ))

  # Blinds
  sb_idx, bb_idx = policy.blind_indices(cfg, b_idx)
  _apply_delta(ps, sb_idx, cfg.small_blind, cfg)
  ps.table_max = max(ps.table_max, cfg.small_blind)
  ps.contrib_street[sb_idx] = cfg.small_blind
  events.append(Event(
      Street.BLINDS, sb_idx, ActionKind.SB, to_amount=cfg.small_blind,
      delta=cfg.small_blind
  ))

  _apply_delta(ps, bb_idx, cfg.big_blind, cfg)
  ps.table_max = max(ps.table_max, cfg.big_blind)
  ps.contrib_street[bb_idx] = cfg.big_blind
  events.append(Event(
      Street.BLINDS,
      bb_idx,
      ActionKind.BB,
      to_amount=cfg.big_blind,
      delta=cfg.big_blind
  ))

  # Streets actions
  # preflop/flop/turn/river (some may be empty)
  streets_text = actions_blob.split("/")
  street_map = [Street.PREFLOP, Street.FLOP, Street.TURN, Street.RIVER]

  for s_idx, text in enumerate(streets_text):
    ps.street = street_map[s_idx] if s_idx < len(street_map) else Street.RIVER
    order = policy.action_order(cfg, b_idx, ps.street)
    if not text:
      # Move to next street: freeze table_max
      _advance_street(ps)
      continue
    tokens = tokenize_actions(text)
    # iterator over order cycling, but skip folded/all-in players
    cur_pos = 0
    # can_check if no wager yet this street (table_max equals prev_street_max
    # except preflop where prev_street_max is BB)
    # We'll determine checks by comparing needed delta to 0.
    for tok, amount in tokens:
      # Find next actor
      steps = 0
      while steps < cfg.seats and (
          not ps.active[order[cur_pos]] or ps.all_in[order[cur_pos]]
      ):
        cur_pos = (cur_pos + 1) % cfg.seats
        steps += 1
      actor = order[cur_pos]

      if tok == "f":
        ps.active[actor] = False
        events.append(Event(ps.street, actor, ActionKind.FOLD))
      elif tok == "c":
        need = max(ps.table_max - ps.contrib_street[actor], 0)
        if need == 0:
          events.append(Event(ps.street, actor, ActionKind.CHECK))
        else:
          _apply_delta(ps, actor, need, cfg)
          ps.uncalled_amount = 0
          events.append(Event(ps.street, actor, ActionKind.CALL, delta=need))
      elif tok == "r":
        assert amount is not None and amount >= 0
        # ACPC for OpenSpiel: amount = total chips committed during hand.
        chips_this_action = amount - ps.contrib_total[actor]
        if chips_this_action < 0:
          raise ValueError(
              f"Raise to {amount} is less than current commitment "
              f"{ps.contrib_total[actor]}"
          )

        kind = ActionKind.BET if ps.table_max == 0 else ActionKind.RAISE

        _apply_delta(ps, actor, chips_this_action, cfg)

        street_total_after_action = ps.contrib_street[actor]

        if kind == ActionKind.BET:
          event_delta = street_total_after_action
          event_to_amount = street_total_after_action
        else:  # ActionKind.RAISE
          event_delta = street_total_after_action - ps.table_max
          event_to_amount = street_total_after_action

        if event_delta < 0 and kind == ActionKind.RAISE:
          raise ValueError("Non-monotonic raise detected")

        ps.uncalled_amount = street_total_after_action - ps.table_max
        ps.table_max = street_total_after_action
        ps.last_aggressor = actor

        events.append(
            Event(
                ps.street,
                actor,
                kind,
                to_amount=event_to_amount,
                delta=event_delta,
                all_in=ps.all_in[actor],
            )
        )
      else:
        raise ValueError("Unknown token")

      # advance seat
      cur_pos = (cur_pos + 1) % cfg.seats

    # next street
    _advance_street(ps)

  # Derive folded summary for renderer
  summary_folded = [not a for a in ps.active]

  # Compute uncalled receiver and adjust pot contributions for side-pot calc
  uncalled_amount = ps.uncalled_amount
  uncalled_receiver = ps.last_aggressor if uncalled_amount > 0 else None

  hand = Hand(
      hand_id=hand_id_override if hand_id_override is not None else hand_num,
      config=cfg,
      players=players,
      button_index=None if isinstance(policy, LegacyACPCPolicy) else b_idx,
      hole_cards=hole_cards,
      community=community,
      events=events,
      winners=winners,
      profits=profits,
      uncalled_amount=uncalled_amount,
      uncalled_receiver=uncalled_receiver,
      summary_folded=summary_folded,
  )
  return hand, ps


# ------------------------------------------------------------------------------
# Renderer (Poker site-like formatting)
# ------------------------------------------------------------------------------


def render_pokersite(
    hand: Hand,
    observer_id: str | None = None,
    sitename: str = "",
) -> str:
  """Renders a Hand object into a hand history using a similar format as various poker sites.

  Args:
    hand: The Hand object to render.
    observer_id: If provided, only the hole cards of the player with this ID
      will be shown. Otherwise, all hole cards are shown.
    sitename: If provided, this string will be included in the header. With
      appropriate sitenames, the output can be handled by popular poker
      software.

  Returns:
    A multi-line string hand history.
  """
  cfg = hand.config

  lines: list[str] = []
  sitename_prefix = f"{sitename} " if sitename else ""
  top_line = (
      f"{sitename_prefix}Hand #{int(hand.hand_id)}: Hold'em No Limit"
      f" ({cfg.small_blind}/{cfg.big_blind})"
  )
  if cfg.timestamp is not None:
    top_line += f" - {cfg.timestamp.strftime('%Y/%m/%d %H:%M:%S ET')}"
  lines.append(top_line)
  # Note: some form of currency must be included for some software to work
  # correctly. Its absence can manifest in unexpected behavior such as seats
  # being out of order.
  second_line = f"Table '{cfg.table_name}' {cfg.seats}-max (USD)"
  button_display_number = (
      hand.button_index + 1 if hand.button_index is not None else cfg.seats
  )
  second_line += f" Seat #{button_display_number} is the button"
  lines.append(second_line)

  for p in hand.players:
    lines.append(
        f"Seat {p.seat + 1}: {p.id} ({cfg.starting_stacks[p.seat]} in chips)"
    )

  bb_idx = -1
  for e in hand.events:
    if e.kind == ActionKind.BB:
      bb_idx = e.actor
      break
  # Check if the hand is a preflop fold around. In this case, the expected
  # output is that the big blind is returned the difference between the big
  # blind and the small blind, and they collect 2x the small blind from the
  # pot. Not all visualizers enforce this behavior, but PokerTracker4 fails to
  # parse the hand history otherwise.
  is_preflop_fold_around = True
  for e in hand.events:
    if e.kind not in (
        ActionKind.SB,
        ActionKind.BB,
        ActionKind.ANTE,
        ActionKind.FOLD,
    ):
      is_preflop_fold_around = False
      break
  is_preflop_fold_around = (
      is_preflop_fold_around
      and bb_idx != -1
      and hand.summary_folded.count(False) == 1
      and not hand.summary_folded[bb_idx]
  )
  pfa_uncalled_amount = 0
  if is_preflop_fold_around:
    pfa_uncalled_amount = cfg.big_blind - cfg.small_blind

  # Blinds/Antes
  for event in [e for e in hand.events if e.street == Street.BLINDS]:
    lines.append(event.ps_text(hand.players[event.actor].id))

  # Hole cards
  lines.append("*** HOLE CARDS ***")
  for p, (c1, c2) in zip(hand.players, hand.hole_cards):
    if observer_id is None or p.id == observer_id:
      lines.append(f"Dealt to {p.id} [{c1} {c2}]")
    else:
      lines.append(f"Dealt to {p.id} [?? ??]")

  # Preflop
  for event in [e for e in hand.events if e.street == Street.PREFLOP]:
    lines.append(event.ps_text(hand.players[event.actor].id))

  # Flop, turn, river
  # Example street lines:
  # *** FLOP *** [As 5c Js]
  # *** TURN *** [As 5c Js] [Ac]
  # *** RIVER *** [As 5c Js] [Ac] [Ah]
  comm_accum: list[Card] = []
  board_cards_by_street: list[list[Card]] = []
  for st, name in [
      (Street.FLOP, "FLOP"),
      (Street.TURN, "TURN"),
      (Street.RIVER, "RIVER"),
  ]:
    idx = {Street.FLOP: 0, Street.TURN: 1, Street.RIVER: 2}[st]
    street_cards = hand.community[idx] if idx < len(hand.community) else []
    if street_cards:
      comm_accum.extend(street_cards)
      board_cards_by_street.append(street_cards)
      if st == Street.FLOP:
        board_txt = (
            "[" + " ".join(str(c) for c in board_cards_by_street[0]) + "]"
        )
      else:
        board_txt = " ".join(
            "[" + " ".join(str(c) for c in cards) + "]"
            for cards in board_cards_by_street
        )
      lines.append(f"*** {name} *** {board_txt}")
      for event in [e for e in hand.events if e.street == st]:
        lines.append(event.ps_text(hand.players[event.actor].id))

  hand_is_over = bool(hand.profits)
  if is_preflop_fold_around:
    if pfa_uncalled_amount > 0:
      receiver = hand.players[bb_idx]
      lines.append(
          f"Uncalled bet ({pfa_uncalled_amount}) returned to {receiver.id}"
      )
  elif (
      hand_is_over
      and hand.uncalled_amount > 0
      and hand.uncalled_receiver is not None
  ):
    receiver = hand.players[hand.uncalled_receiver]
    lines.append(
        f"Uncalled bet ({hand.uncalled_amount}) returned to {receiver.id}"
    )

  if not hand.profits:
    return "\n".join(lines)

  # Showdown decision (if all 5 board cards dealt and last action is check/call)
  river_events = [e for e in hand.events if e.street == Street.RIVER]
  last_river_action_is_passive = river_events and river_events[-1].kind in (
      ActionKind.CALL, ActionKind.CHECK)
  saw_showdown = bool(len(comm_accum) == 5 and last_river_action_is_passive)
  if saw_showdown:
    lines.append("*** SHOWDOWN ***")
    for wi in hand.winners:
      (c1, c2) = hand.hole_cards[wi]
      lines.append(f"{hand.players[wi].id}: shows [{c1} {c2}]")

  # Pot computation (total contributions minus uncalled), winners collection
  # Rebuild total contributions from events
  contrib_for_pot = [0] * cfg.seats
  street_contrib = [0] * cfg.seats
  current_street = Street.BLINDS
  for event in hand.events:
    if event.street != current_street:
      if not (
          current_street == Street.BLINDS and event.street == Street.PREFLOP
      ):
        street_contrib = [0] * cfg.seats
      current_street = event.street

    if event.kind in (
        ActionKind.SB,
        ActionKind.BB,
        ActionKind.ANTE,
    ):
      contrib_for_pot[event.actor] += event.delta
      street_contrib[event.actor] += event.delta
    elif event.kind == ActionKind.CALL:
      contrib_for_pot[event.actor] += event.delta
      street_contrib[event.actor] += event.delta
    elif event.kind == ActionKind.BET:
      contrib_for_pot[event.actor] += event.delta
      street_contrib[event.actor] += event.delta
    elif event.kind == ActionKind.RAISE:
      chips_this_action = event.to_amount - street_contrib[event.actor]
      contrib_for_pot[event.actor] += chips_this_action
      street_contrib[event.actor] += chips_this_action

  if is_preflop_fold_around:
    if pfa_uncalled_amount > 0:
      contrib_for_pot[bb_idx] -= pfa_uncalled_amount
  elif hand.uncalled_amount > 0 and hand.uncalled_receiver is not None:
    contrib_for_pot[hand.uncalled_receiver] -= hand.uncalled_amount
  total_pot = sum(contrib_for_pot)

  # Distribute winnings
  if hand.winners:
    share = total_pot / len(hand.winners)
    for winner in hand.winners:
      lines.append(f"{hand.players[winner].id} collected {share} from pot")
  else:
    # legacy edge case: SB/BB chop when both are winners implicitly
    if cfg.seats >= 2:
      lines.append(f"{hand.players[0].id} collected {total_pot/2} from pot")
      lines.append(f"{hand.players[1].id} collected {total_pot/2} from pot")

  # Summary
  lines.append("*** SUMMARY ***")
  lines.append(f"Total pot {total_pot} | Rake 0")
  if comm_accum:
    lines.append("Board [" + " ".join(str(c) for c in comm_accum) + "]")

  # Seat lines: show/fold
  if saw_showdown:
    for p in hand.players:
      idx = p.seat
      (c1, c2) = hand.hole_cards[idx]
      outcome = (
          "won ({})".format(total_pot / len(hand.winners))
          if hand.profits[idx] > 0
          else "lost"
      )
      lines.append(f"Seat {idx+1}: {p.id} showed [{c1} {c2}] and {outcome}")
  else:
    for p in hand.players:
      if hand.summary_folded[p.seat]:
        lines.append(f"Seat {p.seat+1}: {p.id} folded")

  return "\n".join(lines)
