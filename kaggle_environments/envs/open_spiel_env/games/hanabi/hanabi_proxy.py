"""Structured JSON observations for Hanabi.

Hanabi is fully cooperative: every player shares one score (OpenSpiel reports
`Utility.IDENTICAL`), so there is no winner -- the outcome is the team's score,
0 to `colors * ranks`. Note that running out of life tokens zeroes the score
outright, no matter how many cards had already been played.

OpenSpiel's default observation_string is multi-line text like::

    Life tokens: 3
    Info tokens: 7
    Fireworks: R0 Y1 G0 W0 B0
    Hands:
    Cur player
    XX || XX|YGWB12345
    XX || RX|R12345
    -----
    W1 || XX|RYGWB12345
    B4 || XX|RYGWB12345
    Deck size: 40
    Discards: G5 W4

Hand blocks are separated by ``-----`` and ordered *relative to the observer*:
block ``i`` belongs to absolute player ``(observer + i) % num_players``, so
block 0 is always the observer's own hand. This proxy converts those relative
blocks to absolute player indices. The observer's own cards read ``XX``
(hidden) under the default and ``card_knowledge`` observation types; under
``observation_type=seer`` every card is face-up.

Each card line is ``<card> || <hinted>|<plausible>``:

- ``<card>`` is a color letter plus a rank digit (``R3``), or ``XX`` when the
  card is hidden from this observer.
- ``<hinted>`` is what this card's holder has been *told*: a color letter or
  ``X``, then a rank digit or ``X``. ``RX`` means "you know it is red, but not
  its rank".
- ``<plausible>`` is the set of colors and ranks still consistent with every
  hint the holder has received, e.g. ``R12345`` or ``YGWB12345``.

Hint information is the whole game, so this proxy preserves it exactly rather
than flattening it: `hinted_color` / `hinted_rank` are what was said out loud,
and `plausible_colors` / `plausible_ranks` are what can be deduced -- these
differ, because a hint aimed at other cards also rules possibilities out of
the ones it skipped.
"""

import json
import re
from typing import Any

import pyspiel

from ... import proxy

_LIFE_RE = re.compile(r"^Life tokens: (\d+)$")
_INFO_RE = re.compile(r"^Info tokens: (\d+)$")
_FIREWORKS_RE = re.compile(r"^Fireworks: (.*)$")
_FIREWORK_TOKEN_RE = re.compile(r"([A-Z])(\d+)")
_DECK_RE = re.compile(r"^Deck size: (\d+)$")
_DISCARDS_RE = re.compile(r"^Discards:\s*(.*)$")
_CARD_LINE_RE = re.compile(r"^(\S+) \|\| (\S)(\S)\|(\S*)$")
_HAND_SEPARATOR = "-----"


def _parse_card(token: str) -> dict[str, Any] | None:
    """Parse a ``R3``-style card token; ``XX`` (hidden from observer) is None."""
    if len(token) != 2 or token == "XX":
        return None
    color, rank = token[0], token[1]
    if not rank.isdigit():
        return None
    return {"color": color, "rank": int(rank)}


def _split_plausible(token: str) -> tuple[list[str], list[int]]:
    """Split ``RYGWB12345`` into its color letters and rank digits."""
    colors = [ch for ch in token if not ch.isdigit()]
    ranks = [int(ch) for ch in token if ch.isdigit()]
    return colors, ranks


class HanabiState(proxy.State):
    """Hanabi state proxy with structured JSON observations."""

    def _params(self) -> dict[str, Any]:
        params = self.get_game().get_parameters()
        num_players = int(params.get("players", 2))
        return {
            "num_players": num_players,
            "colors": int(params.get("colors", 5)),
            "ranks": int(params.get("ranks", 5)),
            # OpenSpiel's default hand size shrinks at 4+ players.
            "hand_size": int(params.get("hand_size", 5 if num_players < 4 else 4)),
            "max_life_tokens": int(params.get("max_life_tokens", 3)),
            "max_information_tokens": int(params.get("max_information_tokens", 8)),
        }

    def _parse_observation(self, observer: int, num_players: int) -> dict[str, Any]:
        """Parse one player's observation_string into a dict."""
        text = self.__wrapped__.observation_string(observer)
        result: dict[str, Any] = {
            "life_tokens": 0,
            "info_tokens": 0,
            "fireworks": {},
            "hands": [],
            "deck_size": 0,
            "discards": [],
        }
        # Hand blocks are relative to the observer and separated by "-----".
        # A "Cur player" marker line may precede any block; current_player()
        # is authoritative, so the marker is dropped.
        relative_hands: list[list[dict[str, Any]]] = [[]]
        in_hands = False
        for line in text.split("\n"):
            line = line.rstrip()
            if not line:
                continue
            if line == "Hands:":
                in_hands = True
                continue
            m = _LIFE_RE.match(line)
            if m:
                result["life_tokens"] = int(m.group(1))
                continue
            m = _INFO_RE.match(line)
            if m:
                result["info_tokens"] = int(m.group(1))
                continue
            m = _FIREWORKS_RE.match(line)
            if m:
                result["fireworks"] = {color: int(height) for color, height in _FIREWORK_TOKEN_RE.findall(m.group(1))}
                continue
            m = _DECK_RE.match(line)
            if m:
                in_hands = False
                result["deck_size"] = int(m.group(1))
                continue
            m = _DISCARDS_RE.match(line)
            if m:
                in_hands = False
                result["discards"] = [
                    card for card in (_parse_card(token) for token in m.group(1).split()) if card is not None
                ]
                continue
            if not in_hands:
                continue
            if line == _HAND_SEPARATOR:
                relative_hands.append([])
                continue
            m = _CARD_LINE_RE.match(line)
            if m:
                card_token, hinted_color, hinted_rank, plausible = m.groups()
                plausible_colors, plausible_ranks = _split_plausible(plausible)
                relative_hands[-1].append(
                    {
                        "card": _parse_card(card_token),
                        "hinted_color": None if hinted_color == "X" else hinted_color,
                        "hinted_rank": None if hinted_rank == "X" else int(hinted_rank),
                        "plausible_colors": plausible_colors,
                        "plausible_ranks": plausible_ranks,
                    }
                )

        # Convert observer-relative blocks to absolute player indices. Blocks
        # are padded because the pre-deal chance node emits no hands at all.
        while len(relative_hands) < num_players:
            relative_hands.append([])
        for offset, cards in enumerate(relative_hands[:num_players]):
            player = (observer + offset) % num_players
            result["hands"].append(
                {
                    "player": player,
                    "is_observer": player == observer,
                    "cards": cards,
                }
            )
        result["hands"].sort(key=lambda hand: hand["player"])
        return result

    def _final_turns_remaining(self, num_players: int) -> int:
        """Turns left in the endgame countdown, once the deck has run dry.

        The engine keeps this as ``turns_to_play_`` -- seeded at
        ``NumPlayers()`` and decremented on every decision applied after the
        deck empties -- but exposes it nowhere readable, so it is recovered
        from the history. The deck empties exactly at the final deal, and
        deals are the only chance nodes, so the decisions taken since the
        countdown began are the run of player moves at the tail.
        """
        taken = 0
        for item in reversed(self.full_history()):
            if item.player < 0:
                break
            taken += 1
        return max(num_players - taken, 0)

    def _outcome(self, life_tokens: int, fireworks_total: int, max_score: int) -> str:
        """Why the episode ended (or that it has not).

        Checked in the engine's own order (``EndOfGameStatus``): the life
        check precedes the completed-fireworks check. Judged on stack heights
        rather than the banked score, which is already zeroed by then.
        """
        if not self.is_terminal():
            return "in_progress"
        if life_tokens == 0:
            # Losing the last life zeroes the score regardless of progress.
            return "lives_exhausted"
        if fireworks_total == max_score:
            return "perfect_score"
        return "deck_exhausted"

    def state_dict(self, player: int | None = None) -> dict[str, Any]:
        observer = player if player is not None else 0
        params = self._params()
        parsed = self._parse_observation(observer, params["num_players"])

        # `fireworks_total` is the height the stacks reached; `score` is what
        # the team actually banks. They differ on the last life: the engine's
        # HanabiState::Score() returns 0 outright once life tokens hit 0, so
        # summing the stacks here would report points for a game that scored
        # none. Rather than restate that rule, read it off the engine --
        # Returns() is Score() for every seat and is defined at every state,
        # not just terminal ones. It is also what the env pays out as reward,
        # so score cannot drift from the result.
        fireworks_total = sum(parsed["fireworks"].values())
        score = int(self.returns()[0])
        max_score = params["colors"] * params["ranks"]

        # Every card is in exactly one of these four places, so the deck's
        # starting size is recoverable at any state -- no need to restate
        # NumberCardInstances, whose bottom-rank-before-top-rank ordering is
        # easy to get wrong at ranks=1.
        deck_total = (
            parsed["deck_size"]
            + sum(len(hand["cards"]) for hand in parsed["hands"])
            + fireworks_total
            + len(parsed["discards"])
        )

        # The countdown only exists once the deck is exhausted, and a finished
        # game has none left to take.
        final_turns_remaining = None
        if parsed["deck_size"] == 0 and not self.is_terminal():
            final_turns_remaining = self._final_turns_remaining(params["num_players"])

        # Only the acting player's own moves. The set of legal hints against a
        # hand is exactly the colors and ranks *in* that hand, so handing the
        # actor's move list to a non-acting observer would spell out the hand
        # that observer is not allowed to see. Asking the engine per observer
        # answers that directly: it yields [] for a non-actor, at a chance node
        # (whose "legal actions" are deck draws, not anything an agent picks),
        # and at a terminal state. Note this must be the wrapped call -- the
        # proxy's own legal_actions() takes no observer and returns the chance
        # outcomes at a deal.
        legal_actions = [
            {"action": action, "label": self.__wrapped__.action_to_string(observer, action)}
            for action in self.__wrapped__.legal_actions(observer)
        ]

        return {
            "num_players": params["num_players"],
            "colors": params["colors"],
            "ranks": params["ranks"],
            "hand_size": params["hand_size"],
            "observer": observer,
            "current_player": self.current_player(),
            "life_tokens": parsed["life_tokens"],
            "max_life_tokens": params["max_life_tokens"],
            "info_tokens": parsed["info_tokens"],
            "max_info_tokens": params["max_information_tokens"],
            "fireworks": parsed["fireworks"],
            "score": score,
            # Stack heights regardless of lives, so a bomb-out can still be
            # shown as "N fireworks lost" rather than vanishing into score 0.
            "fireworks_total": fireworks_total,
            "max_score": max_score,
            "hands": parsed["hands"],
            "deck_size": parsed["deck_size"],
            "deck_total": deck_total,
            # Null until the deck runs dry, then counts down the last turns.
            "final_turns_remaining": final_turns_remaining,
            "discards": parsed["discards"],
            "is_terminal": self.is_terminal(),
            # Hanabi is cooperative -- there is no winner, only a shared score.
            "winner": None,
            "outcome": self._outcome(parsed["life_tokens"], fireworks_total, max_score),
            "returns": list(self.returns()) if self.is_terminal() else [],
            "legal_actions": legal_actions,
            "move_number": self.move_number(),
        }

    def to_json(self, player: int | None = None) -> str:
        return json.dumps(self.state_dict(player))

    def observation_string(self, player: int) -> str:
        return self.to_json(player)

    def __str__(self) -> str:
        return self.to_json()


# Hanabi parameters whose spec default is a sentinel ("unset") rather than a
# usable value. See HanabiGame.__init__.
_SENTINEL_PARAMS = (
    "players",
    "colors",
    "ranks",
    "hand_size",
    "max_life_tokens",
    "max_information_tokens",
    "observation_type",
)


class HanabiGame(proxy.Game):
    """Wraps OpenSpiel's hanabi game to use the proxy state."""

    def __init__(self, params: Any | None = None):
        params = dict(params) if params else {}
        # pyspiel.load_game("hanabi_proxy", ...) backfills every key in the
        # inherited parameter_specification, and hanabi's spec defaults are
        # sentinels it does not accept as real values: 0 for the numeric
        # params and "" for observation_type. Passing those through means
        # `load_game("hanabi_proxy")` with no params hands hanabi
        # players=colors=ranks=0, which fails a C++ SPIEL_CHECK and aborts the
        # whole process -- not raises, aborts, so env registration cannot skip
        # it. Drop the sentinels and let hanabi apply its own defaults.
        for key in _SENTINEL_PARAMS:
            if not params.get(key):
                params.pop(key, None)
        wrapped = pyspiel.load_game("hanabi", params)
        super().__init__(
            wrapped,
            short_name="hanabi_proxy",
            long_name="Hanabi (proxy)",
        )

    def new_initial_state(self, *args) -> HanabiState:
        return HanabiState(self.__wrapped__.new_initial_state(*args), game=self)


pyspiel.register_game(HanabiGame().get_type(), HanabiGame)
