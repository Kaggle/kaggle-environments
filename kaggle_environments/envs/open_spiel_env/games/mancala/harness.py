"""LLM harness for OpenSpiel Mancala (Kalah ruleset).

Drop the body of this file into the notebook attached to the competition via
HarnessKernelId. The auto-generated ``main.py`` calls these three module-level
functions: ``get_legal_moves``, ``generate_prompt``, ``parse_response``.

Mancala (Kalah) is a two-player game played on a 14-cell board: each player
has a row of 6 pits and a store. Player 0's pits are indices 1-6 (store 7);
Player 1's pits are indices 8-13 (store 0). On a turn, a player picks one of
their pits with seeds and sows them counter-clockwise, one seed per cell,
skipping the opponent's store. If the last seed lands in the player's own
store, the player takes another turn. If it lands in an empty pit on the
player's own side and the opposite pit has seeds, the player captures those
seeds (plus the sowing seed) into their store. The game ends when one player
has no seeds left in their row; the player with more seeds in their store
wins.

Action strings are the integer pit index as a string, e.g. ``"3"`` for
player 0's 3rd pit, ``"11"`` for one of player 1's pits.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

import pyspiel

from kaggle_environments.core_harness import ParseResult, create_agent_fn, extract_last_json_object

_QUALIFIED_PIT_RE = re.compile(r"\bpit\s*(\d{1,2})\b", re.IGNORECASE)
_PIT_TOKEN_RE = re.compile(r"\b(\d{1,2})\b")


# --- Prompt -----------------------------------------------------------------


MANCALA_PROMPT_TEMPLATE = """Let's play Mancala (Kalah ruleset).

Rules: 14-cell board with two players. Each player owns a row of 6 pits
and one store. Player 0 owns pits 1-6 and store 7. Player 1 owns pits
8-13 and store 0. Seeds are sown counter-clockwise -- after pit 6, the
next cell is store 7, then pit 8; after pit 13, the next cell is store 0,
then pit 1. A player ALWAYS skips the opponent's store while sowing.

On your turn, pick one of YOUR pits that contains seeds. Pick them all
up and sow them one-at-a-time counter-clockwise into the following cells.

Bonus turn: if the last seed lands in your OWN store, you immediately take
another turn.

Capture: if the last seed lands in an empty pit on YOUR own side, AND the
pit directly opposite (on the opponent's side) contains seeds, you capture
both the sowing seed and all seeds in the opposite pit into your store.

A single move cannot both capture and earn a bonus turn: the bonus requires
the last seed to land in your store, while a capture requires it to land in
a regular pit on your side.

Game end: as soon as one player has no seeds in any of their 6 pits, the
game ends. Each player's final score is the total number of seeds on their
side of the board: their store PLUS any seeds remaining in their 6 pits.
(Equivalently, when the game ends, all seeds still on a side are credited
to that side's owner.) The player with the higher final score wins; equal
scores is a draw.

Board layout (fixed orientation, indices labeled):

    Player 1's pits:   [13] [12] [11] [10] [ 9] [ 8]
    Stores:        [0]                              [7]
    Player 0's pits:   [ 1] [ 2] [ 3] [ 4] [ 5] [ 6]

Current board (seed counts):

    Player 1 pits (13..8):   {p1_row}
    Player 1 store [0] = {p1_store}
    Player 0 store [7] = {p0_store}
    Player 0 pits (1..6):    {p0_row}

You are Player {player_id}. Move number: {move_number}.
{last_action_line}
Your move history: {move_history}.

It is your turn. Choose one of YOUR own pits that contains seeds.

Respond with your reasoning followed by your chosen pit index in a JSON
block:

```json
{{
  "move": "<pit index, e.g. 3>"
}}
```

Failure to output your final answer in the specified format, or choosing
an illegal pit, will result in a loss.
"""


RETHINK_SUFFIX = """

Your previous response was:
{previous_response}

You suggested move "{previous_action}" but it is not a legal move.
Reconsider the rules and the current state, then pick one of your own
pits that contains seeds.
"""


# --- Helpers ----------------------------------------------------------------


def _parse_observation_payload(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Pull the structured mancala state dict out of the observation."""
    raw = observation.get("observationString", "") or ""
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    serialized = observation.get("serializedGameAndState", "")
    if serialized:
        _, state = pyspiel.deserialize_game_and_state(serialized)
        try:
            return json.loads(state.observation_string(0))
        except (json.JSONDecodeError, RuntimeError):
            pass
    return {}


def _format_row(values: Sequence[int], width: int = 3) -> str:
    return " ".join(str(v).rjust(width) for v in values)


def _extract_move_from_json(response: str) -> str | None:
    """Pull the move string out of the LAST JSON object in the response."""
    data = extract_last_json_object(response, required_keys=("move",))
    if data is None:
        return None
    move = str(data.get("move") or "").strip()
    return move or None


# --- Public functions (called by main.py) -----------------------------------


def get_legal_moves(observation: Mapping[str, Any]) -> dict[int, str]:
    """Return ``{action_id: action_string}`` for the current state."""
    legal_actions = observation.get("legalActions")
    legal_action_strings = observation.get("legalActionStrings")
    if legal_actions and legal_action_strings:
        return dict(zip(legal_actions, legal_action_strings))

    serialized = observation.get("serializedGameAndState", "")
    if not serialized:
        return {}
    _, state = pyspiel.deserialize_game_and_state(serialized)
    player_id = observation.get("playerId", state.current_player())
    actions = state.legal_actions()
    return {a: state.action_to_string(player_id, a) for a in actions}


def generate_prompt(
    observation: Mapping[str, Any],
    move_history: list[str],
    previous_response: str | None = None,
    previous_action: str | None = None,
) -> str:
    """Build the LLM prompt for the current mancala state."""
    state = _parse_observation_payload(observation)
    player_id = observation.get("playerId", 0)

    pits = state.get("pits") or {}
    p0_pits = list(pits.get("0", [0] * 6))
    p1_pits = list(pits.get("1", [0] * 6))
    stores = state.get("stores") or {}
    p0_store = stores.get("0", 0)
    p1_store = stores.get("1", 0)

    # Player 1's row is displayed right-to-left (pits 13..8) so the visual
    # layout matches a real mancala board with counter-clockwise sowing.
    p1_row_display = list(reversed(p1_pits))

    move_number = state.get("move_number", 0)
    last_action = state.get("last_action")
    last_action_player = state.get("last_action_player")
    if last_action is None:
        last_action_line = "Last action played: (none yet)."
    elif last_action_player == player_id:
        last_action_line = (
            f"Last action played: you played pit {last_action} and your last "
            f"seed landed in your own store, so it is your BONUS TURN."
        )
    elif last_action_player is not None and last_action_player >= 0:
        last_action_line = (
            f"Last action played: Opponent (Player {last_action_player}) "
            f"played pit {last_action}."
        )
    else:
        last_action_line = f"Last action played: pit {last_action}."

    move_history_str = ", ".join(move_history) if move_history else "None"

    prompt = MANCALA_PROMPT_TEMPLATE.format(
        p1_row=_format_row(p1_row_display),
        p1_store=p1_store,
        p0_store=p0_store,
        p0_row=_format_row(p0_pits),
        player_id=player_id,
        move_number=move_number,
        last_action_line=last_action_line,
        move_history=move_history_str,
    )

    if previous_response is not None:
        prompt += RETHINK_SUFFIX.format(
            previous_response=previous_response[:500],
            previous_action=previous_action or "(could not parse)",
        )

    return prompt


def parse_response(
    response: str,
    legal_action_strings: Sequence[str],
) -> ParseResult:
    """Extract a legal Mancala pit index from the LLM response.

    Tries a ```json``` block first, then a bare ``{"move": ...}``, then
    falls back to scanning the response text for the first numeric token
    that matches a legal pit index.
    """
    legal_set = {legal.strip(): legal for legal in legal_action_strings}

    raw = _extract_move_from_json(response)
    if raw is not None and raw in legal_set:
        return ParseResult(legal_action=legal_set[raw], raw_action=raw)

    # Prefer mentions qualified with "pit" so a bare seed count like
    # "playing pit 4 distributes 9 seeds" picks 4, not 9. Iterate in reverse
    # so the *last* qualified mention wins -- models typically enumerate
    # rejected options before stating the final move.
    for token in reversed(_QUALIFIED_PIT_RE.findall(response)):
        if token in legal_set:
            return ParseResult(legal_action=legal_set[token], raw_action=raw or token)

    # Fallback: any bare numeric token that matches a legal pit, last wins.
    for token in reversed(_PIT_TOKEN_RE.findall(response)):
        if token in legal_set:
            return ParseResult(legal_action=legal_set[token], raw_action=raw or token)

    return ParseResult(legal_action=None, raw_action=raw)


# --- Adapter & agent function -----------------------------------------------


class _MancalaHarness:
    """Adapter wrapping module-level functions into the GameHarness protocol."""

    def get_legal_moves(self, observation):
        return get_legal_moves(observation)

    def make_prompt(
        self,
        observation,
        move_history,
        previous_response=None,
        previous_action=None,
    ):
        return generate_prompt(
            observation,
            move_history,
            previous_response=previous_response,
            previous_action=previous_action,
        )

    def parse_response(self, response, legal_action_strings):
        return parse_response(response, legal_action_strings)


agent_fn = create_agent_fn(_MancalaHarness())
