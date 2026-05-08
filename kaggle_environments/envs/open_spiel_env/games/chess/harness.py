"""LLM harness for OpenSpiel Chess.

Migrated from the Google DeepMind GameArena chess harness. The prompts
generated here are identical to those produced by ``game_arena.harness.games.chess``.

The auto-generated ``main.py`` calls these three module-level functions:
``get_legal_moves``, ``generate_prompt``, ``parse_response``.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

import pyspiel

from kaggle_environments.core_harness import ParseResult, create_agent_fn

# ---------------------------------------------------------------------------
# Player mapping (matches OpenSpiel chess convention)
# ---------------------------------------------------------------------------

_PLAYER_MAP = {0: "Black", 1: "White"}

# ---------------------------------------------------------------------------
# Prompt templates — exact replicas from GameArena prompt_templates.py
# ---------------------------------------------------------------------------

# GameArena's NO_LEGAL_ACTIONS template, with {rethink_prompt} appended.
_PROMPT_TEMPLATE = """\
Let's play {game_short_name}. The current game state in {notation} is:
{readable_state_str}
The moves played so far are:
{move_history}
You are playing as player {player_name}.
It is now your turn. Play your strongest move. The move MUST be legal. Reason step by step to come up with your move, then output your final answer in the format "Final Answer: X" where X is your chosen move in {move_notation}.
{rethink_prompt}"""

# GameArena's RETHINK_WITH_ENV_UNPARSABLE
_RETHINK_UNPARSABLE = """\
Your previously suggested move was not parsable.
Please think carefully and generate a new and legal move. Your previous response was:
{generation}
"""

# GameArena's RETHINK_WITH_ENV_ILLEGAL
_RETHINK_ILLEGAL = """\
Your previously suggested move was: {last_move}, which is an illegal move.
Please think carefully and generate a new and legal move.
"""


# ---------------------------------------------------------------------------
# PGN movetext builder (replaces GameArena's get_pgn + format_chess_movetext)
# ---------------------------------------------------------------------------


def _build_pgn_movetext(state: pyspiel.State) -> str:
    """Build PGN movetext from a pyspiel chess state.

    Replicates the output of GameArena's ``format_chess_movetext`` with
    ``numbering_scheme="default"``, ``use_lan=False``, ``add_current_fen=False``.

    Format examples:
    - Empty game (White to play):  ``1.``
    - After 1. e4 (Black to play):  ``1. e4``
    - After 1. e4 e5 (White to play):  ``1. e4 e5 2.``
    - After 1. e4 e5 2. Nf3 (Black to play):  ``1. e4 e5 2. Nf3``

    A trailing move number is appended when it is White's turn, matching
    GameArena's behavior where the loop iterates ``range(len(nodes) + 1)``
    and adds the move number before checking for end-of-mainline.
    """
    history = state.history()
    game = state.get_game()
    tmp = game.new_initial_state()
    parts: list[str] = []

    for i, action in enumerate(history):
        if i % 2 == 0:
            # White's move — prepend move number
            parts.append(f"{i // 2 + 1}.")
        san = tmp.action_to_string(tmp.current_player(), action)
        parts.append(san)
        tmp.apply_action(action)

    # Trailing move number when it's White's turn (even number of moves).
    n = len(history)
    if n % 2 == 0:
        parts.append(f"{n // 2 + 1}.")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Move extraction (replicates GameArena's RuleBasedMoveParser)
# ---------------------------------------------------------------------------

_HTML_TAG_RE = re.compile(r"<.*?>")


def _extract_move_from_response(
    response: str,
    action_tag: str = "Final Answer:",
    additional_tags: Sequence[str] = (":", "is"),
) -> str | None:
    """Extract a move string from the LLM response.

    Finds the last occurrence of ``action_tag`` or any ``additional_tags``
    in the response and extracts the text that follows, stripping noise.

    This is a faithful port of GameArena's ``parse_move_from_response``
    (from ``parsers.py``) followed by the ``RuleBasedMoveParser``.
    """
    if response is None:
        return None

    last_index = -1
    final_split_token = ""
    for split_token in [action_tag, *additional_tags]:
        tmp_index = response.rfind(split_token)
        if tmp_index > last_index:
            last_index = tmp_index
            final_split_token = split_token

    if last_index == -1:
        return None

    suffix = response[last_index + len(final_split_token) :]
    if suffix is None:
        return None

    move_str = (
        suffix.strip(" .")
        .replace("$", "")
        .replace("\\boxed{", "")
        .replace("\\text{", "")
        .replace("\boxed{", "")
        .replace("\text{", "")
        .replace("}", "")
        .replace("*", "")
        .replace(" ", "")
        .replace("`", "")
        .replace("\n", "")
    )
    move_str = _HTML_TAG_RE.sub("", move_str)

    if not move_str:
        return None

    return move_str


# ---------------------------------------------------------------------------
# Soft move parser (replicates GameArena's ChessSoftParser / chess_soft_parser_v1)
# ---------------------------------------------------------------------------

# Characters to strip from a candidate move (same list as GameArena).
_CHARS_TO_REMOVE = ":.*, &^\\<>{}[]?!"


def _soft_match_move(
    candidate: str,
    legal_moves: Sequence[str],
) -> str | None:
    """Try to match a candidate move string to one of the legal moves.

    Replicates the cleaning and matching logic from GameArena's
    ``chess_soft_parser_v1``, adapted to work without python-chess by
    matching against the OpenSpiel legal-move strings directly.
    """
    if not candidate:
        return None

    candidate = candidate.strip()
    if not candidate:
        return None

    # Strip leading move-number prefix (e.g. "1." or "2...")
    if not candidate.startswith("0-0") and candidate[0].isdigit():
        match = re.search(r"(\d+)(\.{1,3})(.*)", candidate)
        if match is not None:
            _, _, candidate = match.groups()
        else:
            return None

    candidate = candidate.lstrip()

    # Remove noise characters
    for ch in _CHARS_TO_REMOVE:
        candidate = candidate.replace(ch, "")

    # Remove en passant annotation
    candidate = candidate.removesuffix("ep")

    if not candidate:
        return None

    # --- Matching stages ---

    # Stage 1: exact match
    if candidate in legal_moves:
        return candidate

    # Stage 2: match ignoring check/checkmate symbols
    candidate_stripped = candidate.rstrip("+#")
    for legal in legal_moves:
        if candidate_stripped == legal.rstrip("+#"):
            return legal

    # Stage 3: case-insensitive match (SAN is normally case-sensitive,
    # but LLMs sometimes change case)
    candidate_lower = candidate_stripped.lower()
    for legal in legal_moves:
        if candidate_lower == legal.rstrip("+#").lower():
            return legal

    return None


# ---------------------------------------------------------------------------
# Public functions (called by core_harness / main.py)
# ---------------------------------------------------------------------------


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
    actions = state.legal_actions()
    return {a: state.action_to_string(state.current_player(), a) for a in actions}


def generate_prompt(
    observation: Mapping[str, Any],
    move_history: list[str],
    previous_response: str | None = None,
    previous_action: str | None = None,
) -> str:
    """Build the LLM prompt for the current chess position.

    Produces a prompt identical to GameArena's ``NO_LEGAL_ACTIONS_RETHINK_APPENDED``
    template with ``RETHINK_WITH_ENV`` strategy.
    """
    # --- Build FEN (readable state) ---
    fen = observation.get("observationString", "")

    # --- Build PGN movetext from pyspiel state ---
    serialized = observation.get("serializedGameAndState", "")
    if serialized:
        _, state = pyspiel.deserialize_game_and_state(serialized)
        pgn_movetext = _build_pgn_movetext(state)
        player_id = state.current_player()
    else:
        pgn_movetext = "None"
        player_id = observation.get("playerId", 0)

    player_name = _PLAYER_MAP.get(player_id, str(player_id))

    # --- Build rethink prompt ---
    if previous_response is not None:
        if previous_action is None:
            # Unparseable response
            rethink_prompt = _RETHINK_UNPARSABLE.format(
                generation=previous_response,
            )
        else:
            # Parsed but illegal move
            rethink_prompt = _RETHINK_ILLEGAL.format(
                last_move=previous_action,
            )
    else:
        rethink_prompt = ""

    return _PROMPT_TEMPLATE.format(
        game_short_name="chess",
        notation="Forsyth-Edwards Notation (FEN) notation",
        readable_state_str=fen,
        move_history=pgn_movetext,
        player_name=player_name,
        move_notation="standard algebraic notation (SAN)",
        rethink_prompt=rethink_prompt,
    )


def parse_response(
    response: str,
    legal_action_strings: Sequence[str],
) -> ParseResult:
    """Extract a legal chess move from the model response.

    Two-stage pipeline matching GameArena's approach:
    1. ``RuleBasedMoveParser`` — extract text after "Final Answer:" tag
    2. ``ChessSoftParser`` — validate/match against legal moves
    """
    # Stage 1: extract candidate move
    raw = _extract_move_from_response(response)

    if raw is None:
        return ParseResult(legal_action=None, raw_action=None)

    # Stage 2: soft-match against legal moves
    matched = _soft_match_move(raw, legal_action_strings)

    if matched is not None:
        return ParseResult(legal_action=matched, raw_action=raw)

    return ParseResult(legal_action=None, raw_action=raw)


# ---------------------------------------------------------------------------
# Adapter + agent factory
# ---------------------------------------------------------------------------


class _ChessHarness:
    """Adapts module-level harness functions to the ``GameHarness`` protocol."""

    def get_legal_moves(self, observation):
        return get_legal_moves(observation)

    def make_prompt(self, observation, move_history, previous_response=None, previous_action=None):
        return generate_prompt(observation, move_history, previous_response, previous_action)

    def parse_response(self, response, legal_action_strings):
        return parse_response(response, legal_action_strings)


agent_fn = create_agent_fn(_ChessHarness())
