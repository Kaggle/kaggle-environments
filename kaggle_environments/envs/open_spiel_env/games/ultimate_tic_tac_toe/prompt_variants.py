"""Prompt variants for the Ultimate Tic-Tac-Toe ablation study.

Each variant implements the :class:`core_harness.GameHarness` protocol so the
ablation runner can do ``create_agent_fn(variant)`` directly. The production
prompt lives in :mod:`harness`; this file is the experimental surface.

Conventions enforced by ``python -m kaggle_environments.ablation check``:
  * The ``baseline`` variant in ``VARIANTS`` must produce a prompt
    BYTE-IDENTICAL to ``harness.generate_prompt`` for the same observation.
  * Every variant must render without error on five seeded observations.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

import pyspiel

from kaggle_environments.core_harness import (
    ParseResult,
    parse_json_action,
    render_rethink_suffix,
)


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

_BASELINE_TEMPLATE = """Let's play Ultimate Tic-Tac-Toe.

Ultimate Tic-Tac-Toe is played on a board of nine 3x3 local boards arranged in a larger 3x3 grid.
The nine local boards are indexed 0 to 8, numbered left-to-right, top-to-bottom.
Within each local board, the nine cells are also indexed 0 to 8 using the exact same left-to-right, top-to-bottom convention.
The coordinates (row, col) also map to indexes as index = row * 3 + col (where row and col are 0 to 2).

A local board is considered *active* (or legal to play in) if it has not yet been won, drawn, or fully filled. Once a local board is won or drawn, it is no longer active, and no further moves can be played in it.
In the overall game state display, finished local boards are listed under "Local Board Winners" with their status: '[x]', '[o]', or '[draw]'. Active boards are shown as '[ ]'.

CRITICAL RULE: The cell you choose within a local board determines which local board your opponent must play in next. Specifically, the index of the chosen cell (0 to 8) maps directly to the index of the target local board. For example, playing cell index 4 (center cell, coordinates 1,1) sends your opponent to Local Board 4. If the target local board is not active (i.e. already won, drawn, or full), your opponent gets a "free move" and can choose any active local board.

To win a local board, you must place three of your marks in a row on that 3x3 local board.
A local board can also end in a draw (all 9 cells filled with no 3-in-a-row); drawn local boards count for neither player in the overall game.
To win the overall game, you must win three local boards in a row (horizontally, vertically, or diagonally) in the overall 3x3 game.
The game ends in a draw if all 9 local boards finish without either player completing 3-in-a-row in the overall 3x3 game.

On your turn:
{phase_instructions}

Overall Game State:
{board_ascii}

You are Player {player_id} ('{my_piece}').
Opponent is Player {opp_player_id} ('{opp_piece}').

Moves played so far this game (both players, oldest first):
{move_history}

Choose your move now. Respond with your reasoning followed by your final move in a JSON block:
{json_format_example}

Failure to output your final answer in the specified format, or selecting an illegal move, will result in a loss.
"""


_COMPACT_TEMPLATE = """Ultimate Tic-Tac-Toe.

Nine 3x3 local boards arranged in a larger 3x3 grid. Local boards indexed 0-8 (left-to-right, top-to-bottom). Cells within each local board also indexed 0-8 using the same convention; coordinates (row, col) map to index = row*3 + col (row, col in 0-2).

A local board is *active* if it has not yet been won, drawn, or fully filled. Under "Local Board Winners", finished boards show '[x]', '[o]', or '[draw]'; active boards show '[ ]'.

CRITICAL RULE: The cell you choose within a local board (index 0-8) determines which local board your opponent must play in next. Cell index 4 (center, coordinates 1,1) sends the opponent to Local Board 4. If that target board is not active, the opponent gets a "free move" in any active board.

Win a local board with three-in-a-row on that 3x3. Drawn local boards count for neither player.
Win the game with three-in-a-row on the overall 3x3 of local board winners. If all 9 local boards finish without either player scoring, the game is a draw.

On your turn:
{phase_instructions}

Overall Game State:
{board_ascii}

You are Player {player_id} ('{my_piece}'). Opponent is Player {opp_player_id} ('{opp_piece}').

Moves played so far this game (both players, oldest first):
{move_history}

Respond with your reasoning, then end your response with your final move as JSON:
{json_format_example}
"""


_MINIMAL_TEMPLATE = """Ultimate Tic-Tac-Toe.

Nine 3x3 local boards arranged in a larger 3x3 grid. Local boards indexed 0-8 (left-to-right, top-to-bottom). Cells within each local board also indexed 0-8, same convention; (row, col) = row*3 + col.

A local board is active until it is won, drawn, or full; then it is closed and no further moves may be played in it. "Local Board Winners" shows '[x]'/'[o]'/'[draw]' for closed boards and '[ ]' for active ones.

The cell you choose within a local board (index 0-8) determines which local board your opponent must play in next. If that board is closed, the opponent may pick any active local board.

Win a local board with three-in-a-row (drawn locals count for neither). Win the game with three-in-a-row on the overall 3x3 of local board winners. Draw if all 9 local boards close without either player scoring.

On your turn:
{phase_instructions}

Overall Game State:
{board_ascii}

You are Player {player_id} ('{my_piece}'). Opponent is Player {opp_player_id} ('{opp_piece}').

Moves played so far this game (both players, oldest first):
{move_history}

Respond with your reasoning, then end your response with your final move as JSON:
{json_format_example}
"""


_RETHINK_ILLEGAL = """

You suggested move "{previous_action}" but this is not a legal move.
Reconsider the rules and the current state, then pick a legal move.

(Keep using the same JSON output format as before -- only the move value needs to change.)
"""

_RETHINK_UNPARSABLE = """

Your previous response ended with:
{previous_response}

No JSON answer could be parsed from that. Conclude your response
with your final move as JSON in a ```json fenced block, exactly
as the original instructions required:

{format_reminder}

The move you choose must also be legal in the current state.
"""


# ---------------------------------------------------------------------------
# Helpers ported verbatim from harness.py
# ---------------------------------------------------------------------------


def _parse_observation_payload(observation: Mapping[str, Any]) -> dict[str, Any]:
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


def _format_board_ascii(
    board: list[list[str]],
    subgrid_winners: list[str],
    active_subgrid: int | None = None,
    *,
    include_active_marker: bool = True,
) -> str:
    if not board:
        return "(board state unavailable)"

    sep = "      "
    lines = []
    for major_row in range(3):
        header_parts = []
        for mc in range(3):
            subgrid_idx = major_row * 3 + mc
            if include_active_marker and active_subgrid == subgrid_idx:
                header_parts.append(f"> Local Board {subgrid_idx} <")
            else:
                header_parts.append(f"  Local Board {subgrid_idx}  ")
        lines.append(sep.join(header_parts))

        divider = sep.join("  +---+---+---+  " for _ in range(3))
        lines.append(divider)

        for minor_row in range(3):
            row_parts = []
            for major_col in range(3):
                subgrid_idx = major_row * 3 + major_col
                cells = []
                for minor_col in range(3):
                    cell_idx = minor_row * 3 + minor_col
                    char = board[subgrid_idx][cell_idx]
                    cells.append(char if char else ".")
                row_parts.append(f"{minor_row} | " + " | ".join(cells) + " |  ")
            lines.append(sep.join(row_parts))

        lines.append(divider)
        footer = sep.join("    0   1   2    " for _ in range(3))
        lines.append(footer)
        lines.append("")

    lines.append("Local Board Winners (overall 3x3 game):")
    for r in range(3):
        winners_row = []
        for c in range(3):
            idx = r * 3 + c
            w = subgrid_winners[idx]
            w_disp = f"[{w}]" if w else "[ ]"
            winners_row.append(f"{idx}: {w_disp}")
        lines.append("  ".join(winners_row))

    return "\n".join(lines)


def _reconstruct_move_history(
    observation: Mapping[str, Any],
    *,
    include_idx: bool = True,
) -> list[str]:
    serialized = observation.get("serializedGameAndState", "")
    if not serialized:
        return []
    try:
        game, state = pyspiel.deserialize_game_and_state(serialized)
        temp_state = game.new_initial_state()
        history_strings = []
        for action in state.history():
            player = temp_state.current_player()
            action_str = temp_state.action_to_string(player, action)
            symbol = "x" if player == 0 else "o"

            m_board = re.match(r"^choose local board (\d)", action_str, re.IGNORECASE)
            if m_board:
                board_idx = m_board.group(1)
                history_strings.append(f"Player {player} ({symbol}): chose board {board_idx}")
            else:
                m_cell = re.match(r"^local board (\d):\s*([xo])\((\d),(\d)\)", action_str, re.IGNORECASE)
                if m_cell:
                    board_idx, sym, r, c = m_cell.groups()
                    cell_idx = int(r) * 3 + int(c)
                    if include_idx:
                        history_strings.append(
                            f"Player {player} ({symbol}): board {board_idx} cell ({r},{c}) [idx {cell_idx}]"
                        )
                    else:
                        history_strings.append(
                            f"Player {player} ({symbol}): board {board_idx} cell ({r},{c})"
                        )
                else:
                    history_strings.append(f"Player {player} ({symbol}): {action_str}")
            temp_state.apply_action(action)
        return history_strings
    except Exception:
        return []


def _match_ultimate_tic_tac_toe(raw: str, legal_action_strings: Sequence[str]) -> str | None:
    if not legal_action_strings:
        return None

    raw = raw.strip().lower()

    for legal in legal_action_strings:
        if raw == legal.lower():
            return legal

    if legal_action_strings[0].lower().startswith("choose local board"):
        matches = list(re.finditer(r"\b([0-8])\b", raw))
        if matches:
            subgrid = matches[-1].group(1)
            target = f"choose local board {subgrid}"
            for legal in legal_action_strings:
                if legal.lower() == target:
                    return legal
        return None

    if legal_action_strings[0].lower().startswith("local board"):
        first_legal = legal_action_strings[0]
        m = re.match(r"^local board (\d):\s*([xo])\(", first_legal, re.IGNORECASE)
        if not m:
            return None
        subgrid, symbol = m.group(1), m.group(2).lower()

        m_coords = None
        matches_coords = list(re.finditer(r"\b([0-2])\s*[,.\s-]\s*([0-2])\b", raw))
        if not matches_coords:
            matches_coords = list(re.finditer(r"\(([0-2])\s*,\s*([0-2])\)", raw))
        if matches_coords:
            m_coords = matches_coords[-1]

        if m_coords:
            r, c = m_coords.group(1), m_coords.group(2)
            target = f"local board {subgrid}: {symbol}({r},{c})"
            for legal in legal_action_strings:
                if legal.lower() == target.lower():
                    return legal

        m_cell = re.match(r"^([0-8])$", raw)
        if m_cell:
            cell_idx = int(m_cell.group(1))
            r, c = cell_idx // 3, cell_idx % 3
            target = f"local board {subgrid}: {symbol}({r},{c})"
            for legal in legal_action_strings:
                if legal.lower() == target.lower():
                    return legal

        matches_fallback = list(re.finditer(r"([0-2])\s*,\s*([0-2])", raw))
        if matches_fallback:
            m_coords_fallback = matches_fallback[-1]
            r, c = m_coords_fallback.group(1), m_coords_fallback.group(2)
            target = f"local board {subgrid}: {symbol}({r},{c})"
            for legal in legal_action_strings:
                if legal.lower() == target.lower():
                    return legal

    return None


def _get_legal_moves(observation: Mapping[str, Any]) -> dict[int, str]:
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


# ---------------------------------------------------------------------------
# Phase-instruction / JSON-format builders (baseline text)
# ---------------------------------------------------------------------------


def _baseline_choose_subgrid_instructions(*, include_reemphasis: bool) -> str:
    base = (
        "You are currently allowed to choose ANY active local board to play in (either because it is the first turn of the game, or because your opponent's previous move sent you to a local board that is no longer active).\n"
        "A local board is active if it has not yet been won, drawn, or fully filled.\n"
        "Select one of the active local boards (index 0 to 8) to target."
    )
    if include_reemphasis:
        base += "\n(The CRITICAL RULE about cell->board routing applies to your *next* turn, when you select a cell within this board.)"
    return base


def _baseline_choose_cell_instructions(active_subgrid: int, my_piece: str, *, include_reemphasis: bool) -> str:
    base = (
        f"You must play in Local Board {active_subgrid}. Choose an empty cell in Local Board {active_subgrid} to place your '{my_piece}'.\n"
        "You can specify your move either by row and column coordinates (e.g. '1,1') or by cell index (0 to 8, numbered left-to-right, top-to-bottom)."
    )
    if include_reemphasis:
        base += "\nRemember: the cell you choose (0 to 8) determines which local board your opponent must play in next."
    return base


_CHOOSE_SUBGRID_JSON_FORMAT = (
    '```json\n{\n  "move": "<subgrid_index>"\n}\n```\nFor example: `{"move": "0"}` to choose Local Board 0.'
)
_CHOOSE_SUBGRID_FORMAT_REMINDER = (
    '```json\n{{\n  "move": "<subgrid_index>"\n}}\n```\nFor example: `{{"move": "0"}}`'
)
_CHOOSE_CELL_JSON_FORMAT = (
    "```json\n"
    "{\n"
    '  "move": "<row>,<col>"\n'
    "}\n"
    "```\n"
    'For example: `{"move": "1,1"}` or `{"move": "4"}` — both choose the center cell of the local board.'
)
_CHOOSE_CELL_FORMAT_REMINDER = (
    '```json\n{{\n  "move": "<row>,<col>"\n}}\n```\nFor example: `{{"move": "1,1"}}` or `{{"move": "4"}}`'
)


# ---------------------------------------------------------------------------
# BASELINE — byte-identical port of harness.generate_prompt
# ---------------------------------------------------------------------------


class BaselineVariant:
    """Byte-identical port of harness.py. DO NOT paraphrase."""

    name = "baseline"
    template = _BASELINE_TEMPLATE

    # Feature flags for helpers. Variants flip these to strip specific
    # decorative or computed help while keeping the rest identical.
    include_phase_reemphasis = True
    include_history_idx = True
    include_active_marker = True

    def get_legal_moves(self, observation):
        return _get_legal_moves(observation)

    def make_prompt(
        self,
        observation,
        move_history,
        previous_response=None,
        previous_action=None,
    ):
        body = self._build_body(observation)
        body += self._render_rethink(previous_response, previous_action, observation)
        return body

    def parse_response(self, response, legal_action_strings, *, observation=None):
        del observation
        return parse_json_action(
            response,
            legal_action_strings,
            matcher=_match_ultimate_tic_tac_toe,
        )

    # ------------------------------------------------------------------

    def _build_body(self, observation):
        state = _parse_observation_payload(observation)
        player_id = observation.get("playerId", 0)

        board = state.get("board") or []
        subgrid_winners = state.get("subgrid_winners") or [""] * 9
        active_subgrid = state.get("active_subgrid")
        phase = state.get("phase", "choose_subgrid")

        my_piece = "x" if player_id == 0 else "o"
        opp_piece = "o" if player_id == 0 else "x"
        opp_player_id = 1 - player_id

        if phase == "choose_subgrid":
            phase_instructions = _baseline_choose_subgrid_instructions(
                include_reemphasis=self.include_phase_reemphasis,
            )
            json_format_example = _CHOOSE_SUBGRID_JSON_FORMAT
        elif phase == "choose_cell":
            phase_instructions = _baseline_choose_cell_instructions(
                active_subgrid,
                my_piece,
                include_reemphasis=self.include_phase_reemphasis,
            )
            json_format_example = _CHOOSE_CELL_JSON_FORMAT
        else:
            raise ValueError(f"Invalid or terminal phase: {phase}")

        full_history = _reconstruct_move_history(
            observation,
            include_idx=self.include_history_idx,
        )
        move_history_str = ", ".join(full_history) if full_history else "None"

        return self.template.format(
            phase_instructions=phase_instructions,
            board_ascii=_format_board_ascii(
                board,
                subgrid_winners,
                active_subgrid,
                include_active_marker=self.include_active_marker,
            ),
            player_id=player_id,
            my_piece=my_piece,
            opp_piece=opp_piece,
            opp_player_id=opp_player_id,
            move_history=move_history_str,
            json_format_example=json_format_example,
        )

    def _render_rethink(self, previous_response, previous_action, observation):
        state = _parse_observation_payload(observation)
        phase = state.get("phase", "choose_subgrid")
        if phase == "choose_subgrid":
            format_reminder = _CHOOSE_SUBGRID_FORMAT_REMINDER
        else:
            format_reminder = _CHOOSE_CELL_FORMAT_REMINDER

        rethink_unparsable_formatted = _RETHINK_UNPARSABLE.format(
            previous_response="{previous_response}",
            format_reminder=format_reminder,
        )
        return render_rethink_suffix(
            _RETHINK_ILLEGAL,
            rethink_unparsable_formatted,
            previous_response,
            previous_action,
        )


# ---------------------------------------------------------------------------
# NULL — byte-identical duplicate; calibrates the LLM-sampling noise floor
# ---------------------------------------------------------------------------


class NullVariant(BaselineVariant):
    name = "null"


# ---------------------------------------------------------------------------
# COMPACT — same info + helpers, terser prose; drops the loss-threat closing
# ---------------------------------------------------------------------------


class CompactVariant(BaselineVariant):
    """Tighter rewrite of the baseline. Every helper baseline computes is
    still emitted verbatim — active-board arrows in the ASCII, [idx N]
    annotations in the move history, phase-instruction re-emphasis. Only
    the prose framing is compressed, plus the loss-threat closing is
    removed. Tests whether the verbose scaffolding is load-bearing while
    holding the helper surface constant."""

    name = "compact"
    template = _COMPACT_TEMPLATE


# ---------------------------------------------------------------------------
# MINIMAL — strips all computed helpers; keeps mechanics + reason-first ask
# ---------------------------------------------------------------------------


class MinimalVariant(BaselineVariant):
    """Strips computed helpers on top of a terser template:
      * No '> Local Board X <' arrows on the ASCII board.
      * No '[idx N]' annotation on cell rows in move history.
      * No phase-instruction re-emphasis of the CRITICAL RULE.
    Mechanics are preserved (indexing rules, cell->board routing rule,
    active-board definition, win/draw conditions) and the reason-first
    ask is retained. Tests whether the model can derive what baseline
    spoon-feeds."""

    name = "minimal"
    template = _MINIMAL_TEMPLATE
    include_phase_reemphasis = False
    include_history_idx = False
    include_active_marker = False


# ---------------------------------------------------------------------------
# NO_CRITICAL_RULE_REEMPHASIS — baseline with only the phase-level
# re-emphasis of the CRITICAL RULE removed
# ---------------------------------------------------------------------------


class NoCriticalRuleReemphasisVariant(BaselineVariant):
    """Baseline verbatim except the phase-instruction re-emphasis of the
    cell->board routing rule is dropped:
      * choose_subgrid: no parenthetical '(The CRITICAL RULE about
        cell->board routing applies to your *next* turn ...)' tail.
      * choose_cell: no 'Remember: the cell you choose (0 to 8)
        determines which local board your opponent must play in next.'
    The CRITICAL RULE paragraph in the rules intro is untouched, so the
    model still learns the rule once. Tests whether restating it inside
    the per-turn instructions is doing work independent of the rule
    statement."""

    name = "no_critical_rule_reemphasis"
    include_phase_reemphasis = False


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


VARIANTS: dict[str, Any] = {
    "baseline": BaselineVariant(),
    "null": NullVariant(),
    "compact": CompactVariant(),
    "minimal": MinimalVariant(),
    "no_critical_rule_reemphasis": NoCriticalRuleReemphasisVariant(),
}
