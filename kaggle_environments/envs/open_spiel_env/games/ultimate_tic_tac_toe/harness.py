"""LLM harness for OpenSpiel Ultimate Tic Tac Toe.

Ultimate Tic-Tac-Toe is a strategic variant of Tic-Tac-Toe played on a board
consisting of nine 3x3 Tic-Tac-Toe sub-grids arranged in a larger 3x3 grid.
Player 0 ('x') moves first; player 1 ('o') follows.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

import pyspiel

from kaggle_environments.core_harness import ParseResult, parse_json_action, render_rethink_suffix

# --- Prompt Templates --------------------------------------------------------

ULTIMATE_TIC_TAC_TOE_PROMPT_TEMPLATE = """Let's play Ultimate Tic-Tac-Toe.

Ultimate Tic-Tac-Toe is played on a board of nine 3x3 local boards arranged in a larger 3x3 grid.
The nine local boards are indexed 0 to 8, numbered left-to-right, top-to-bottom.
Within each local board, the nine cells are also indexed 0 to 8 using the exact same left-to-right, top-to-bottom convention.
The coordinates (row, col) also map to indexes as index = row * 3 + col (where row and col are 0 to 2).

CRITICAL RULE: The cell you choose within a local board determines which local board your opponent must play in next. Specifically, the index of the chosen cell (0 to 8) maps directly to the index of the target local board. For example, playing cell index 4 (center cell, coordinates 1,1) sends your opponent to Local Board 4. If the target local board is already won, drawn, or full, your opponent gets a "free move" and can choose any active local board.

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

RETHINK_ILLEGAL = """

You suggested move "{previous_action}" but this is not a legal move.
Reconsider the rules and the current state, then pick a legal move.

(Keep using the same JSON output format as before -- only the move value needs to change.)
"""

RETHINK_UNPARSABLE = """

Your previous response ended with:
{previous_response}

No JSON answer could be parsed from that. Conclude your response
with your final move as JSON in a ```json fenced block, exactly
as the original instructions required:

{format_reminder}

The move you choose must also be legal in the current state.
"""

# --- Helpers -----------------------------------------------------------------


def _parse_observation_payload(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Pull the structured ultimate tic tac toe state dict out of the observation."""
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


def _format_board_ascii(board: list[list[str]], subgrid_winners: list[str], active_subgrid: int | None = None) -> str:
    """Format the 9x9 board into a 3x3 layout of 3x3 subgrids."""
    if not board:
        return "(board state unavailable)"

    sep = "      "
    lines = []
    # Loop over major rows (0, 1, 2)
    for major_row in range(3):
        # Header line for major row
        header_parts = []
        for mc in range(3):
            subgrid_idx = major_row * 3 + mc
            if active_subgrid == subgrid_idx:
                header_parts.append(f"> Local Board {subgrid_idx} <")
            else:
                header_parts.append(f"  Local Board {subgrid_idx}  ")
        lines.append(sep.join(header_parts))

        divider = sep.join("  +---+---+---+  " for _ in range(3))
        lines.append(divider)

        # Loop over minor rows (0, 1, 2)
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
        lines.append("")  # empty line between major rows

    # Add subgrid winners
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


def _reconstruct_move_history(observation: Mapping[str, Any]) -> list[str]:
    """Reconstruct the list of all played moves with player labels from deserialized state."""
    serialized = observation.get("serializedGameAndState", "")
    if not serialized:
        return []
    try:
        game, state = pyspiel.deserialize_game_and_state(serialized)
        temp_state = game.new_initial_state()
        history_strings = []
        pending_board_choice = None
        for action in state.history():
            player = temp_state.current_player()
            action_str = temp_state.action_to_string(player, action)
            symbol = "x" if player == 0 else "o"

            m_board = re.match(r"^choose local board (\d)", action_str, re.IGNORECASE)
            if m_board:
                pending_board_choice = m_board.group(1)
            else:
                m_cell = re.match(r"^local board \d:\s*(.*)", action_str, re.IGNORECASE)
                cell_part = m_cell.group(1) if m_cell else action_str
                if pending_board_choice is not None:
                    history_strings.append(
                        f"Player {player} ({symbol}): choose board {pending_board_choice}, play {cell_part}"
                    )
                    pending_board_choice = None
                else:
                    history_strings.append(f"Player {player} ({symbol}): play {cell_part}")
            temp_state.apply_action(action)
        return history_strings
    except Exception:
        return []


def match_ultimate_tic_tac_toe(raw: str, legal_action_strings: Sequence[str]) -> str | None:
    """Game-specific matcher for Ultimate Tic-Tac-Toe actions."""
    if not legal_action_strings:
        return None

    raw = raw.strip().lower()

    # 1. Exact case-insensitive match check
    for legal in legal_action_strings:
        if raw == legal.lower():
            return legal

    # 2. Check if we are in choose_subgrid phase
    # Legal actions: "Choose local board <idx>"
    if legal_action_strings[0].lower().startswith("choose local board"):
        # Match single digit or "subgrid/board <digit>" (take the last occurrence)
        matches = list(re.finditer(r"\b([0-8])\b", raw))
        if matches:
            subgrid = matches[-1].group(1)
            target = f"choose local board {subgrid}"
            for legal in legal_action_strings:
                if legal.lower() == target:
                    return legal
        return None

    # 3. Check if we are in choose_cell phase
    # Legal actions: "Local board <subgrid>: <symbol>(<row>,<col>)"
    if legal_action_strings[0].lower().startswith("local board"):
        first_legal = legal_action_strings[0]
        m = re.match(r"^local board (\d):\s*([xo])\(", first_legal, re.IGNORECASE)
        if not m:
            return None
        subgrid, symbol = m.group(1), m.group(2).lower()

        # Parse row,col coordinates (take the last occurrence)
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

        # Parse single cell index (0-8)
        m_cell = re.match(r"^([0-8])$", raw)
        if m_cell:
            cell_idx = int(m_cell.group(1))
            r, c = cell_idx // 3, cell_idx % 3
            target = f"local board {subgrid}: {symbol}({r},{c})"
            for legal in legal_action_strings:
                if legal.lower() == target.lower():
                    return legal

        # Fallback: search for row,col anywhere in the string (take the last occurrence)
        matches_fallback = list(re.finditer(r"([0-2])\s*,\s*([0-2])", raw))
        if matches_fallback:
            m_coords_fallback = matches_fallback[-1]
            r, c = m_coords_fallback.group(1), m_coords_fallback.group(2)
            target = f"local board {subgrid}: {symbol}({r},{c})"
            for legal in legal_action_strings:
                if legal.lower() == target.lower():
                    return legal

    return None


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
    """Build the LLM prompt for the current Ultimate Tic-Tac-Toe state."""
    state = _parse_observation_payload(observation)
    player_id = observation.get("playerId", 0)

    board = state.get("board") or []
    subgrid_winners = state.get("subgrid_winners") or [""] * 9
    active_subgrid = state.get("active_subgrid")
    phase = state.get("phase", "choose_subgrid")

    my_piece = "x" if player_id == 0 else "o"
    opp_piece = "o" if player_id == 0 else "x"
    opp_player_id = 1 - player_id

    # Format phase-specific instructions and JSON templates
    if phase == "choose_subgrid":
        phase_instructions = (
            "You are currently allowed to choose ANY active local board to play in (either because it is the first turn of the game, or because your opponent's previous move sent you to a local board that is already won or full).\n"
            "Select one of the legal local boards (index 0 to 8) to target."
        )
        json_format_example = (
            '```json\n{\n  "move": "<subgrid_index>"\n}\n```\nFor example: `{"move": "0"}` to choose Local Board 0.'
        )
        format_reminder = '```json\n{{\n  "move": "<subgrid_index>"\n}}\n```\nFor example: `{{"move": "0"}}`'
    elif phase == "choose_cell":
        phase_instructions = (
            f"You must play in Local Board {active_subgrid}. Choose an empty cell in Local Board {active_subgrid} to place your '{my_piece}'.\n"
            "You can specify your move either by row and column coordinates (e.g. '1,1') or by cell index (0 to 8, numbered left-to-right, top-to-bottom).\n"
            "Remember: the cell you choose (0 to 8) determines which local board your opponent must play in next."
        )
        json_format_example = (
            "```json\n"
            "{\n"
            '  "move": "<row>,<col>"\n'
            "}\n"
            "```\n"
            'For example: `{"move": "1,1"}` to choose the center cell of the local board.'
        )
        format_reminder = '```json\n{{\n  "move": "<row>,<col>"\n}}\n```\nFor example: `{{"move": "1,1"}}`'
    else:
        raise ValueError(f"Invalid or terminal phase: {phase}")

    # Reconstruct history of moves from both players
    full_history = _reconstruct_move_history(observation)
    move_history_str = ", ".join(full_history) if full_history else "None"

    prompt = ULTIMATE_TIC_TAC_TOE_PROMPT_TEMPLATE.format(
        phase_instructions=phase_instructions,
        board_ascii=_format_board_ascii(board, subgrid_winners, active_subgrid),
        player_id=player_id,
        my_piece=my_piece,
        opp_piece=opp_piece,
        opp_player_id=opp_player_id,
        move_history=move_history_str,
        json_format_example=json_format_example,
    )

    rethink_unparsable_formatted = RETHINK_UNPARSABLE.format(
        previous_response="{previous_response}",
        format_reminder=format_reminder,
    )

    prompt += render_rethink_suffix(
        RETHINK_ILLEGAL,
        rethink_unparsable_formatted,
        previous_response,
        previous_action,
    )

    return prompt


def parse_response(
    response: str,
    legal_action_strings: Sequence[str],
) -> ParseResult:
    """Trust the model's JSON answer; let the rethink loop fix anything else."""
    return parse_json_action(
        response,
        legal_action_strings,
        matcher=match_ultimate_tic_tac_toe,
    )
