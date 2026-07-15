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

ULTIMATE_TIC_TAC_TOE_PROMPT_TEMPLATE = """Ultimate Tic-Tac-Toe.

Nine 3x3 local boards arranged in a larger 3x3 grid. Local boards indexed 0-8 (left-to-right, top-to-bottom). Cells within each local board are also indexed 0-8 (same convention); coordinates (row, col) map to index = row*3 + col (row, col in 0-2).

A local board is *active* if it has not yet been won, drawn, or fully filled. Under "Local Board Winners", finished boards show '[x]', '[o]', or '[draw]'; active boards show '[ ]'.

CRITICAL RULE: The cell you choose within a local board (index 0-8) determines which local board your opponent must play in next. Cell index 4 (center, coordinates 1,1) sends the opponent to Local Board 4. If that target board is not active, the opponent gets a "free move" in any active board.

Win a local board with three-in-a-row on that 3x3. Drawn local boards count for neither player.
Win the game with three-in-a-row on the overall 3x3 of local board winners. If all 9 local boards finish without either player scoring, the game is a draw.

Objective:
Your goal is to WIN -- be the first to make three-in-a-row of won local boards on the overall grid. Draws count for neither, so aim for a win rather than settling when a winning line is available.
Long-term planning matters: the cell you play now sends your opponent to a specific local board, which shapes their reply and constrains where you'll play after. Reason about downstream consequences, not just the immediate value of a move.

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
            # Center the title within the same 17-char slot the frame
            # below occupies ("  +---+---+---+  ") so it sits visually
            # above the sub-grid.
            if active_subgrid == subgrid_idx:
                header_parts.append(f"> Board {subgrid_idx} <".center(17))
            else:
                header_parts.append(f"Board {subgrid_idx}".center(17))
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
                    history_strings.append(
                        f"Player {player} ({symbol}): board {board_idx} cell ({r},{c}) [idx {cell_idx}]"
                    )
                else:
                    history_strings.append(f"Player {player} ({symbol}): {action_str}")
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
    # Default to "opening" on synthetic / setup-step observations that
    # haven't been through the proxy yet (matches phase's default and
    # keeps the ablation-check parity harness happy on pre-move obs).
    board_context = state.get("board_context") or "opening"

    my_piece = "x" if player_id == 0 else "o"
    opp_piece = "o" if player_id == 0 else "x"
    opp_player_id = 1 - player_id

    # JSON format templates depend only on phase (subgrid pick vs cell pick).
    if phase == "choose_subgrid":
        json_format_example = (
            '```json\n{\n  "move": "<subgrid_index>"\n}\n```\nFor example: `{"move": "0"}` to choose Local Board 0.'
        )
        format_reminder = '```json\n{{\n  "move": "<subgrid_index>"\n}}\n```\nFor example: `{{"move": "0"}}`'
    elif phase == "choose_cell":
        json_format_example = (
            "```json\n"
            "{\n"
            '  "move": "<row>,<col>"\n'
            "}\n"
            "```\n"
            'For example: `{"move": "1,1"}` or `{"move": "4"}` — both choose the center cell of the local board.'
        )
        format_reminder = (
            '```json\n{{\n  "move": "<row>,<col>"\n}}\n```\nFor example: `{{"move": "1,1"}}` or `{{"move": "4"}}`'
        )
    else:
        raise ValueError(f"Invalid or terminal phase: {phase}")

    # phase_instructions differentiate by *why* the player is in this
    # situation. The four cases (opening / redirected / self_selected /
    # opponent_directed) share the same JSON format inside a given phase
    # but merit different strategic framing — see the ultimate_tic_tac_toe
    # proxy's board_context docstring for the enumeration.
    if board_context == "opening":
        phase_instructions = (
            "FIRST MOVE of the game: choose any local board (0-8). "
            "Plan the board AND the cell you'll play in it next turn together -- "
            "the cell's index (0-8) sends the opponent to the same-numbered board."
        )
    elif board_context == "redirected":
        phase_instructions = (
            "Opponent's cell mapped to an inactive board, so you have a FREE MOVE -- "
            "choose any currently-active local board (0-8). "
            "Plan the board AND the cell together (the cell you play next turn sends the opponent to that-numbered board)."
        )
    elif board_context == "self_selected":
        phase_instructions = (
            f"You picked Local Board {active_subgrid} last turn; now choose an empty cell (0-8) in it to place '{my_piece}'. "
            "If you had a specific cell in mind when you picked this board, play it. "
            "(Move by (row,col) e.g. '1,1' or by cell index 0-8.)"
        )
    elif board_context == "opponent_directed":
        phase_instructions = (
            f"Opponent's cell sent you to Local Board {active_subgrid}. Choose an empty cell (0-8) in it to place '{my_piece}'. "
            "(Move by (row,col) e.g. '1,1' or by cell index 0-8.)"
        )
    else:
        raise ValueError(f"Unexpected board_context: {board_context!r}")

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
