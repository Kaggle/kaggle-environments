"""LLM harness for Word Association, built on core_harness.

Implements the ``GameHarness`` protocol:

- ``get_legal_moves(observation)`` — returns ``None`` for Cluemaster turns
  (free-form clue-giving) and ``{index: "INDEX: WORD"}`` for Guesser turns
  (enumerable word selection).
- ``generate_prompt(observation, move_history, ...)``
- ``parse_response(response, legal_action_strings)``

This is a **mixed harness**: Cluemaster turns use free-form actions
(the ``freeForm`` config flag must be ``true``), while Guesser turns use
enumerable actions validated against the legal-moves list.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

from kaggle_environments.core_harness import ParseResult, create_agent_fn

BOARD_SIZE = 25

_JSON_BLOCK_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)


# --- Helpers ----------------------------------------------------------------


def _get_team(turn: int) -> str:
    """Return the team name for a given turn index."""
    return "BLUE" if turn in (0, 1) else "YELLOW"


def _is_cluemaster(turn: int) -> bool:
    return turn in (0, 2)


def _inject_memory_context(observation: Mapping[str, Any]) -> str:
    """Build memory-context string from observation history fields.

    Note: history is already trimmed to ``memory_window_size`` at save time
    by ``memory.save_game_to_history``, so no additional slicing is needed here.
    """
    parts: list[str] = []

    history = observation.get("history")
    if history:
        parts.append("\nHere is the history of past games in this session:\n")
        parts.append(json.dumps(history, indent=2))
        parts.append("\n\n")

    current_game_turns = observation.get("current_game_turns")
    if current_game_turns:
        parts.append("Clues and guesses in this game so far:\n")
        parts.append(json.dumps(current_game_turns, indent=2))
        parts.append("\n\n")

    return "".join(parts)


def _build_cluemaster_board(observation: Mapping[str, Any]) -> str:
    """Board state for Cluemaster (full visibility)."""
    words = observation.get("words", [])
    roles = observation.get("roles", [])
    revealed = observation.get("revealed", [])
    lines: list[str] = []
    for i in range(BOARD_SIZE):
        status = "Revealed" if revealed[i] else "Hidden"
        lines.append(f"- {words[i]}: {roles[i].upper()} ({status})")
    return "\n".join(lines)


def _build_guesser_board(observation: Mapping[str, Any]) -> str:
    """Board state for Guesser (roles masked by the interpreter)."""
    words = observation.get("words", [])
    roles = observation.get("roles", [])
    revealed = observation.get("revealed", [])
    lines: list[str] = []
    for i in range(BOARD_SIZE):
        status = "Revealed" if revealed[i] else "Hidden"
        lines.append(f"- {i}: {words[i]} ({roles[i].upper()}, {status})")
    return "\n".join(lines)


def _build_clue_context(observation: Mapping[str, Any]) -> str:
    """Build the clue / guess-progress context string for Guesser turns."""
    clue = observation.get("clue", "")
    clue_number = observation.get("clue_number", 0)
    remaining = observation.get("guesses_remaining", 0)

    parts: list[str] = []

    # Clarify the last entry in current_game_turns for the guesser.
    if observation.get("current_game_turns"):
        parts.append(
            "Note: The last entry in the 'Clues and guesses in this game so "
            "far' list above represents your current turn, showing the guesses "
            "you have already made for the current clue.\n\n",
        )

    parts.append(
        f"The clue from your Cluemaster is: '{clue}' for {clue_number} "
        f"words. (You have {remaining} guesses remaining this turn.)\n\n",
    )

    if clue_number > 0:
        parts.append(
            f"If you correctly guess {clue_number} words based on this clue, "
            "you may make a bonus guess based on all information you've "
            "received so far.\n\n",
        )

        correct_guesses = (clue_number + 1) - remaining
        words_remaining = remaining - 1

        if correct_guesses > 0:
            current_guesses: list[str] = []
            game_turns = observation.get("current_game_turns")
            if game_turns:
                current_guesses = game_turns[-1].get("guesses", [])
            guesses_str = ", ".join(current_guesses)

            if words_remaining == 0:
                parts.append(
                    f"You have correctly guessed all {clue_number} words for "
                    f"this clue (Guessed: {guesses_str}). You are now on your "
                    "bonus guess!\n\n",
                )
            else:
                parts.append(
                    f"You have correctly guessed {correct_guesses} times for "
                    f"this clue already (Guessed: {guesses_str}), meaning "
                    f"there are {words_remaining} words related to the clue "
                    "remaining.\n\n",
                )
    elif clue_number == 0:
        parts.append(
            "A clue number of 0 means NONE of your remaining words relate to "
            "this clue (often used to point out the trap). You get unlimited "
            "guesses, but you MUST still make at least one guess.\n\n",
        )
    elif clue_number == -1:
        parts.append(
            "A clue number of -1 means 'Infinity'. You get unlimited guesses "
            "based on this clue and previous clues. You must make at least one "
            "guess.\n\n",
        )

    return "".join(parts)


def _extract_json(response: str) -> dict[str, Any] | None:
    """Try to extract a JSON object from the LLM response.

    Tries three strategies in order:
    1. Fenced ````` ```json {...} ``` ````` block.
    2. Strip markdown fences and parse the whole response.
    3. Extract substring between the first ``{`` and last ``}``.
    """
    # 1. Fenced JSON block.
    match = _JSON_BLOCK_RE.search(response)
    if match:
        try:
            return json.loads(match.group(1), strict=False)
        except json.JSONDecodeError:
            pass

    # 2. Strip markdown fences and try the whole response.
    clean = response.strip()
    if clean.startswith("```json"):
        clean = clean[7:]
    if clean.startswith("```"):
        clean = clean[3:]
    if clean.endswith("```"):
        clean = clean[:-3]
    try:
        return json.loads(clean.strip(), strict=False)
    except json.JSONDecodeError:
        pass

    # 3. First '{' … last '}'.
    first = response.find("{")
    last = response.rfind("}")
    if first != -1 and last > first:
        try:
            return json.loads(response[first : last + 1], strict=False)
        except json.JSONDecodeError:
            pass

    return None


# --- Rethink ----------------------------------------------------------------


RETHINK_SUFFIX = """

Your previous response was:
{previous_response}

You suggested "{previous_action}" but it could not be parsed as a valid
action. Reconsider and provide a valid response in the required JSON format.
"""


# --- Public functions (GameHarness protocol) --------------------------------


def get_legal_moves(observation: Mapping[str, Any]) -> dict[int, str] | None:
    """Return legal moves for the current turn.

    Returns ``None`` for Cluemaster turns (free-form action space).
    Returns ``{index: "INDEX: WORD", ..., -1: "-1: PASS"}`` for Guesser
    turns (enumerable).  PASS is only included when at least one guess has
    already been made for the current clue.
    """
    turn = observation.get("current_turn", 0)
    if _is_cluemaster(turn):
        return None

    words = observation.get("words", [])
    revealed = observation.get("revealed", [])
    clue_number = observation.get("clue_number", 0)
    guesses_remaining = observation.get("guesses_remaining", 0)

    moves: dict[int, str] = {}
    for i in range(len(words)):
        if not revealed[i]:
            moves[i] = f"{i}: {words[i]}"

    # PASS is legal only after at least one guess in this clue round.
    expected_remaining = BOARD_SIZE if clue_number <= 0 else clue_number + 1
    if guesses_remaining != expected_remaining:
        moves[-1] = "-1: PASS"

    return moves


def generate_prompt(
    observation: Mapping[str, Any],
    move_history: list[str],  # unused — protocol requires it but this game is stateless across turns
    previous_response: str | None = None,
    previous_action: str | None = None,
) -> str:
    """Build the LLM prompt for the current turn."""
    turn = observation.get("current_turn", 0)
    team = _get_team(turn)
    memory_context = _inject_memory_context(observation)

    if _is_cluemaster(turn):
        prompt = f"You are the {team} Cluemaster in Word Association.\n\n"
        prompt += (
            f"Your goal is to get your team to guess all your {team} words "
            "while avoiding the opposite team's words and the trap word.\n"
        )
        prompt += memory_context
        prompt += "Here is the board state:\n"
        prompt += _build_cluemaster_board(observation) + "\n"
        prompt += (
            "\nThink step-by-step about which unrevealed words you can "
            "connect with a single-word clue. Provide your reasoning in a "
            "'thinking' key.\n"
        )
        prompt += "VALIDITY RULES FOR CLUES:\n"
        prompt += (
            "- The clue must be a SINGLE WORD. It CANNOT contain spaces or "
            "hyphens.\n"
        )
        prompt += (
            "- The clue CANNOT contain or be contained within any unrevealed "
            "word currently hidden on the board (e.g., if 'DOG' is hidden, "
            "your clue cannot be 'DOGS' or 'HOTDOG').\n"
        )
        prompt += (
            "Note: A clue number of 0 means 'unlimited guesses, but 0 words "
            "relate to this clue' (often used to help guessers avoid the trap "
            "or opponent words). A clue number of -1 means 'infinity' "
            "(unlimited guesses, for when you want them to guess remaining "
            "words from previous clues).\n"
        )
        prompt += "You MUST format your response as valid JSON like this:\n"
        prompt += (
            '{"thinking": "I see CAT and DOG, so ANIMAL connects 2 '
            'words...", "clue": "ANIMAL", "number": 2}\n'
        )
        prompt += (
            "Do not include any other text or markdown formatting outside "
            "of the JSON block."
        )
    else:
        prompt = f"You are the {team} Guesser in Word Association.\n\n"
        prompt += (
            "Your goal is to correctly guess your team's words based on the "
            "Cluemaster's clues while avoiding the opposite team's words and "
            "the trap word.\n"
        )
        prompt += (
            "You must make at least one guess before you are allowed to pass, "
            "or else you forfeit the game.\n"
        )
        prompt += memory_context
        prompt += _build_clue_context(observation)
        prompt += "Here is the board state:\n"
        prompt += _build_guesser_board(observation) + "\n"
        prompt += (
            "\nThink step-by-step about which unrevealed word matches the "
            "clue best. Provide your reasoning in a 'thinking' key.\n"
        )
        prompt += (
            "Then provide the integer index of the ONE word you want to guess "
            "right now in a 'guess' key.\n"
        )
        prompt += (
            "If you want to pass, set 'guess' to -1. NOTE: You are NOT "
            "allowed to pass (-1) on your very first turn without making at "
            "least one guess for the current clue, or else you forfeit the "
            "game. If you do, your action will be marked INVALID and your "
            "team will lose.\n"
        )
        prompt += "You MUST format your response as valid JSON like this:\n"
        prompt += (
            '{"thinking": "The clue is ANIMAL. Cat is at index 4, so I will '
            'guess 4...", "guess": 4}\n'
        )
        prompt += (
            "Do not include any other text or markdown formatting outside "
            "of the JSON block."
        )

    if previous_response is not None:
        prompt += RETHINK_SUFFIX.format(
            previous_response=previous_response[:500],
            previous_action=previous_action or "(could not parse)",
        )

    return prompt


def parse_response(
    response: str,
    legal_action_strings: Sequence[str] | None,
) -> ParseResult:
    """Extract a move from the LLM response.

    For Cluemaster (free-form, ``legal_action_strings is None``):
        Returns ``ParseResult(submission={"clue": ..., "number": ...})``.

    For Guesser (enumerable):
        Returns ``ParseResult(legal_action=matched_string)``.
    """
    parsed = _extract_json(response)
    if parsed is None:
        return ParseResult(raw_action=response[:200])

    # --- Cluemaster (free-form) ---
    thinking = parsed.get("thinking")
    if legal_action_strings is None:
        clue = parsed.get("clue")
        number = parsed.get("number")
        if clue is not None and number is not None:
            try:
                num = int(number)
            except (ValueError, TypeError):
                return ParseResult(raw_action=response[:200], thoughts=thinking)
            return ParseResult(
                submission={"clue": str(clue), "number": num},
                raw_action=json.dumps({"clue": str(clue), "number": num}),
                thoughts=thinking,
            )
        return ParseResult(raw_action=response[:200], thoughts=thinking)

    # --- Guesser (enumerable) ---
    guess = parsed.get("guess")
    if guess is None:
        return ParseResult(raw_action=response[:200], thoughts=thinking)

    raw = str(guess)
    try:
        guess_int = int(guess)
    except (ValueError, TypeError):
        return ParseResult(legal_action=None, raw_action=raw, thoughts=thinking)

    # Match by index prefix in legal_action_strings (e.g. "4: APPLE").
    target = f"{guess_int}:"
    for legal in legal_action_strings:
        if legal.startswith(target):
            return ParseResult(legal_action=legal, raw_action=raw, thoughts=thinking)

    return ParseResult(legal_action=None, raw_action=raw, thoughts=thinking)


# --- Agent wiring -----------------------------------------------------------


class _WordAssociationHarness:
    """Adapter bridging module-level functions to the GameHarness protocol."""

    def get_legal_moves(self, observation):
        return get_legal_moves(observation)

    def make_prompt(
        self, observation, move_history, previous_response=None, previous_action=None,
    ):
        return generate_prompt(
            observation, move_history, previous_response, previous_action,
        )

    def parse_response(self, response, legal_action_strings):
        return parse_response(response, legal_action_strings)


agent_fn = create_agent_fn(_WordAssociationHarness())
