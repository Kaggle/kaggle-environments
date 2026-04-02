"""LLM harness for OpenSpiel Go."""

import json
import os
import re
from typing import Any, Mapping, Sequence

import litellm
import pyspiel

litellm.drop_params = True

_JSON_BLOCK_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)


# --- Prompt ---


GO_PROMPT_TEMPLATE = """Let's play Go.

Rules: Tromp-Taylor scoring (area scoring — count stones on the board plus
empty territory enclosed by a single color; all stones are treated as alive).
Komi is given in the game state below. Two differences from standard
Tromp-Taylor: (1) suicide is illegal — you may not place a stone that would
be immediately captured unless it captures enemy stones first, and
(2) positional superko violations end the game as a draw rather than simply
making the move illegal. The game ends when both players pass consecutively.

The current game state is:
{state_str}
The moves played so far are:
{move_history}
You are playing as player {player_name} ({player_code}).
It is now your turn. Play your strongest move.
The move MUST be legal.
Your response should include the reasoning that led you to your move, and
conclude with your final move as a JSON formatted as follows:

```json
{{
  "move": "<move>"
}}
```

Where move is the coordinate only (e.g. "a1", "b2", "e5") or "PASS" if you wish to pass.
Coordinates use GTP notation: columns are lowercase letters a-h, j (the letter
"i" is skipped to avoid confusion with "l"), rows are numbers starting from 1.
For example on a 9x9 board, columns are a-h,j and rows are 1-9.
Failure to output your final answer in the specified format will result in a loss.
Begin!
"""


RETHINK_SUFFIX = """

Your previous response was:
{previous_response}

You suggested move "{previous_action}" but this is not in the legal moves list.
Reconsider and play a legal move.
"""


def _make_go_prompt(
    observation: Mapping[str, Any],
    move_history: list[str],
    previous_response: str | None = None,
    previous_action: str | None = None,
) -> str:
    """Create a Go prompt from the observation."""
    obs_string = observation.get("observationString", "")
    player_id = observation.get("playerId", 0)
    player_name = "Black" if player_id == 0 else "White"
    player_code = "B" if player_id == 0 else "W"

    move_history_str = " ".join(move_history) if move_history else "None"

    prompt = GO_PROMPT_TEMPLATE.format(
        state_str=obs_string,
        move_history=move_history_str,
        player_name=player_name,
        player_code=player_code,
    )

    if previous_response is not None:
        prompt += RETHINK_SUFFIX.format(
            previous_response=previous_response[:500],
            previous_action=previous_action or "(could not parse)",
        )

    return prompt


# --- Parser ---


def _extract_move_from_json(response: str) -> str | None:
    """Try to extract a move string from a JSON code block in the response."""
    match = _JSON_BLOCK_RE.search(response)
    if not match:
        return None
    try:
        data = json.loads(match.group(1))
        move = data.get("move", "").strip()
        return move or None
    except json.JSONDecodeError:
        return None


def _match_move_to_legal(
    move: str, legal_moves: Sequence[str],
) -> str | None:
    """Match a move string (e.g. "e5", "PASS") to a legal move string."""
    move_lower = move.lower()

    if move_lower == "pass":
        for legal in legal_moves:
            if legal.upper().endswith("PASS"):
                return legal
        return None

    for legal in legal_moves:
        parts = legal.split()
        if len(parts) == 2 and parts[1].lower() == move_lower:
            return legal

    return None


def _parse_go_response(response: str, legal_moves: Sequence[str]) -> str | None:
    """Parse model response to extract a Go move.

    Tries to extract move from JSON block first, then falls back to
    searching for coordinates in the response text.

    Args:
        response: The model's response text.
        legal_moves: List of legal move strings (e.g., ["B a1", "B b2", ...]).

    Returns:
        The matching legal move string, or None if no match found.
    """
    move = _extract_move_from_json(response)
    if move:
        result = _match_move_to_legal(move, legal_moves)
        if result:
            return result

    # Fallback: search for coordinates in response
    response_lower = response.lower()
    for legal in legal_moves:
        parts = legal.split()
        if len(parts) == 2:
            coord = parts[1].lower()
            if coord in response_lower:
                return legal

    return None


# --- Legal moves ---


def _get_legal_moves(
    observation: Mapping[str, Any],
) -> tuple[list[int], list[str]]:
    """Get legal actions and their string representations from observation."""
    legal_actions = observation.get("legalActions")
    legal_action_strings = observation.get("legalActionStrings")
    if legal_actions and legal_action_strings:
        return legal_actions, legal_action_strings

    serialized = observation.get("serializedGameAndState", "")
    if not serialized:
        return [], []
    _, state = pyspiel.deserialize_game_and_state(serialized)
    legal_actions = state.legal_actions()
    legal_action_strings = [state.action_to_string(a) for a in legal_actions]
    return legal_actions, legal_action_strings


# --- Agent ---


_SETUP_COMPLETE = False
_MODEL_NAME = None
_LITELLM_KWARGS: dict[str, str] = {}
_MOVE_HISTORY: list[str] = []


def agent_fn(
    obs: dict[str, Any] | Any, config: dict[str, Any],
) -> dict[str, int]:
    """Kaggle-compatible Go agent backed by an LLM."""
    global _SETUP_COMPLETE, _MODEL_NAME, _LITELLM_KWARGS

    if not _SETUP_COMPLETE:
        if "MODEL_NAME" not in os.environ:
            raise ValueError("MODEL_NAME environment variable is required.")
        if "MODEL_PROXY_KEY" not in os.environ:
            raise ValueError("MODEL_PROXY_KEY environment variable is required.")
        if "MODEL_PROXY_URL" not in os.environ:
            raise ValueError("MODEL_PROXY_URL environment variable is required.")

        _MODEL_NAME = os.environ["MODEL_NAME"]
        if os.environ["MODEL_PROXY_URL"] != "dummy_url":
            _MODEL_NAME = f"openai/{_MODEL_NAME}"
            _LITELLM_KWARGS = {
                "api_base": f"{os.environ['MODEL_PROXY_URL']}/openapi",
                "api_key": os.environ["MODEL_PROXY_KEY"],
            }
        elif (
            "gemini" in _MODEL_NAME.lower()
            and not _MODEL_NAME.startswith("gemini/")
        ):
            _MODEL_NAME = f"gemini/{_MODEL_NAME}"

        _SETUP_COMPLETE = True

    observation = obs if isinstance(obs, dict) else vars(obs)
    legal_actions, legal_action_strings = _get_legal_moves(observation)

    previous_response = None
    previous_action = None
    content = ""

    for attempt in range(2):
        prompt = _make_go_prompt(
            observation,
            _MOVE_HISTORY,
            previous_response=previous_response,
            previous_action=previous_action,
        )

        try:
            response = litellm.completion(
                model=_MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                **_LITELLM_KWARGS,
            )
            content = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[Go Harness] LLM call failed on attempt {attempt + 1}: {e}")
            raise

        action_str = _parse_go_response(content, legal_action_strings)
        if action_str is not None:
            idx = legal_action_strings.index(action_str)
            _MOVE_HISTORY.append(action_str)
            return {"submission": legal_actions[idx]}

        previous_action = _extract_move_from_json(content)
        previous_response = content
        print(f"[Go Harness] Attempt {attempt + 1} failed to parse a legal move.")

    raise ValueError(
        f"Failed to parse a legal move after 2 attempts. "
        f"Last response: {content[:200]}"
    )
