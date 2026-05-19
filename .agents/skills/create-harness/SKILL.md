---
name: create-harness
description: Create or update an LLM harness that lets a language model play a kaggle-environments game. Use this skill whenever the user wants to write a harness, LLM agent, or game-playing prompt for any kaggle-environments game — including OpenSpiel games, word games, or custom environments. Also use it when the user mentions core_harness.py, GameHarness, ParseResult, or asks how to connect an LLM to a game.
---

# Create an LLM Harness

A **harness** bridges a game environment and an LLM. It translates game observations into prompts, sends them to the model, and parses the model's response back into a legal game action. The framework in `kaggle_environments/core_harness.py` handles the LLM call, retry loop, telemetry, and agent lifecycle — you only implement the game-specific logic.

## What you're building

A harness module that implements three functions (the `GameHarness` protocol):

| Method | Purpose |
|--------|---------|
| `get_legal_moves(observation)` | Extract legal actions from the observation |
| `make_prompt(observation, move_history, ...)` | Build the LLM prompt |
| `parse_response(response, legal_action_strings)` | Extract the chosen action from the LLM's text |

Plus an adapter class that wraps these into the protocol, and a call to `create_agent_fn()` to produce the final Kaggle agent.

## Step 1: Understand the game

Before writing any code, understand the game you're building a harness for:

1. **Read the game's interpreter** to understand actions, observations, and turn structure.
2. **For OpenSpiel games**, check for a proxy (`*_proxy.py`) — if one exists, the observation will be structured JSON rather than raw OpenSpiel text. The proxy's `state_dict()` method shows exactly what fields the LLM will see. Non-OpenSpiel environments already provide structured JSON observations directly, so no proxy is needed.
3. **Identify the action space type:**
   - **Enumerable** (most games): a fixed set of legal moves per turn (e.g., board coordinates, directions, bids). `get_legal_moves` returns `dict[int, str]`.
   - **Free-form** (rare): the agent can submit any structured object (e.g., a clue in a word game). `get_legal_moves` returns `None`.
   - **Mixed** (rarest): some turns are enumerable, others are free-form, determined at runtime.
4. **Understand the observation dict.** The harness receives an observation dict with these standard fields:
   - `observationString` — the game state (often JSON from a proxy)
   - `legalActions` — list of action IDs (ints)
   - `legalActionStrings` — list of human-readable action strings
   - `currentPlayer` — whose turn it is (`-2` means simultaneous)
   - `playerId` — this agent's player ID
   - `isTerminal` — whether the game is over
   - `serializedGameAndState` — full pyspiel state (OpenSpiel games only)

## Step 2: Implement `get_legal_moves`

This function extracts legal actions from the observation and returns them as `{action_id: action_string}`.

```python
from typing import Any, Mapping, Sequence
from kaggle_environments.core_harness import ParseResult

def get_legal_moves(observation: Mapping[str, Any]) -> dict[int, str] | None:
    """Return {action_id: action_string} for enumerable games, or None for free-form."""
    legal_actions = observation.get("legalActions")
    legal_action_strings = observation.get("legalActionStrings")
    if legal_actions and legal_action_strings:
        return dict(zip(legal_actions, legal_action_strings))
    return {}
```

**For OpenSpiel games with a fallback** (when the environment doesn't always provide `legalActions`):

```python
def get_legal_moves(observation: Mapping[str, Any]) -> dict[int, str]:
    legal_actions = observation.get("legalActions")
    legal_action_strings = observation.get("legalActionStrings")
    if legal_actions and legal_action_strings:
        return dict(zip(legal_actions, legal_action_strings))
    # Fallback: deserialize pyspiel state
    serialized = observation.get("serializedGameAndState", "")
    if not serialized:
        return {}
    _, state = pyspiel.deserialize_game_and_state(serialized)
    return {a: state.action_to_string(a) for a in state.legal_actions()}
```

**For mixed action spaces** (like word_association), return `None` on free-form turns:

```python
def get_legal_moves(observation: Mapping[str, Any]) -> dict[int, str] | None:
    if observation.get("isCluemaster"):
        return None  # Free-form: LLM submits any valid clue
    # Enumerable: return word choices
    words = observation.get("words", [])
    return {i: f"{i}: {words[i]}" for i in range(len(words))}
```

## Step 3: Write the prompt

The prompt is the most important part of a harness — it determines how well the LLM plays. The `make_prompt` function (sometimes named `generate_prompt` in older code) builds the full text sent to the model.

### Signature

```python
def generate_prompt(
    observation: Mapping[str, Any],
    move_history: list[str],
    previous_response: str | None = None,
    previous_action: str | None = None,
) -> str:
```

- `observation` — the current game state
- `move_history` — list of this agent's past action strings (persists across turns)
- `previous_response` — the LLM's last response if this is a retry after an illegal move
- `previous_action` — the illegal move that was attempted

### Prompt structure

A good prompt follows this pattern:

1. **Game rules** — concise explanation of how the game works
2. **Current state** — the board/situation, parsed from `observation["observationString"]`
3. **Move history** — what's happened so far
4. **Player identity** — which side the LLM is playing
5. **Output format** — tell the LLM exactly how to format its move
6. **Rethink suffix** — appended on retries to explain what went wrong

**Do not enumerate the legal moves in the prompt.** The framework already
validates the LLM's response against `legalActionStrings` and triggers a
retry with the rethink suffix on an illegal move, so listing every legal
action just bloats the prompt, encourages the model to pick mechanically
from the list instead of reasoning about the position, and trivializes the
benchmark for games whose challenge is partly *finding* the legal moves
(e.g. checkers' forced captures). Instead, describe the move-legality rules
clearly and let the model derive legality from the state. The lone
exception is when the legal-action set is not derivable from the visible
state at all (e.g. a hidden hand of cards the LLM owns but the rules text
cannot enumerate) — in that case, list only what the model cannot infer.

### Prompt template example

```python
GAME_PROMPT_TEMPLATE = """\
You are playing {game_name}.

Rules: {rules_description}

Current game state:
{state_str}

Moves played so far: {move_history_str}

You are player {player_id}.

Your response should include your reasoning, then conclude with your move as JSON:

```json
{{"move": "<your_move>"}}
```
"""

RETHINK_SUFFIX = """
Your previous response was:
{previous_response}

You suggested move "{previous_action}" but this is not a legal move.
Reconsider the rules and the current state, then pick a legal move.
"""
```

### Prompt writing tips

- **Parse the observation string.** If the game has a proxy, `observation["observationString"]` is JSON — parse it and present the state clearly rather than dumping raw JSON.
- **Be specific about the output format.** The LLM's response needs to be parseable. JSON with a clear field name (like `"move"`) works well.
- **Explain the coordinate system.** If the game uses a board, explain notation clearly (e.g., "columns are letters a-h, rows are numbers 1-9").
- **Include rules that affect strategy.** Don't just list rules mechanically — highlight the ones that matter for making good decisions.
- **Don't give strategy advice.** The prompt should explain rules and mechanics, not coach the model on how to play. Saying "capture pieces to win" is fine (that's a rule); saying "control the center early" or "prefer defensive moves" is not (that's strategy). The LLM should reason about strategy on its own from the rules and game state.
- **Keep the rethink suffix concise.** Truncate the previous response to ~500 characters. Include the illegal move attempt and remind the LLM to re-derive a legal move from the state and rules (do *not* paste in a legal-moves list on retry either — same reasoning as above).
- **Move history formatting.** Show it as a readable list or "None" if empty. Don't let an empty string confuse the model.

```python
move_history_str = ", ".join(move_history) if move_history else "None"
```

## Step 4: Implement `parse_response`

The parser extracts the LLM's chosen move from its text response and matches it to a legal action. This is where robustness matters — LLMs don't always follow instructions perfectly.

### Signature and return type

```python
def parse_response(
    response: str,
    legal_action_strings: Sequence[str] | None,
) -> ParseResult:
```

`ParseResult` is a frozen dataclass with three fields:

```python
@dataclasses.dataclass(frozen=True)
class ParseResult:
    legal_action: str | None = None   # Matched legal move string (enumerable)
    raw_action: str | None = None     # What the model actually said (for rethink context)
    submission: Any = None            # Parsed object (free-form only)
```

**For enumerable actions:** set `legal_action` to the matched string from `legal_action_strings`, and `raw_action` to what the model originally said.

**For free-form actions:** set `submission` to the parsed object (e.g., a dict), and `raw_action` to a string representation.

**On failure:** return `ParseResult(raw_action=<what_was_attempted>)` — the framework will retry with the rethink prompt.

### Multi-stage parsing pattern

Every harness in this repo follows a multi-stage fallback pattern for robustness:

```python
import json
import re

_JSON_BLOCK_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)
_BARE_JSON_RE = re.compile(r'\{[^{}]*"move"\s*:\s*"([^"]+)"[^{}]*\}')

def _extract_move_from_json(response: str) -> str | None:
    """Try to extract a move from JSON in the response."""
    # Stage 1: fenced ```json ... ``` block
    match = _JSON_BLOCK_RE.search(response)
    if match:
        try:
            data = json.loads(match.group(1))
            if "move" in data:
                return str(data["move"]).strip()
        except json.JSONDecodeError:
            pass

    # Stage 2: bare JSON object with "move" key
    match = _BARE_JSON_RE.search(response)
    if match:
        return match.group(1).strip()

    return None


def _match_move_to_legal(
    raw_move: str, legal_action_strings: Sequence[str]
) -> str | None:
    """Case-insensitive match of a raw move string to legal actions."""
    lower = raw_move.lower().strip()
    for legal in legal_action_strings:
        # Direct match
        if lower == legal.lower():
            return legal
        # Coordinate match: raw="e5" matches legal="B e5"
        if lower in legal.lower().split():
            return legal
    return None


def parse_response(
    response: str, legal_action_strings: Sequence[str] | None
) -> ParseResult:
    # --- Free-form path ---
    if legal_action_strings is None:
        data = _extract_json(response)  # your JSON extractor
        if data and "clue" in data:
            return ParseResult(
                submission=data,
                raw_action=json.dumps(data),
            )
        return ParseResult(raw_action=response[:200])

    # --- Enumerable path ---
    # Stage 1: try JSON extraction
    raw = _extract_move_from_json(response)
    if raw:
        matched = _match_move_to_legal(raw, legal_action_strings)
        if matched:
            return ParseResult(legal_action=matched, raw_action=raw)

    # Stage 2: scan response text for any legal action string
    for legal in legal_action_strings:
        # Extract the coordinate/keyword part of the action
        parts = legal.split()
        for part in parts:
            if re.search(r'\b' + re.escape(part) + r'\b', response, re.IGNORECASE):
                return ParseResult(legal_action=legal, raw_action=part)

    # Stage 3: nothing found — return failure for retry
    return ParseResult(raw_action=raw)
```

### Parsing tips

- **Always try JSON first**, then fall back to text scanning. LLMs usually follow the JSON format but not always.
- **Case-insensitive matching** is essential — models often change case.
- **Handle special actions** like "PASS" or "STAND" explicitly if the game has them.
- **The fallback scan matters.** When the model ignores the JSON instruction and writes "I'll play e5", the fallback should still find the move.
- **Adapt the JSON key** to your game. Use `"move"` for board games, `"bid"` for auction games, `"action"` for generic games — whatever matches your prompt.
- **For numeric actions** (like bids), parse the integer and validate it's in the legal range.

## Step 5: Create the adapter class and agent function

Wrap your module-level functions into a class that satisfies the `GameHarness` protocol, then call `create_agent_fn`:

```python
from kaggle_environments.core_harness import create_agent_fn

class _MyGameHarness:
    """Adapts module-level harness functions to the GameHarness protocol."""

    def get_legal_moves(self, observation):
        return get_legal_moves(observation)

    def make_prompt(self, observation, move_history,
                    previous_response=None, previous_action=None):
        return generate_prompt(
            observation, move_history, previous_response, previous_action
        )

    def parse_response(self, response, legal_action_strings):
        return parse_response(response, legal_action_strings)

agent_fn = create_agent_fn(_MyGameHarness())
```

The adapter is thin by design — it just bridges the naming convention. `create_agent_fn` handles everything else: model setup, the retry loop, inactive-call guards, move history tracking, and telemetry.

`create_agent_fn` accepts an optional `max_retries` parameter (default 2) — the total number of LLM calls before giving up.

## Step 6: Write tests

Harness tests follow a consistent 4-class structure. Create your test file at the same relative path as the harness, under `tests/`.

For example:
- Harness at `kaggle_environments/envs/open_spiel_env/games/myg/harness.py`
- Tests at `tests/envs/open_spiel_env/games/myg/harness_test.py`

### Test helpers

```python
import unittest
from unittest.mock import MagicMock, patch

from kaggle_environments.core_harness import ParseResult, create_agent_fn

def _make_mock_response(content):
    """Create a mock litellm response."""
    resp = MagicMock()
    resp.usage = MagicMock(prompt_tokens=10, completion_tokens=20)
    resp.choices = [
        MagicMock(
            message=MagicMock(content=content),
            finish_reason="stop",
        )
    ]
    return resp

_ENV = {
    "MODEL_NAME": "test-model",
    "MODEL_PROXY_KEY": "test-key",
    "MODEL_PROXY_URL": "dummy_url",
}
```

### Test class 1: ParseResponseTest

Test the parser in isolation — no mocking needed.

```python
class ParseResponseTest(unittest.TestCase):
    def test_json_block(self):
        response = 'I think e5.\n```json\n{"move": "e5"}\n```'
        result = parse_response(response, ["B e5", "W d4", "B Pass"])
        self.assertEqual(result.legal_action, "B e5")

    def test_case_insensitive(self):
        response = '```json\n{"move": "E5"}\n```'
        result = parse_response(response, ["B e5"])
        self.assertEqual(result.legal_action, "B e5")

    def test_fallback_text_scan(self):
        response = "I'll play at e5 because it controls the center."
        result = parse_response(response, ["B e5", "B d4"])
        self.assertEqual(result.legal_action, "B e5")

    def test_no_match_returns_none(self):
        response = "I have no idea what to do."
        result = parse_response(response, ["B e5", "B d4"])
        self.assertIsNone(result.legal_action)
```

### Test class 2: GeneratePromptTest

Test that prompts contain the right information.

```python
class GeneratePromptTest(unittest.TestCase):
    def test_includes_rules(self):
        obs = {"observationString": "...", "playerId": 0}
        prompt = generate_prompt(obs, [])
        self.assertIn("Rules", prompt)  # or a game-specific keyword

    def test_rethink_suffix(self):
        obs = {"observationString": "...", "playerId": 0}
        prompt = generate_prompt(obs, [], previous_response="old", previous_action="bad")
        self.assertIn("previous response", prompt.lower())
        self.assertIn("bad", prompt)

    def test_no_rethink_on_first_attempt(self):
        obs = {"observationString": "...", "playerId": 0}
        prompt = generate_prompt(obs, [])
        self.assertNotIn("previous response", prompt.lower())
```

### Test class 3: GetLegalMovesTest

```python
class GetLegalMovesTest(unittest.TestCase):
    def test_from_observation(self):
        obs = {"legalActions": [0, 1], "legalActionStrings": ["a1", "b2"]}
        result = get_legal_moves(obs)
        self.assertEqual(result, {0: "a1", 1: "b2"})
```

### Test class 4: AgentIntegrationTest

Test the full harness through `create_agent_fn`, mocking the LLM.

```python
class AgentIntegrationTest(unittest.TestCase):
    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_successful_move(self, mock_litellm):
        mock_litellm.completion.return_value = _make_mock_response(
            '```json\n{"move": "a1"}\n```'
        )
        agent = create_agent_fn(_MyGameHarness())
        obs = {
            "legalActions": [0, 1],
            "legalActionStrings": ["a1", "b2"],
            "currentPlayer": 0,
            "playerId": 0,
            "isTerminal": False,
            "observationString": "...",
        }
        result = agent(obs, {})
        self.assertEqual(result["status"], "OK")
        self.assertEqual(result["submission"], 0)

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_retry_then_succeed(self, mock_litellm):
        mock_litellm.completion.side_effect = [
            _make_mock_response("gibberish"),
            _make_mock_response('```json\n{"move": "a1"}\n```'),
        ]
        agent = create_agent_fn(_MyGameHarness())
        obs = {
            "legalActions": [0, 1],
            "legalActionStrings": ["a1", "b2"],
            "currentPlayer": 0,
            "playerId": 0,
            "isTerminal": False,
            "observationString": "...",
        }
        result = agent(obs, {})
        self.assertEqual(result["status"], "OK")
        self.assertEqual(mock_litellm.completion.call_count, 2)

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_terminal_returns_inactive(self, mock_litellm):
        agent = create_agent_fn(_MyGameHarness())
        obs = {"isTerminal": True, "playerId": 0, "currentPlayer": 0}
        result = agent(obs, {})
        self.assertEqual(result["status"], "INACTIVE")
        mock_litellm.completion.assert_not_called()
```

## Step 7: Write `test_llm_game.py`

Every harness should include a `test_llm_game.py` script that runs a full game locally using a real LLM. This is invaluable for integration testing — it catches issues that unit tests with mocked LLMs miss (bad prompts, unparseable responses, environment interaction bugs).

Place this file next to the harness module:
- OpenSpiel: `kaggle_environments/envs/open_spiel_env/games/<name>/test_llm_game.py`
- Non-OpenSpiel: `kaggle_environments/envs/<name>/test_llm_game.py`

### Template

```python
"""Run a full game with LLM agents for local integration testing.

Usage:
    GEMINI_API_KEY=... uv run python -m \\
        kaggle_environments.envs.<path>.test_llm_game
"""

import json
import os

from kaggle_environments import make


def run_llm_game():
    if not os.environ.get("GEMINI_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        print("Set GEMINI_API_KEY or OPENAI_API_KEY to run this test.")
        print("Example: export GEMINI_API_KEY=your_key")
        return

    # Set up environment variables for the harness
    if "MODEL_NAME" not in os.environ:
        os.environ["MODEL_NAME"] = "gemini-2.5-flash"
    if "MODEL_PROXY_KEY" not in os.environ:
        os.environ["MODEL_PROXY_KEY"] = os.environ.get(
            "GEMINI_API_KEY", os.environ.get("OPENAI_API_KEY", "dummy")
        )
    if "MODEL_PROXY_URL" not in os.environ:
        os.environ["MODEL_PROXY_URL"] = "dummy_url"

    # Create the environment
    env = make("<env_name>", debug=True)

    # Point to the harness agent
    dir_path = os.path.dirname(os.path.abspath(__file__))
    agent_path = os.path.join(dir_path, "harness.py")  # or "harness/main.py"

    # Run the game
    print(f"Running game with LLM agents...")
    env.run([agent_path, agent_path])

    # Print step-by-step results
    print("\n=== GAME STEPS ===")
    for idx, step in enumerate(env.steps):
        print(f"--- Step {idx} ---")
        for agent_idx, agent_state in enumerate(step):
            action = agent_state.action
            status = agent_state.status
            print(f"  Agent {agent_idx} ({status}): {action}")

    # Print final results
    print("\n=== RESULTS ===")
    for i, state in enumerate(env.state):
        print(f"Agent {i}: status={state.status}, reward={state.reward}")

    # Save replay for visualizer debugging
    replay_dir = os.path.join(dir_path, "visualizer", "default", "replays")
    os.makedirs(replay_dir, exist_ok=True)
    replay_path = os.path.join(replay_dir, "test-replay.json")
    with open(replay_path, "w") as f:
        json.dump(env.toJSON(), f)
    print(f"\nReplay saved to {replay_path}")


if __name__ == "__main__":
    run_llm_game()
```

Adapt this template for your game: change `<env_name>`, adjust the number of agents, and add any game-specific configuration or result display.

See `kaggle_environments/envs/word_association/test_llm_game.py` and `kaggle_environments/envs/open_spiel_env/games/go/debug_match_runner.py` for real examples.

## File organization

### OpenSpiel games

```
kaggle_environments/envs/open_spiel_env/games/<name>/
├── __init__.py
├── <name>_proxy.py          # proxy (if not already created)
├── harness.py                # <-- your harness
└── test_llm_game.py          # local LLM integration test

tests/envs/open_spiel_env/games/<name>/
└── harness_test.py
```

### Non-OpenSpiel games

```
kaggle_environments/envs/<name>/
├── harness/
│   ├── __init__.py
│   └── main.py               # <-- your harness
├── test_llm_game.py          # local LLM integration test
└── ...

tests/envs/<name>/
└── harness_test.py
```

## Running tests

```bash
uv sync && uv run pytest tests/envs/open_spiel_env/games/<name>/harness_test.py -v
```

## Checklist

- [ ] Identified the action space type (enumerable, free-form, or mixed)
- [ ] `get_legal_moves` returns `dict[int, str]` or `None` as appropriate
- [ ] Prompt includes: rules, state, move history, player identity, output format
- [ ] Prompt does **not** enumerate the legal moves (let the model derive legality from the state and rules)
- [ ] Prompt has a rethink suffix for retries (uses `previous_response` and `previous_action`)
- [ ] `parse_response` has multi-stage fallback (JSON block -> bare JSON -> text scan)
- [ ] `parse_response` does case-insensitive matching
- [ ] `ParseResult` fields are set correctly (enumerable: `legal_action`; free-form: `submission`)
- [ ] Adapter class wraps functions into `GameHarness` protocol
- [ ] `agent_fn = create_agent_fn(adapter)` is defined at module level
- [ ] Tests cover: parsing, prompt generation, legal moves, and integration with mocked LLM
- [ ] `test_llm_game.py` script runs a full game with real LLM agents
- [ ] Linting passes: `uv run ruff check --fix . && uv run ruff format .`

## Reference files

| File | What to learn from it |
|------|----------------------|
| `kaggle_environments/core_harness.py` | The framework — `GameHarness` protocol, `ParseResult`, `create_agent_fn` |
| `kaggle_environments/envs/open_spiel_env/games/go/harness.py` | Full enumerable harness with coordinate parsing |
| `kaggle_environments/envs/open_spiel_env/games/coin_game/harness.py` | Simple enumerable harness with keyword actions |
| `kaggle_environments/envs/open_spiel_env/games/oshi_zumo/harness.py` | Numeric bid parsing with aggressive fallbacks |
| `kaggle_environments/envs/word_association/harness/main.py` | Mixed harness — free-form clues + enumerable guesses |
| `tests/core_harness_test.py` | Framework test patterns |
| `tests/envs/open_spiel_env/games/go/harness_test.py` | Full harness test suite example |
| `kaggle_environments/envs/word_association/test_llm_game.py` | LLM integration test script (non-OpenSpiel) |
| `kaggle_environments/envs/open_spiel_env/games/go/debug_match_runner.py` | LLM integration test script (OpenSpiel) |
