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

# Used when the parser extracted a move but it was illegal. Leads
# with the model's attempted move (the most useful signal) and asks
# for a legal one; we deliberately do NOT include the previous
# response here -- the action string is what matters. A brief tail
# reminds the model to keep using the same JSON format.
RETHINK_ILLEGAL = """
You suggested move "{previous_action}" but this is not a legal move.
Reconsider the rules and the current state, then pick a legal move.

(Keep using the same JSON output format as before -- only the move
value needs to change.)
"""

# Used when the parser couldn't extract a move at all. There's no
# previous_action to show, so we lead with the previous response (last
# 500 chars -- the model's conclusion is what we want to see, not the
# preamble) so the model can see what it tried, then restate the exact
# JSON format with a concrete example so it's unambiguous.
RETHINK_UNPARSABLE = """
Your previous response ended with:
{previous_response}

No JSON answer could be parsed from that. Conclude your response with
your final move as JSON in a ```json fenced block, exactly as the
original instructions required:

```json
{{"move": "<your_move>"}}
```

For example: `{{"move": "<concrete_example_for_this_game>"}}`

The move you choose must also be legal in the current state.
"""


def _build_rethink(previous_response, previous_action):
    # `previous_action` is the parser's `raw_action`. When it's None the
    # parser couldn't extract anything -- the format itself failed, so
    # we lead with the previous response so the model can see what went
    # wrong. When it IS set the action string itself is the most useful
    # signal -- show it back and ask for a legal one.
    #
    # Truncate to the LAST 500 chars (not the first 500). The model
    # almost always puts its answer at the end of the response; the
    # opening tokens are preamble we don't need.
    if not previous_action:
        return RETHINK_UNPARSABLE.format(
            previous_response=(previous_response or "")[-500:],
        )
    return RETHINK_ILLEGAL.format(previous_action=previous_action)
```

### Prompt writing tips

- **Parse the observation string.** If the game has a proxy, `observation["observationString"]` is JSON — parse it and present the state clearly rather than dumping raw JSON.
- **Be specific about the output format.** The LLM's response needs to be parseable. JSON with a clear field name (like `"move"`) works well.
- **Explain the coordinate system.** If the game uses a board, explain notation clearly (e.g., "columns are letters a-h, rows are numbers 1-9").
- **Include rules that affect strategy.** Don't just list rules mechanically — highlight the ones that matter for making good decisions.
- **Don't give strategy advice.** The prompt should explain rules and mechanics, not coach the model on how to play. Saying "capture pieces to win" is fine (that's a rule); saying "control the center early" or "prefer defensive moves" is not (that's strategy). The LLM should reason about strategy on its own from the rules and game state.
- **Keep the rethink suffix concise.** Truncate the previous response to ~500 characters. Include the illegal move attempt and remind the LLM to re-derive a legal move from the state and rules (do *not* paste in a legal-moves list on retry either — same reasoning as above).
- **Branch the rethink on whether `previous_action` was extracted.** Each case wants a different signal back at the model:
  - `previous_action` is set (parser pulled a value from the JSON) → the action string itself is the most useful signal. Lead with `"You suggested move {previous_action} but this is not legal."` Do NOT also include the full previous response — that's noise the model has to skim past to find the actual correction signal. Tail with a brief reminder to keep using the same JSON format.
  - `previous_action` is `None` (parser found nothing) → there's no action string to show. Lead with the previous response (last 500 chars, not first 500 — the model's conclusion is at the end, not the preamble) so the model can see what it tried, then restate the JSON format with a concrete example. Tail with a brief reminder that the move must also be legal.
  - Pick the suffix at render time based on `previous_action`. A single one-size-fits-all suffix that always restates the format teaches the wrong fix on illegal-move retries; a single suffix that never restates the format leaves the model in the dark on unparseable retries.
- **Make the JSON example unambiguous.** In the rethink-unparseable suffix, write the JSON example so the *placeholder* and the *concrete example* are clearly separated. ```` ```json `{"move": "<notation>, e.g. 24/23 24/22"}` ``` ```` reads like the value should literally be that whole string. Instead, use a clean placeholder in the fenced block and put a concrete example on its own line right after:
  ````
  ```json
  {"move": "<your_move>"}
  ```
  For example: `{"move": "24/23 24/22"}`
  ````
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

### The default: `parse_json_action` from `core_harness`

For an enumerable harness, your `parse_response` should be one line —
delegate to `parse_json_action`:

```python
from kaggle_environments.core_harness import ParseResult, parse_json_action


def parse_response(
    response: str, legal_action_strings: Sequence[str],
) -> ParseResult:
    """Trust the model's JSON answer; let the rethink loop fix anything else."""
    return parse_json_action(response, legal_action_strings)
```

`parse_json_action` enforces the design rule that the parser has exactly
one intent surface: the model's structured JSON answer. It uses
`extract_last_json_object` under the hood (last-block-wins, both fenced
and bare), and:

- If the model wrote no JSON answer at all → `legal_action=None,
  raw_action=None` → rethink loop asks for one.
- If the JSON is parseable but the move is illegal → `legal_action=None,
  raw_action=<what_the_model_said>` → rethink loop shows the model what
  it tried and asks it to pick a legal move.
- If the JSON is legal → `legal_action=<matched>, raw_action=<raw>` →
  submit.

**Do not write a prose-scan fallback.** Coord regex, keyword regex,
`response.rfind(legal)`, anything that picks a token mentioned in the
reasoning text — these silently substitute moves the model never
explicitly chose (almost always a rejected option from the prose). The
review-harness skill's "Ghost-fallback / prose-scan rescue" entry has
the empirical case: across 12 harnesses, 7,477 substitutions in one
game's replay archive touched 74% of episodes.

#### Customizing the JSON key

If your game uses a key other than `"move"`, pass `json_key=`:

```python
return parse_json_action(response, legal_action_strings, json_key="bid")
```

#### Customizing the matcher

The default matcher is case-insensitive and strips whitespace. For
games that need notation tolerance, alias handling, or canonicalization
(e.g. `'A7'` → `'a7'`, `'b1-c2'` → `'b1xc2'`), pass `matcher=`:

```python
def _match_move_to_legal(
    raw: str, legal_action_strings: Sequence[str],
) -> str | None:
    """Game-specific normalization, e.g. canonicalize a coord."""
    canonical = _canonicalize(raw)
    return canonical if canonical in set(legal_action_strings) else None


def parse_response(
    response: str, legal_action_strings: Sequence[str],
) -> ParseResult:
    return parse_json_action(
        response, legal_action_strings, matcher=_match_move_to_legal,
    )
```

The matcher operates on the *raw value the model wrote inside the JSON*,
never the full response. This is the one place where game-specific
parsing logic belongs.

#### Free-form harnesses

For free-form turns (where `legal_action_strings is None`), don't use
`parse_json_action` — write the extraction by hand using
`extract_last_json_object`, since the resulting object goes into
`submission` rather than `legal_action`:

```python
import json
from kaggle_environments.core_harness import (
    ParseResult, extract_last_json_object, parse_json_action,
)


def parse_response(response, legal_action_strings):
    if legal_action_strings is None:
        data = extract_last_json_object(response, required_keys=("clue",))
        if data and "clue" in data:
            return ParseResult(
                submission=data,
                raw_action=json.dumps(data),
            )
        return ParseResult(raw_action=response[:200])
    return parse_json_action(response, legal_action_strings)
```

### Last-mention-wins — when extracting the structured answer

When the model writes the structured answer surface more than once — a
draft, then a revision — the **last** occurrence is the intent. Models
almost always enumerate options ("considered a1, then b2, but going
with e5") before stating the final answer. Forward iteration silently
picks the rejected first candidate.

`parse_json_action` and `extract_last_json_object` already handle this
for JSON answers. If your harness uses a different stage-1 surface
(e.g. a tagged `Final Answer: <x>` line), apply the same rule:

| Pattern | Wrong | Right |
|---|---|---|
| Multiple JSON blocks | `_JSON_BLOCK_RE.search(response)` | `parse_json_action(response, legal_action_strings)` (preferred) or `extract_last_json_object(response, required_keys=(...))` |
| Single-match regex extract for an answer tag | `m = r.search(response)` | `matches = list(r.finditer(response)); m = matches[-1] if matches else None` |
| Substring of a fixed answer tag | `response.find("Final Answer:")` | `response.rfind("Final Answer:")` |

**Do not scan the response for unstructured candidates.** Iterating a
regex (`finditer` / `findall`) over the response for coords or keywords,
or iterating `legal_action_strings` and checking `response.rfind(legal)`,
is the prose-scan rescue antipattern even with the "reverse-iter" fix:
the model often discusses several options it doesn't choose, and any
loop over those mentions silently substitutes a non-chosen move. If the
structured answer is missing or illegal, return `legal_action=None` and
let the rethink loop ask the model to fix its format. See the
review-harness "Ghost-fallback / prose-scan rescue" entry for the
empirical impact.

### Parsing tips

- **Prefer `parse_json_action`** over rolling your own. It's the audited single-stage default and removes the temptation to add a "smart" fallback later.
- **JSON is the only intent surface.** No prose fallback. If the model doesn't give a JSON answer, return `legal_action=None` and let the rethink loop ask for one. Prose-scan fallbacks reliably substitute moves the model never chose (rejected options it discussed in its reasoning).
- **Case-insensitive matching** is essential — the default matcher already lowercases and strips whitespace. Pass a custom `matcher=` only when you need notation tolerance, alias handling, or canonicalization.
- **Handle special actions** like "PASS" or "STAND" by including them in `legal_action_strings`; the default matcher will pick them up.
- **Adapt the JSON key** to your game by passing `json_key=`: `"move"` for board games (default), `"bid"` for auction games, `"action"` for generic games — whatever matches your prompt.
- **For numeric actions** (like bids), validate in the matcher: convert raw → int, check membership in the legal-action-encoded integers, and return the canonical legal string.

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
