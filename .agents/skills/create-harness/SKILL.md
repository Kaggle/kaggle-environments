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

Read `kaggle_environments/envs/open_spiel_env/games/checkers/harness.py` —
it's the canonical shape: a single `*_PROMPT_TEMPLATE` constant with
explicit `{field}` placeholders, plus two named rethink templates
(`RETHINK_ILLEGAL`, `RETHINK_UNPARSABLE`) selected by
`render_rethink_suffix` from `core_harness`.

The required pieces:

- A main template that interpolates game state, player identity, move
  history, and a JSON output spec with a concrete `For example:` line.
- `RETHINK_ILLEGAL` — fires when the parser extracted a move that wasn't
  legal. Leads with `{previous_action}` ("You suggested … but this is
  not a legal move"). Does NOT include the previous response.
- `RETHINK_UNPARSABLE` — fires when the parser extracted nothing. Leads
  with `{previous_response}` and restates the JSON format with a clean
  placeholder + concrete example.
- A `generate_prompt` that builds the main prompt and appends
  `render_rethink_suffix(RETHINK_ILLEGAL, RETHINK_UNPARSABLE,
  previous_response, previous_action)`. The helper returns `""` on the
  first attempt, picks the right branch otherwise, and truncates
  `previous_response` to the last 500 chars.

For an imperfect-information variant (per-player observation, custom
`matcher=`), use `dark_hex/harness.py` as the model instead.

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
- **Every concrete claim must trace to a code path.** Before you ship a prompt, take each specific field, rule, or value it references and walk it backwards: (a) which proxy `state_dict()` key produces it? (b) which engine call produces THAT? (c) what params is the env actually loaded with? If you can't trace a claim to a source, it's a phantom — either implement the missing path or delete the claim. The two recurring failure shapes are drift (the field was renamed or the env switched params underneath the prompt) and aspirational copy (the author described a feature they planned but never wired up — e.g. oshi-zumo's docstring once claimed opponent coin counts were "encoded as a hidden suffix" of `move_history` entries, but no code surfaced them). A prompt that lies is worse than one that says less.
- **Cross-check rule claims against the env's *actual* `load_game` params.** Don't trust the game's name or docstring — read the env factory. Gin rummy's prompt described the "Oklahoma variant" because the author assumed the default, but the env loads `gin_rummy` with `oklahoma=false`. Whenever the prompt says "in this variant…" or names a ruleset, confirm the params at the `pyspiel.load_game(name, params)` call site match.
- **Parameterize directional/orientation language on `player_id`.** Sentences like "lower is your goal", "toward the top edge", "first move", or "your stones are at the bottom" are almost always asymmetric — correct for one player and wrong for the other. Render the prompt for `player_id=0` AND `player_id=1` and diff them; any directional text that's byte-identical between the two is probably the bug. Factor through a per-player helper (e.g. `_player_info(player_id) -> (label, code, direction_text)` like dark_hex does) so the asymmetry is in one place. Oshi-zumo shipped with "lower is your goal" baked in once; 7.7% of P0 turns echoed the wrong direction.
- **For multi-phase games, dispatch on an explicit phase identifier, not legals-shape.** When phases share `{Pass, Knock}`-shaped legals (gin rummy's Wall and Layoff) or any other coincidental legal-action signature, a legals-based fallback will silently misroute. Build a `{phase_name: template}` table keyed on the engine's own phase string/enum, assert at construction time that every engine phase is covered, and raise on an unknown phase instead of falling through to a default — a new phase should fail loudly, not silently get the wrong prompt.
- **Read board dimensions from the observation, not hardcoded constants.** Many games support multiple board sizes via env configuration (havannah's `board_size`, dark_hex's `num_rows`/`num_cols`, amazons' build-dependent defaults). A prompt that bakes in `"10x10 grid"`, hardcodes column letters `a–j`, or assumes a fixed coordinate range will silently lie to the model whenever the env is loaded with a different size. Source dims from the parsed `observationString` (proxy `state_dict()` usually exposes `board` plus `num_rows`/`num_cols` or `board_size`), or — as a fallback — deserialize `serializedGameAndState` and read `state.get_game().get_parameters()`. See `amazons/harness.py:113` (`_board_dims`) for the canonical pattern: prefer the actual board grid, fall back to explicit dimension fields, only then a default. Interpolate `num_rows` / `num_cols` into the prompt template (`"on a {num_rows}x{num_cols} grid"`) and derive any coordinate-system text from those dims (e.g. compute the column-letter range from `num_cols` rather than literal `"a-j"`). Render the prompt at every configured size the env supports and verify each renders correctly.
- **Include the full game's move history, not just this agent's moves.** The framework-provided `move_history` argument is *this agent's* past actions only — it omits the opponent's moves entirely. A prompt that uses only this is showing the model half the game. The model needs both sides' moves in order to reason about the position. Sources, in order of preference:
  - The proxy's `state_dict()` exposes a `move_history` (or `action_history`, `moves`, `played_moves`) field covering both players — see `coin_game/harness.py` and `ant_foraging_arena/harness.py` for proxies that surface this and harnesses that render it.
  - If the proxy doesn't expose one, deserialize `serializedGameAndState` and walk `state.history()` / `state.full_history()` to reconstruct it (see `chess/harness.py:36` `_build_pgn_movetext` for a worked example that emits PGN-style movetext from the pyspiel state).
  - Last resort: have the proxy add a `move_history` field. Don't fall back to the per-agent `move_history` argument and call it "history" — that's the "Move history framing wrong" anti-pattern.
  Render it readably (alternating players, grouped per turn for multi-sub-action games like amazons, or PGN-style for chess) and label it accurately — `"Moves played so far this game (both players):"` reads true; `"Moves you have played:"` does not.
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

**Pre-flight check before shipping the default matcher.** Print a handful
of `state.action_to_string(...)` outputs at representative states
(initial, mid-game, post-capture, multi-component turns). For every
non-alphanumeric marker that appears — `*` (backgammon hit), `x`
(checkers/chess capture), trailing `Pass` (backgammon per-die filler),
`-` (move separator), `O-O` (castling) — ask: "would a model naturally
omit or add this?" If yes, the default `_default_match` won't tolerate
it and the rethink loop pays for every drift. Backgammon's audit found
97.3% of episodes forfeited; adding just `*` and `Pass` tolerance via
`matcher=` recovered 34.2% of forfeit turns. Build the tolerant
normalization once, keep it in the matcher.

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

Copy `tests/envs/open_spiel_env/games/checkers/harness_test.py` as the
starting point. It's the canonical layout: a `_make_observation` helper
that builds a harness-style obs dict from a proxy state, and four test
classes that together cover the surface area.

| Class | What it covers |
|---|---|
| `ParseResponseTest` | Parser in isolation. Cover at minimum: fenced JSON, bare JSON, case-insensitive match, illegal-move-returns-raw, prose-only-returns-None (no ghost fallback), multiple-JSON-last-wins. Add a `test_illegal_json_does_not_ghost_substitute_from_prose` regression — the model writes a legal token in prose, then commits to an illegal one in JSON; the parser must NOT silently substitute the prose token. |
| `GeneratePromptTest` | Prompt contents from a real proxy state. Cover: rules keywords present, board orientation, player-asymmetric text differs between `player_id=0` and `player_id=1`, captures/phase flags render correctly, rethink suffixes appear under the right conditions, the JSON example format is unambiguous. If the harness has multi-branch prompts (roles/phases), assert each branch contains its required rules. |
| `GetLegalMovesTest` | Round-trip from `legalActions` + `legalActionStrings`, fallback from `serializedGameAndState`, empty-obs returns `{}`. |
| `AgentIntegrationTest` | Full harness through `create_agent_fn` with `litellm.completion` patched. Cover: successful move, retry-on-bad-parse, raise-after-two-failures, terminal-step-returns-inactive, and a short scripted game (first-legal-each-turn) that round-trips through pyspiel without raising. |

Mock helpers (`_StreamDelta`, `_StreamChoice`, `_StreamChunk`,
`_make_mock_response`, `_ENV`) live at the top of checkers'
`harness_test.py` — copy them verbatim; they're game-independent.

## Step 7: Write `test_llm_game.py`

Every harness should include a `test_llm_game.py` script that runs a
full game locally with a real LLM — catches issues mocked unit tests
miss (bad prompts, unparseable responses, env interaction bugs).

Use `run_llm_game` from `kaggle_environments.local_harness_runner`. The
helper handles API-key discovery, env-var defaults, `--model` /
`--replay-path` CLI flags, game execution, per-step printing, and
replay save. Per-game files are 3 lines plus a docstring:

```python
"""Run a full Checkers game with LLM agents for local integration testing."""
from kaggle_environments.local_harness_runner import run_llm_game

if __name__ == "__main__":
    run_llm_game("open_spiel_checkers", caller_file=__file__)
```

For games that need extra config, pass `configuration=`,
`replay_filename=`, `num_agents=`, or `agent_module=`. See
`havannah/test_llm_game.py` (custom board size), `word_association/test_llm_game.py`
(4 agents, packaged harness, custom post-run output), and
`python_ant_foraging/test_llm_game.py` (custom replay filename) for
real examples. Capture the returned `env` if you need to print
game-specific results after the run.

Place it next to the harness module:
- OpenSpiel: `kaggle_environments/envs/open_spiel_env/games/<name>/test_llm_game.py`
- Non-OpenSpiel: `kaggle_environments/envs/<name>/test_llm_game.py`

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
- [ ] Every concrete field/value/rule in the prompt traces to a code path (no phantom claims; rules match the env's actual `load_game` params)
- [ ] Board dimensions and coordinate-system text are derived from the observation / env config — not hardcoded — and the prompt renders correctly at every supported size
- [ ] Move history shown to the model covers BOTH players (sourced from the proxy's `state_dict()` or reconstructed from `serializedGameAndState`), not just the per-agent `move_history` argument
- [ ] Prompt rendered for `player_id=0` and `player_id=1` — directional/orientation language mirrors correctly
- [ ] Multi-phase games dispatch on the engine's explicit phase identifier, not legals-shape; unknown phase raises instead of silently falling through
- [ ] Prompt has a rethink suffix for retries (uses `previous_response` and `previous_action`)
- [ ] `parse_response` delegates to `parse_json_action` (uses last-mention-wins JSON extraction; no prose-scan fallback — that's the ghost-fallback anti-pattern)
- [ ] `parse_response` does case-insensitive matching (default matcher) or passes a custom `matcher=` for notation tolerance (check `state.action_to_string` outputs for `*`, `x`, trailing `Pass`, `-`, etc. that models drop or add)
- [ ] `ParseResult` fields are set correctly (enumerable: `legal_action`; free-form: `submission`)
- [ ] Adapter class wraps functions into `GameHarness` protocol
- [ ] `agent_fn = create_agent_fn(adapter)` is defined at module level
- [ ] Tests cover: parsing, prompt generation, legal moves, and integration with mocked LLM
- [ ] `test_llm_game.py` script runs a full game with real LLM agents
- [ ] Linting passes: `uv run ruff check --fix . && uv run ruff format .`

## Reference files

| File | What to learn from it |
|------|----------------------|
| `kaggle_environments/core_harness.py` | The framework — `GameHarness` protocol, `ParseResult`, `create_agent_fn`, `parse_json_action`, `render_rethink_suffix` |
| **Canonical templates — start here:** | |
| `kaggle_environments/envs/open_spiel_env/games/checkers/harness.py` | Modern enumerable shape: delegates to `parse_json_action`, branches `render_rethink_suffix`, demonstrates a phase-conditional prompt section (multi-jump continuation) |
| `kaggle_environments/envs/open_spiel_env/games/dark_hex/harness.py` | Same modern shape with a custom `matcher=` callable for notation tolerance; per-player rendering for imperfect-information games |
| `kaggle_environments/envs/word_association/harness/main.py` | Mixed free-form + enumerable harness (non-OpenSpiel) |
| **Test patterns:** | |
| `tests/envs/open_spiel_env/games/checkers/harness_test.py` | Full enumerable test suite (parser stress, prompt assertions, `create_agent_fn` integration with mocked LLM) |
| `tests/core_harness_test.py` | Framework test patterns |
| `kaggle_environments/envs/open_spiel_env/games/checkers/test_llm_game.py` | LLM integration test script (OpenSpiel) |
| `kaggle_environments/envs/word_association/test_llm_game.py` | LLM integration test script (non-OpenSpiel) |
