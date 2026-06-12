"""Core harness infrastructure for OpenSpiel LLM agents.

Provides shared functionality for all game-specific harnesses:
- LiteLLM / model proxy setup
- Prompt-with-retry loop
- Injectable telemetry

Game-specific harnesses implement the ``GameHarness`` protocol by providing
three methods:

- ``get_legal_moves(observation)``
- ``generate_prompt(observation, move_history, previous_response?, previous_action?)``
- ``parse_response(response, legal_action_strings)``

Note that the provided harness requires a ``generate_prompt`` while this uses
``make_prompt`` internally. (The static submission maps between these two)

Use ``create_agent_fn(game_harness)`` to produce a Kaggle-compatible
``agent_fn(obs, config) -> {"submission": int, ...}`` callable.
"""

import dataclasses
import json
import logging
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from typing import Any, Callable, Mapping, Protocol, Sequence

import litellm

litellm.drop_params = True

_log = logging.getLogger(__name__)
_log.setLevel(logging.INFO)
if not _log.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _formatter = logging.Formatter("[core_harness] %(levelname)s: %(message)s")
    _handler.setFormatter(_formatter)
    _log.addHandler(_handler)


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------


class TelemetrySendFn(Protocol):
    """Callback that sends a key-value pair of telemetry data."""

    def __call__(self, module: str, **kwargs: Any) -> None: ...


def _model_proxy_send(module: str, **kwargs: Any) -> None:
    """Default exporter that POSTs to the Kaggle Model Proxy /telemetry/logs."""
    token = os.environ.get("MODEL_PROXY_KEY")
    url = os.environ.get("MODEL_PROXY_URL")
    if not token or not url or url == "dummy_url":
        _log.info("[telemetry] %s %s", module, json.dumps(kwargs, default=str))
        return
    parsed = urllib.parse.urlparse(url)
    host = parsed.netloc or parsed.path.split("/", 1)[0]
    endpoint = f"https://{host}/telemetry/logs"
    payload = json.dumps({module: kwargs}).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=payload,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        urllib.request.urlopen(req, timeout=5).close()
    except Exception as exc:  # noqa: BLE001 - never let telemetry break play
        _log.warning("Failed to post telemetry to Kaggle Model Proxy: %s", exc)


_SEND: TelemetrySendFn = _model_proxy_send


def get_telemetry_logger(module: str) -> Callable[..., None]:
    """Return a module-labelled logger that forwards to the global exporter."""

    def logger(**kwargs: Any) -> None:
        _SEND(module=module, **kwargs)

    return logger


def set_telemetry_exporter(send_fn: TelemetrySendFn) -> None:
    """Inject a custom telemetry backend at runtime."""
    global _SEND
    _SEND = send_fn


_TELEMETRY = get_telemetry_logger(__name__)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------
#
# Reusable prompt templates that game harnesses can use to assemble prompts.
# These mirror the GameArena ``NO_LEGAL_ACTIONS_RETHINK_APPENDED`` /
# ``RETHINK_WITH_ENV_*`` templates and are the format the migrated chess 
# harness was validated against.

# Main game prompt. Placeholders:
#   game_short_name, notation, readable_state_str, move_history,
#   player_name, move_notation, rethink_prompt
BASIC_PROMPT_TEMPLATE = """\
Let's play {game_short_name}. The current game state in {notation} is:
{readable_state_str}
The moves played so far are:
{move_history}
You are playing as player {player_name}.
It is now your turn. Play your strongest move. The move MUST be legal. Reason step by step to come up with your move, then output your final answer in the format "Final Answer: X" where X is your chosen move in {move_notation}.
{rethink_prompt}"""

# Rethink suffix for an unparseable previous response. Placeholder: generation.
BASIC_RETHINK_UNPARSABLE = """\
Your previously suggested move was not parsable.
Please think carefully and generate a new and legal move. Your previous response was:
{generation}
"""

# Rethink suffix for a parseable but illegal previous move. Placeholder: last_move.
BASIC_RETHINK_ILLEGAL = """\
Your previously suggested move was: {last_move}, which is an illegal move.
Please think carefully and generate a new and legal move.
"""


# ---------------------------------------------------------------------------
# JSON extraction helper
# ---------------------------------------------------------------------------


_JSON_FENCE_RE = re.compile(
    r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE,
)
_JSON_DECODER = json.JSONDecoder(strict=False)


def extract_last_json_object(
    response: str,
    *,
    required_keys: Sequence[str] | None = None,
) -> dict[str, Any] | None:
    """Find the LAST parseable JSON object in an LLM response.

    Models routinely write a preliminary answer, reconsider, and write a
    final answer in a later block. Returning the *last* parseable object
    follows the model's last stated intent. Searching the first match
    instead (the common bug this helper exists to prevent) silently picks
    the rejected answer.

    Two strategies are tried in priority order; the FIRST strategy that
    yields any parseable dict wins, and the LAST candidate from that
    strategy is returned:

    1. ``` ```json ... ``` ``` (and unlabelled ``` ``` ``` ```) fenced blocks.
    2. Balanced top-level ``{...}`` substrings, parsed with
       ``json.JSONDecoder.raw_decode`` so nested objects work correctly.

    Args:
        response: The full LLM response text.
        required_keys: If given, candidates lacking every one of these keys
            at the top level are skipped. Lets callers ignore JSON-shaped
            content in the model's reasoning that isn't an action object.

    Returns:
        The matching dict, or ``None`` if no candidate parses.
    """

    def _passes_filter(d: dict[str, Any]) -> bool:
        if required_keys is None:
            return True
        return any(k in d for k in required_keys)

    fenced: list[dict[str, Any]] = []
    for match in _JSON_FENCE_RE.finditer(response):
        try:
            obj = json.loads(match.group(1), strict=False)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and _passes_filter(obj):
            fenced.append(obj)
    if fenced:
        return fenced[-1]

    bare: list[dict[str, Any]] = []
    i = 0
    while True:
        i = response.find("{", i)
        if i < 0:
            break
        try:
            obj, end = _JSON_DECODER.raw_decode(response, i)
        except json.JSONDecodeError:
            i += 1
            continue
        if isinstance(obj, dict) and _passes_filter(obj):
            bare.append(obj)
        i = end
    if bare:
        return bare[-1]

    return None


# ---------------------------------------------------------------------------
# Parse result
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ParseResult:
    """Result of parsing an LLM response.

    Attributes:
        legal_action: The legal action string that was matched, or ``None``
            if no legal move could be matched.  Used for enumerable actions.
        raw_action: The raw move the model attempted to play (even if
            illegal).  Used to build rethink prompts.
        submission: The validated action for free-form action spaces, or
            ``None`` if parsing failed.  Ignored for enumerable actions.
        thoughts: Optional extracted reasoning/thinking text.  When set,
            the core harness records this instead of the full LLM response.
    """

    legal_action: str | None = None
    raw_action: str | None = None
    submission: Any = None
    thoughts: str | None = None


# ---------------------------------------------------------------------------
# Default JSON-only parser
# ---------------------------------------------------------------------------


def parse_json_action(
    response: str,
    legal_action_strings: Sequence[str],
    *,
    json_key: str = "move",
    matcher: Callable[[str, Sequence[str]], str | None] | None = None,
) -> ParseResult:
    """Default ``parse_response`` body for enumerable harnesses.

    The model's stated intent is the value of ``json_key`` in the LAST
    parseable JSON object in the response. This helper:

    1. Extracts that value via ``extract_last_json_object``. If no JSON
       answer is present, returns ``ParseResult(legal_action=None,
       raw_action=None)`` so the rethink loop asks the model for one.
    2. Matches the value against ``legal_action_strings`` (case-insensitive
       and whitespace-stripped by default; pass ``matcher`` for game-
       specific normalization, e.g. notation tolerance or alias handling).
    3. Always returns the raw extracted value as ``raw_action`` -- when the
       JSON is illegal, ``legal_action`` is ``None`` and the rethink prompt
       can show the model what it tried so it can correct itself.

    The helper deliberately has NO prose-scan fallback. Any "guess at
    intent from a coord/keyword/legal-string mentioned in the prose" path
    is the ghost-fallback antipattern: it silently submits moves the model
    never explicitly chose (usually rejected options it discussed in its
    reasoning). The rethink loop is the right way to handle illegal or
    missing structured answers.

    Args:
        response: The full LLM response text.
        legal_action_strings: The legal moves to match against.
        json_key: Top-level JSON field carrying the move (default ``"move"``).
        matcher: Optional ``(raw, legals) -> matched | None`` for game-
            specific normalization. If omitted, exact case-insensitive
            whitespace-stripped matching is used.

    Returns:
        ``ParseResult(legal_action=matched_or_None, raw_action=raw_or_None)``.
    """
    data = extract_last_json_object(response, required_keys=(json_key,))
    if data is None:
        return ParseResult(legal_action=None, raw_action=None)
    raw_value = data.get(json_key)
    if raw_value is None:
        return ParseResult(legal_action=None, raw_action=None)
    raw = str(raw_value).strip()
    if not raw:
        return ParseResult(legal_action=None, raw_action=None)
    if matcher is None:
        matched = _default_match(raw, legal_action_strings)
    else:
        matched = matcher(raw, legal_action_strings)
    return ParseResult(legal_action=matched, raw_action=raw)


def _default_match(raw: str, legals: Sequence[str]) -> str | None:
    """Case-insensitive, whitespace-stripped exact match against legals."""
    target = "".join(raw.split()).lower()
    if not target:
        return None
    for legal in legals:
        if "".join(legal.split()).lower() == target:
            return legal
    return None


# ---------------------------------------------------------------------------
# Rethink suffix
# ---------------------------------------------------------------------------


def render_rethink_suffix(
    illegal_template: str,
    unparsable_template: str,
    previous_response: str | None,
    previous_action: str | None,
) -> str:
    """Pick and render the right rethink suffix for the parse-failure mode.

    The two templates partition the parse-failure space:

    - ``illegal_template`` is used when the parser extracted a value but
      it wasn't a legal move (``previous_action`` is set). Must accept
      the placeholder ``{previous_action}`` -- the action string is the
      most useful signal here; we deliberately do not include the full
      previous response (it dilutes the correction signal).
    - ``unparsable_template`` is used when the parser couldn't extract
      anything (``previous_action`` is ``None``). Must accept the
      placeholder ``{previous_response}`` -- with no action string to
      show, the response itself is what the model needs to see. The
      response is truncated to the LAST 500 characters because models
      almost always put their answer at the end of the response; the
      first 500 chars usually keeps the preamble and drops the answer.

    Returns an empty string when there is no prior attempt to react to
    (``previous_response is None`` and ``previous_action`` is falsy) so
    the caller can unconditionally append the result to the prompt.
    """
    if previous_response is None and not previous_action:
        return ""
    if previous_action:
        return illegal_template.format(previous_action=previous_action)
    return unparsable_template.format(
        previous_response=(previous_response or "")[-500:],
    )


# ---------------------------------------------------------------------------
# Game harness protocol
# ---------------------------------------------------------------------------


class GameHarness(Protocol):
    """Protocol that game-specific harnesses must implement."""

    def get_legal_moves(
        self,
        observation: Mapping[str, Any],
    ) -> dict[int, str] | None:
        """Return a mapping from legal action id to its string form.

        Return ``None`` to indicate a free-form action space where the
        LLM response is not constrained to an enumerable set of moves.
        """
        ...

    def make_prompt(
        self,
        observation: Mapping[str, Any],
        move_history: list[str],
        previous_response: str | None = None,
        previous_action: str | None = None,
    ) -> str:
        """Build the LLM prompt for the current game state.

        On the first attempt ``previous_response`` and ``previous_action`` are
        ``None``.  On rethink attempts, they carry the model's prior output and
        the (possibly illegal) move it tried to play.
        """
        ...

    def parse_response(
        self,
        response: str,
        legal_action_strings: Sequence[str] | None,
        *,
        observation: Mapping[str, Any],
    ) -> ParseResult:
        """Extract a move from the LLM response.

        Returns a ``ParseResult``.  For enumerable actions, if
        ``legal_action`` is not ``None`` it must be one of the strings in
        ``legal_action_strings``.  For free-form actions
        (``legal_action_strings is None``), set ``submission`` on the
        result instead.

        ``observation`` is the current turn's observation dict — always
        passed by ``core_harness``. Most parsers can match the model's
        output against ``legal_action_strings`` alone and may ignore it;
        parsers that genuinely need the env state at parse time (e.g.
        repeated_poker's bet-size soft-matching) forward it to the
        module-level helper that does the work.
        """
        ...


# ---------------------------------------------------------------------------
# LLM invocation
# ---------------------------------------------------------------------------


def _call_llm(
    prompt: str,
    model_name: str,
    litellm_kwargs: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Call the LLM (streaming) and return ``(response_text, call_details)``.

    The response is streamed via ``stream=True`` and assembled before return,
    so callers see the same blocking-style ``(text, details)`` interface.

    ``call_details`` contains per-call usage and metadata::

        {
            "prompt_tokens": int | None,
            "generation_tokens": int | None,
            "reasoning_tokens": int | None,
            "total_tokens": int | None,
            "finish_reason": str | None,
            "duration_secs": float,
            "first_token_secs": float,  # only when any content streamed
        }
    """
    _TELEMETRY(calling_llm=True)
    start = time.perf_counter()
    first_token_secs: float | None = None
    try:
        stream = litellm.completion(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            timeout=3600,
            stream=True,
            stream_options={"include_usage": True},
            **litellm_kwargs,
        )

        content_parts: list[str] = []
        finish_reason: str | None = None
        usage_obj: Any = None
        for chunk in stream:
            choices = getattr(chunk, "choices", None) or []
            if choices:
                delta = getattr(choices[0], "delta", None)
                piece = getattr(delta, "content", None) if delta else None
                if piece:
                    if first_token_secs is None:
                        first_token_secs = time.perf_counter() - start
                    content_parts.append(piece)
                fr = getattr(choices[0], "finish_reason", None)
                if fr:
                    finish_reason = fr
            chunk_usage = getattr(chunk, "usage", None)
            if chunk_usage is not None:
                usage_obj = chunk_usage

        content = "".join(content_parts).strip()
        duration = time.perf_counter() - start

        if not content:
            raise RuntimeError(
                "LLM stream produced no content "
                f"(finish_reason={finish_reason!r}, duration_secs={duration:.3f})"
            )
        if finish_reason is None:
            raise RuntimeError(
                "LLM stream ended without a finish_reason "
                f"(content_length={len(content)}, duration_secs={duration:.3f})"
            )

        ctd = (
            getattr(usage_obj, "completion_tokens_details", None)
            if usage_obj
            else None
        )
        reasoning_tokens = (
            getattr(ctd, "reasoning_tokens", None) if ctd else None
        )

        details: dict[str, Any] = {
            "prompt_tokens": getattr(usage_obj, "prompt_tokens", None),
            "generation_tokens": getattr(
                usage_obj, "completion_tokens", None,
            ),
            "total_tokens": getattr(usage_obj, "total_tokens", None),
            "finish_reason": finish_reason,
            "duration_secs": round(duration, 3),
        }
        if reasoning_tokens is not None:
            details["reasoning_tokens"] = reasoning_tokens
        if first_token_secs is not None:
            details["first_token_secs"] = round(first_token_secs, 3)
        _TELEMETRY(
            llm_call_success=True,
            duration_secs=details["duration_secs"],
            prompt_tokens=details["prompt_tokens"],
            completion_tokens=details["generation_tokens"],
        )
        return content, details
    except Exception as exc:
        duration = time.perf_counter() - start
        _TELEMETRY(
            llm_call_exception=str(exc),
            duration_secs=round(duration, 3),
        )
        raise


def _build_call_detail(
    record: dict[str, Any],
    save_prompt: bool,
    save_response: bool,
) -> dict[str, Any]:
    """Build a single ``call_details`` entry from an internal call record."""
    detail: dict[str, Any] = {
        "model": record["model"],
        "prompt_tokens": record["prompt_tokens"],
        "generation_tokens": record["generation_tokens"],
        "total_tokens": record["total_tokens"],
        "finish_reason": record["finish_reason"],
        "duration_secs": record["duration_secs"],
    }
    if save_response:
        detail["response"] = record["content"]
    if "reasoning_tokens" in record:
        detail["reasoning_tokens"] = record["reasoning_tokens"]
    if "first_token_secs" in record:
        detail["first_token_secs"] = record["first_token_secs"]
    if save_prompt:
        detail["prompt"] = record["prompt"]
    return detail


def _build_generate_return(record: dict[str, Any]) -> dict[str, Any]:
    """Build a legacy ``GenerateReturn``-shaped dict from a call record.

    Omits raw prompts and responses to avoid duplicating large content
    already available via ``call_details`` and ``thoughts``.
    """
    gr: dict[str, Any] = {
        "request_for_logging": {
            "model": record["model"],
        },
        "response_for_logging": {
            "finish_reason": record.get("finish_reason"),
        },
        "generation_tokens": record.get("generation_tokens"),
        "prompt_tokens": record.get("prompt_tokens"),
        "total_tokens": record.get("total_tokens"),
        "duration_success_only_secs": record.get("duration_secs"),
    }
    reasoning = record.get("reasoning_tokens")
    if reasoning is not None:
        gr["reasoning_tokens"] = reasoning
    return gr


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


def _setup_model() -> tuple[str, dict[str, Any]]:
    """Read env vars and return ``(model_name, litellm_kwargs)``."""
    if "MODEL_NAME" not in os.environ:
        raise ValueError("MODEL_NAME environment variable is required.")
    if "MODEL_PROXY_KEY" not in os.environ:
        raise ValueError("MODEL_PROXY_KEY environment variable is required.")
    if "MODEL_PROXY_URL" not in os.environ:
        raise ValueError("MODEL_PROXY_URL environment variable is required.")

    model_name: str = os.environ["MODEL_NAME"]
    litellm_kwargs: dict[str, Any] = {}

    if os.environ["MODEL_PROXY_URL"] != "dummy_url":
        model_name = f"openai/{model_name}"
        litellm_kwargs = {
            "api_base": f"{os.environ['MODEL_PROXY_URL']}/openapi",
            "api_key": os.environ["MODEL_PROXY_KEY"],
            "reasoning_effort": "high",
        }
    elif "gemini" in model_name.lower() and not model_name.startswith("gemini/"):
        model_name = f"gemini/{model_name}"

    return model_name, litellm_kwargs


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------


def create_agent_fn(
    game_harness: GameHarness,
    *,
    max_retries: int = 2,
) -> Callable[[Any, dict[str, Any]], dict[str, Any]]:
    """Create a Kaggle-compatible agent function from a ``GameHarness``.

    Args:
        game_harness: Game-specific harness implementing the three required
            methods.
        max_retries: Maximum number of prompt attempts (including the initial
            attempt).

    Returns:
        ``agent_fn(obs, config) -> {"submission": <action>, ...}``
    """
    # --- closure state (per agent, not per module) ---
    setup_done = False
    model_name: str = ""
    litellm_kwargs: dict[str, Any] = {}
    move_history: list[str] = []

    def agent_fn(
        obs: dict[str, Any] | Any,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        nonlocal setup_done, model_name, litellm_kwargs

        # -- one-time setup --
        if not setup_done:
            model_name, litellm_kwargs = _setup_model()
            _TELEMETRY(
                setup_complete=True,
                model_name=model_name,
            )
            setup_done = True

        save_prompt = bool(config.get("savePrompt", True)) if config else True
        save_response = bool(config.get("saveResponse", True)) if config else True
        include_generate_returns = (
            bool(config.get("includeGenerateReturns", False)) if config else False
        )

        observation = obs if isinstance(obs, dict) else vars(obs)

        # -- inactive-call guard --
        # Runners may invoke the agent when it isn't actually our turn (game
        # over, opponent to move, or the very first probe before the env
        # interpreter has populated state). Return a no-op rather than crash.
        is_terminal = observation.get("isTerminal")
        player_id = observation.get("playerId")
        current_player = observation.get("currentPlayer")
        if is_terminal:
            _TELEMETRY(inactive_call="terminal")
            return {"submission": None, "status": "INACTIVE"}
        # Simultaneous-move games report ``currentPlayer == -2``
        # (pyspiel.PlayerId.SIMULTANEOUS): every player_id is "current" until
        # the round resolves, so skip the not-our-turn check in that case.
        SIMULTANEOUS_PLAYER_ID = -2
        if (
            player_id is not None
            and current_player is not None
            and current_player != SIMULTANEOUS_PLAYER_ID
            and player_id != current_player
        ):
            _TELEMETRY(inactive_call="not_our_turn")
            return {"submission": None, "status": "INACTIVE"}

        # -- legal moves --
        allow_free_form = bool(config.get("freeForm", False)) if config else False
        legal_moves = game_harness.get_legal_moves(observation)
        free_form = legal_moves is None and allow_free_form

        if not legal_moves:
            if free_form:
                legal_action_strings = None
            else:
                # Distinguish "obs not yet populated" (None signals) from
                # a real bug (it IS our turn but the game offers nothing).
                if player_id is None and current_player is None:
                    _log.warning(
                        "core_harness: agent invoked with empty observation "
                        "(keys=%s); returning no-op.",
                        sorted(observation.keys()),
                    )
                    _TELEMETRY(inactive_call="empty_obs")
                    return {"submission": None, "status": "INACTIVE"}
                _TELEMETRY(no_legal_actions=True)
                raise ValueError("No legal actions available.")
        else:
            legal_action_strings = list(legal_moves.values())
            legal_actions = list(legal_moves.keys())

        # -- prompt / parse / retry loop --
        previous_response: str | None = None
        previous_action: str | None = None
        last_content = ""
        all_responses: list[str] = []
        call_records: list[dict[str, Any]] = []
        last_exception: Exception | None = None

        for attempt in range(max_retries):
            if attempt == 0:
                _TELEMETRY(initial_attempt=True)
            else:
                _TELEMETRY(rethinking_attempt={"number": attempt})

            _TELEMETRY(calling_sampler=True)

            prompt = game_harness.make_prompt(
                observation,
                move_history,
                previous_response=previous_response,
                previous_action=previous_action,
            )

            try:
                content, call_details = _call_llm(
                    prompt, model_name, litellm_kwargs,
                )
                last_content = content
                all_responses.append(content)
                call_records.append({
                    "content": content,
                    "prompt": prompt,
                    "model": model_name,
                    **call_details,
                })
                # TODO: drop this fallback once the prod orchestrator
                # template forwards `observation=` to `parse_response`.
                # Until then, old adapters whose `parse_response` signature
                # is `(self, response, legal_action_strings)` would raise
                # TypeError on the kwarg. Narrow the catch to that specific
                # signature mismatch so we don't mask real parser bugs.
                try:
                    result = game_harness.parse_response(
                        content, legal_action_strings, observation=observation,
                    )
                except TypeError as exc:
                    if "observation" not in str(exc):
                        raise
                    result = game_harness.parse_response(
                        content, legal_action_strings,
                    )
                last_exception = None
            except Exception as exc:
                last_exception = exc
                _log.warning(
                    "Attempt %d failed with exception: %s", attempt + 1, exc,
                )
                continue

            # -- check for a valid action --
            matched_submission: Any = None
            action_str: str | None = None

            if free_form and result.submission is not None:
                matched_submission = result.submission
                action_str = str(result.submission)
            elif not free_form and result.legal_action is not None:
                idx = legal_action_strings.index(result.legal_action)
                matched_submission = legal_actions[idx]
                action_str = result.legal_action

            if action_str is not None:
                move_history.append(action_str)
                _TELEMETRY(
                    action_is_legal=True,
                    legal_action={
                        "raw_action": result.raw_action,
                        "legal_action": action_str,
                    },
                )
                action: dict[str, Any] = {
                    "submission": matched_submission,
                    "actionString": action_str,
                    "thoughts": result.thoughts if result.thoughts is not None else last_content,
                    "status": "OK",
                    "call_details": [
                        _build_call_detail(r, save_prompt, save_response)
                        for r in call_records
                    ],
                }
                if include_generate_returns:
                    action["generate_returns"] = [
                        json.dumps(_build_generate_return(r))
                        for r in call_records
                    ]
                return action

            # -- parse failed → prepare rethink --
            # Categorize the failure so dashboards can tell apart:
            #   EMPTY      -> LLM returned no usable content at all
            #   UNPARSABLE -> content present, but parser couldn't extract
            #                 a structured answer (raw_action is None)
            #   ILLEGAL    -> parser extracted something, but it didn't
            #                 match a legal move
            if not (content or "").strip():
                failure_category = "EMPTY"
            elif result.raw_action is None:
                failure_category = "UNPARSABLE"
            else:
                failure_category = "ILLEGAL"
            _TELEMETRY(
                action_is_legal=False,
                parse_failure={
                    "attempt": attempt + 1,
                    "category": failure_category,
                    "raw_action": result.raw_action,
                    "response_preview": content[:200],
                },
            )
            previous_action = result.raw_action
            previous_response = content
            _log.warning(
                "Attempt %d: failed to parse a legal move.", attempt + 1,
            )

        # -- all attempts exhausted --
        # `failure_category` here reports the LAST attempt's failure
        # category (EMPTY / UNPARSABLE / ILLEGAL). Set defensively in
        # case max_retries was 0 and the variable was never assigned.
        _TELEMETRY(
            all_attempts_failed=True,
            total_attempts=max_retries,
            final_failure_category=locals().get("failure_category"),
        )
        if last_exception is not None:
            raise last_exception

        # Fallback is False: -1 as a forfeit signal is an OpenSpiel
        # convention (pyspiel.INVALID_ACTION), and the open_spiel_env spec
        # opts in by declaring `illegalMoveForfeit` with default True. Any
        # other environment can support the parameter by adding it to its
        # own spec and treating submission=-1 as an invalid action.
        illegal_move_forfeit = (
            bool(config.get("illegalMoveForfeit", False)) if config else False
        )
        if illegal_move_forfeit:
            # Mimic game_arena: submit pyspiel.INVALID_ACTION (-1) so the env
            # marks this player INVALID and forfeits them, rather than
            # raising and voiding the whole episode.
            _TELEMETRY(illegal_move_forfeit=True)
            action = {
                "submission": -1,
                "actionString": previous_action,
                "thoughts": last_content,
                "status": (
                    f"Failed to parse a legal move after {max_retries}"
                    " attempts; forfeiting."
                ),
                "call_details": [
                    _build_call_detail(r, save_prompt, save_response)
                    for r in call_records
                ],
            }
            if include_generate_returns:
                action["generate_returns"] = [
                    json.dumps(_build_generate_return(r)) for r in call_records
                ]
            return action

        raise ValueError(
            f"Failed to parse a legal move after {max_retries} attempts. "
            f"End of last response: {last_content[-200:]}"
        )

    return agent_fn
