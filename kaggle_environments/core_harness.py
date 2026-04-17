"""Core harness infrastructure for OpenSpiel LLM agents.

Provides shared functionality for all game-specific harnesses:
- LiteLLM / model proxy setup
- Prompt-with-retry loop
- Injectable telemetry

Game-specific harnesses implement the ``GameHarness`` protocol by providing
three methods:

- ``get_legal_moves(observation)``
- ``make_prompt(observation, move_history, previous_response?, previous_action?)``
- ``parse_response(response, legal_action_strings)``

Use ``create_agent_fn(game_harness)`` to produce a Kaggle-compatible
``agent_fn(obs, config) -> {"submission": int, ...}`` callable.
"""

import dataclasses
import logging
import os
import sys
import time
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


_SEND: TelemetrySendFn = lambda module, **kwargs: None  # noqa: E731


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
# Parse result
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ParseResult:
    """Result of parsing an LLM response.

    Attributes:
        legal_action: The legal action string that was matched, or ``None``
            if no legal move could be matched.
        raw_action: The raw move the model attempted to play (even if
            illegal).  Used to build rethink prompts.
    """

    legal_action: str | None = None
    raw_action: str | None = None


# ---------------------------------------------------------------------------
# Game harness protocol
# ---------------------------------------------------------------------------


class GameHarness(Protocol):
    """Protocol that game-specific harnesses must implement."""

    def get_legal_moves(
        self,
        observation: Mapping[str, Any],
    ) -> dict[int, str]:
        """Return a mapping from legal action id to its string form."""
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
        legal_action_strings: Sequence[str],
    ) -> ParseResult:
        """Extract a move from the LLM response.

        Returns a ``ParseResult``.  If ``legal_action`` is not ``None`` it
        must be one of the strings in ``legal_action_strings``.
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
    """Call the LLM and return ``(response_text, usage_dict)``."""
    _TELEMETRY(calling_llm=True)
    start = time.perf_counter()
    try:
        response = litellm.completion(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            **litellm_kwargs,
        )
        content = response.choices[0].message.content.strip()
        duration = time.perf_counter() - start
        usage = {
            "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
            "completion_tokens": getattr(
                response.usage, "completion_tokens", None,
            ),
        }
        _TELEMETRY(
            llm_call_success=True,
            duration_secs=round(duration, 3),
            **usage,
        )
        return content, usage
    except Exception as exc:
        duration = time.perf_counter() - start
        _TELEMETRY(
            llm_call_exception=str(exc),
            duration_secs=round(duration, 3),
        )
        raise


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
) -> Callable[[Any, dict[str, Any]], dict[str, int]]:
    """Create a Kaggle-compatible agent function from a ``GameHarness``.

    Args:
        game_harness: Game-specific harness implementing the three required
            methods.
        max_retries: Maximum number of prompt attempts (including the initial
            attempt).

    Returns:
        ``agent_fn(obs, config) -> {"submission": int, ...}``
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

        observation = obs if isinstance(obs, dict) else vars(obs)

        # -- legal moves --
        legal_moves = game_harness.get_legal_moves(observation)
        if not legal_moves:
            _TELEMETRY(no_legal_actions=True)
            raise ValueError("No legal actions available.")
        legal_action_strings = list(legal_moves.values())
        legal_actions = list(legal_moves.keys())

        # -- prompt / parse / retry loop --
        previous_response: str | None = None
        previous_action: str | None = None
        last_content = ""
        all_responses: list[str] = []

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
                content, _usage = _call_llm(prompt, model_name, litellm_kwargs)
                last_content = content
                all_responses.append(content)
            except Exception as exc:
                _log.error(
                    "LLM call failed on attempt %d: %s", attempt + 1, exc,
                )
                raise

            result = game_harness.parse_response(content, legal_action_strings)

            if result.legal_action is not None:
                idx = legal_action_strings.index(result.legal_action)
                move_history.append(result.legal_action)
                _TELEMETRY(
                    action_is_legal=True,
                    legal_action={
                        "raw_action": result.raw_action,
                        "legal_action": result.legal_action,
                    },
                )
                return {
                    "submission": legal_actions[idx],
                    "actionString": result.legal_action,
                    "thoughts": last_content,
                    "status": "OK",
                }

            # -- parse failed → prepare rethink --
            _TELEMETRY(
                action_is_legal=False,
                parse_failure={
                    "attempt": attempt + 1,
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
        _TELEMETRY(
            all_attempts_failed=True,
            total_attempts=max_retries,
        )
        raise ValueError(
            f"Failed to parse a legal move after {max_retries} attempts. "
            f"Last response: {last_content[:200]}"
        )

    return agent_fn
