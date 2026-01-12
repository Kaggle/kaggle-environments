import functools
import json
import logging
import os
import re
import traceback
from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Dict, List, Optional

import litellm
import pyjson5
import tenacity
import yaml
from dotenv import load_dotenv
from litellm import completion, cost_per_token
from litellm.types.utils import Usage
from pydantic import BaseModel, Field

from kaggle_environments.envs.werewolf.game.actions import (
    BidAction,
    ChatAction,
    EliminateProposalAction,
    HealAction,
    InspectAction,
    NoOpAction,
    TargetedAction,
    VoteAction,
)
from kaggle_environments.envs.werewolf.game.consts import ActionType, DetailedPhase, EventName, RoleConst
from kaggle_environments.envs.werewolf.game.records import get_raw_observation
from kaggle_environments.envs.werewolf.game.states import get_last_action_request

_LITELLM_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "litellm_models.yaml")
litellm.config_path = _LITELLM_CONFIG_PATH
with open(_LITELLM_CONFIG_PATH, "r") as _file:
    _MODEL_COST_DICT = yaml.safe_load(_file)
litellm.register_model(_MODEL_COST_DICT)


logger = logging.getLogger(__name__)

litellm.drop_params = True

# Load environment variables from a .env file in the same directory
load_dotenv()


class LLMActionException(Exception):
    """Custom exception to carry context from a failed LLM action."""

    def __init__(self, message, original_exception, raw_out=None, prompt=None):
        super().__init__(message)
        self.original_exception = original_exception
        self.raw_out = raw_out
        self.prompt = prompt

    def __str__(self):
        return f"{super().__str__()} | Raw Output: '{self.raw_out}'"


def _log_retry_warning(retry_state: tenacity.RetryCallState):
    assert retry_state.outcome is not None
    exception = retry_state.outcome.exception()
    traceback_str = "".join(traceback.format_exception(exception))
    if retry_state.attempt_number < 1:
        loglevel = logging.INFO
    else:
        loglevel = logging.WARNING
    logging.log(
        loglevel,
        "Retrying: $s attempt # %s ended with: $s Traceback: %s Retry state: %s",
        retry_state.fn,
        retry_state.attempt_number,
        retry_state.outcome,
        traceback_str,
        retry_state,
    )


def _is_rate_limit_error(exception) -> bool:
    """
    Checks if an exception is a RateLimitError that warrants a context reduction retry.
    This checks for both OpenAI's specific error and the generic HTTP 429 status code.
    """
    is_openai_rate_limit = "RateLimitError" in str(type(exception))
    is_http_429 = hasattr(exception, "status_code") and exception.status_code == 429
    return is_openai_rate_limit or is_http_429


def _is_context_window_exceeded_error(exception) -> bool:
    """"""
    is_error = "ContextWindowExceededError" in str(type(exception))
    return is_error


def _is_json_parsing_error(exception) -> bool:
    out = True if isinstance(exception, pyjson5.Json5Exception) else False
    return out


def _truncate_and_log_on_retry(retry_state: tenacity.RetryCallState):
    """
    Tenacity hook called before a retry. It reduces the context size if a
    RateLimitError was detected.
    """
    # The first argument of the retried method is the class instance 'self'
    agent_instance = retry_state.args[0]

    if _is_rate_limit_error(retry_state.outcome.exception()):
        # Reduce the number of history items to keep by 25% on each attempt
        original_count = agent_instance._event_log_items_to_keep
        agent_instance._event_log_items_to_keep = int(original_count * 0.75)

        logger.warning(
            "ContextWindowExceededError detected. Retrying with smaller context. "
            "Reducing event log from %d to %d itms.",
            original_count,
            agent_instance._event_log_items_to_keep,
        )

    # Also call the original logging function for general retry logging
    _log_retry_warning(retry_state)


def _add_error_entry_on_retry(retry_state: tenacity.RetryCallState):
    last_exception_wrapper = retry_state.outcome.exception()
    if isinstance(last_exception_wrapper, LLMActionException):
        last_exception = last_exception_wrapper.original_exception
        # You can also access the failed output here if needed for logging
        raw_out = last_exception_wrapper.raw_out
        prompt = last_exception_wrapper.prompt
        logger.warning(f"Retrying due to JSON parsing error. Failed output: {raw_out} Failed prompt: {prompt}")
    else:
        last_exception = last_exception_wrapper

    stack_trace_list = traceback.format_exception(last_exception)
    stack_trace_str = "".join(stack_trace_list)
    retry_state.kwargs["error_stack_trace"] = stack_trace_str
    _log_retry_warning(retry_state)


TARGETED_ACTION_SCHEMA = TargetedAction.schema_for_player()
CHAT_ACTION_SCHEMA = ChatAction.schema_for_player()

BID_ACTION_SCHEMA = BidAction.schema_for_player()
BID_ACTION_SCHEMA_REASONING = BidAction.schema_for_player(("perceived_threat_level", "reasoning", "target_id"))


TARGETED_ACTION_EXEMPLAR = f"""```json
{json.dumps(dict(perceived_threat_level="SAFE", reasoning="I chose this target randomly.", target_id="Elliott"))}
```"""

BID_ACTION_EXEMPLAR = f"""```json
{json.dumps(dict(perceived_threat_level="UNEASY", amount=4))}
```"""
BID_ACTION_EXEMPLAR_REASONING = f"""```json
{json.dumps(dict(perceived_threat_level="UNEASY", reasoning="I have important information to share, so I am bidding high.", amount=4))}
```"""

AUDIO_EXAMPLE = 'Say in an spooky whisper: "By the pricking of my thumbs... Something wicked this way comes!"'
AUDIO_EXAMPLE_2 = 'Deliver in a thoughtful tone: "I was stunned. I really suspect John\'s intent of bringing up Tim."'
AUDIO_EXAMPLE_3 = (
    'Read this in as fast as possible while remaining intelligible: "My nomination for Jack was purely incidental."'
)
AUDIO_EXAMPLE_4 = 'Sound amused and relaxed: "that was a very keen observation, AND a classic wolf play.\n(voice: curious)\nI\'m wondering what the seer might say."'
CHAT_AUDIO_DICT = {
    "perceived_threat_level": "SAFE",
    "reasoning": "To draw attention to other players ...",
    "message": AUDIO_EXAMPLE,
}
CHAT_AUDIO_DICT_2 = {
    "perceived_threat_level": "DANGER",
    "reasoning": "This accusation is uncalled for ...",
    "message": AUDIO_EXAMPLE_2,
}
CHAT_AUDIO_DICT_3 = {
    "perceived_threat_level": "UNEASY",
    "reasoning": "I sense there are some suspicion directed towards me ...",
    "message": AUDIO_EXAMPLE_3,
}
CHAT_AUDIO_DICT_4 = {
    "perceived_threat_level": "UNEASY",
    "reasoning": "I am redirecting the attention to other leads ...",
    "message": AUDIO_EXAMPLE_4,
}
CHAT_ACTION_EXEMPLAR_2 = f"```json\n{json.dumps(CHAT_AUDIO_DICT)}\n```"
CHAT_ACTION_EXEMPLAR_3 = f"```json\n{json.dumps(CHAT_AUDIO_DICT_2)}\n```"
CHAT_ACTION_EXEMPLAR = f"```json\n{json.dumps(CHAT_AUDIO_DICT_3)}\n```"
CHAT_ACTION_EXEMPLAR_4 = f"```json\n{json.dumps(CHAT_AUDIO_DICT_4)}\n```"


CHAT_ACTION_ADDITIONAL_CONSTRAINTS_AUDIO = [
    f'- The "message" will be rendered to TTS and shown to other players, so make sure to control the style, tone, '
    f"accent and pace of your message using natural language prompt. e.g.\n{CHAT_ACTION_EXEMPLAR_2}",
    "- Since this is a social game, the script in the message should sound conversational.",
    '- Be Informal: Use contractions (like "it\'s," "gonna"), and simple language.',
    "- Be Spontaneous: Vary your sentence length. It's okay to have short, incomplete thoughts or to restart a sentence.",
    "- [Optional] If appropriate, you could add natural sounds in (sound: ...) e.g. (sound: chuckles), or (sound: laughs), etc.",
    "- [Optional] Be Dynamic: A real chat is never monotonous. Use (voice: ...) instructions to constantly and subtly shift the tone to match the words.",
    # f'- Be Expressive: Use a variety of descriptive tones. Don\'t just use happy or sad. Try tones like amused, '
    # f'thoughtful, curious, energetic, sarcastic, or conspiratorial. e.g. \n{CHAT_ACTION_EXEMPLAR_4}'
]


CHAT_TEXT_DICT = {
    "perceived_threat_level": "UNEASY",
    "reasoning": "I want to put pressure on Hayden and see how they react. A quiet player is often a werewolf.",
    "message": "I'm suspicious of Hayden. They've been too quiet. What do you all think?",
}
CHAT_ACTION_EXEMPLAR_TEXT = f"```json\n{json.dumps(CHAT_TEXT_DICT)}\n```"


CHAT_ACTION_ADDITIONAL_CONSTRAINTS_TEXT = [
    '- The "message" will be displayed as text to other players. Focus on being clear and persuasive',
    "- Your goal is to win the game as a team. Think about how to reach that goal strategically.",
    '- Refer to players strictly by their exact string ID as listed in the "all_player_ids" field of the Current Game State. Do NOT use numeric indices like "Player 0".',
    "- Keep your messages concise and to the point. ",
    '- You can simply say "Pass!", if you have nothing valuable you would like to share.',
]


class WerewolfAgentBase(ABC):
    @abstractmethod
    def __call__(self, obs):
        """The instance is meant to be used as callable for kaggle environments."""


DEFAULT_PROMPT_TEMPLATE = """{system_prompt}

### Current Game State
{current_state}

### Game Timeline
This is the complete, chronological timeline of all public events and your private actions.
{event_log}

### Your Instruction
Based on the game state and event log, please respond to the following instruction.

{instruction}{error_instruction}
"""

INSTRUCTION_TEMPLATE = """#### ROLE
{role}

#### TASK
{task}

#### CONSTRAINTS
- Your response MUST be a single, valid JSON object.
- generate the "reasoning" key first to think through your response. Your "reasoning" is invisible to other players.
{additional_constraints}

#### JSON SCHEMA
Your JSON output must conform to the following schema. Do NOT include this schema in your response.
```json
{json_schema}
```

#### EXAMPLE OUTPUT
Please format your response as a Markdown JSON code block, which should include the fences. Here's a valid example:
{exemplar}
"""


class TokenCost(BaseModel):
    total_tokens: int = 0
    total_costs_usd: float = 0.0
    token_count_history: List[int] = []
    cost_history_usd: List[float] = []

    def update(self, token_count, cost):
        self.total_tokens += token_count
        self.total_costs_usd += cost
        self.token_count_history.append(token_count)
        self.cost_history_usd.append(cost)


class LLMCostTracker(BaseModel):
    model_name: str
    query_token_cost: TokenCost = Field(default_factory=TokenCost)
    prompt_token_cost: TokenCost = Field(default_factory=TokenCost)
    completion_token_cost: TokenCost = Field(default_factory=TokenCost)
    usage_history: List[Usage] = []
    """example item from gemini flash model dump: response.usage = {'completion_tokens': 579, 'prompt_tokens': 1112,
     'total_tokens': 1691, 'completion_tokens_details': {'accepted_prediction_tokens': None, 
     'audio_tokens': None, 'reasoning_tokens': 483, 'rejected_prediction_tokens': None, 
     'text_tokens': 96}, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': None, 
     'text_tokens': 1112, 'image_tokens': None}}"""

    def update(self, response):
        completion_tokens = response["usage"]["completion_tokens"]
        prompt_tokens = response["usage"]["prompt_tokens"]
        response_cost = response._hidden_params["response_cost"]

        try:
            prompt_cost, completion_cost = cost_per_token(
                model=self.model_name, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
            )
            logger.info(f"Used litellm cost for {self.model_name}")
        except Exception as exception:
            raise Exception(
                f"Could not find cost for {self.model_name} in litellm or custom dict. "
                f'You can register the cost in "litellm_models.yaml"'
            ) from exception

        self.query_token_cost.update(token_count=prompt_tokens + completion_tokens, cost=response_cost)
        self.prompt_token_cost.update(token_count=prompt_tokens, cost=prompt_cost)
        self.completion_token_cost.update(token_count=completion_tokens, cost=completion_cost)
        self.usage_history.append(response.usage)


class ActionRegistry:
    """A registry for action handler based on phase and role."""

    def __init__(self):
        self._registry = {}

    def register(self, phase: DetailedPhase, role: Optional[RoleConst] = None):
        """If an action is not role specific, role can be left as None, in which case all roles will be
        pointing to the same handler.
        """

        def decorator(func):
            self._registry.setdefault(phase, {})
            if role is not None:
                self._registry[phase][role] = func
            else:
                for item in RoleConst:
                    self._registry[phase][item] = func

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapper

        return decorator

    def get(self, phase: DetailedPhase, role: RoleConst):
        func = self._registry[phase][role]
        return func


class EventLogKeys:
    PUBLIC_EVENT = "public_event"
    PRIVATE_ACTION = "private_action"


EventLogItem = namedtuple("EventLogItem", ["event_log_key", "day", "phase", "log_item"])


class LLMWerewolfAgent(WerewolfAgentBase):
    action_registry = ActionRegistry()

    def __init__(
        self,
        model_name: str,
        agent_config: dict = None,
        system_prompt: str = "",
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
        kaggle_config=None,
        litellm_model_proxy_kwargs: Optional[Dict[str, str]] = None,
    ):
        """This wrapper only support 1 LLM."""
        agent_config = agent_config or {}
        decoding_kwargs = agent_config.get("llms", [{}])[0].get("parameters")

        self._decoding_kwargs = decoding_kwargs or {}
        # If we use Model Proxy
        if litellm_model_proxy_kwargs is not None:
            self._decoding_kwargs.update(litellm_model_proxy_kwargs)

        self._kaggle_config = kaggle_config or {}
        self._chat_mode = agent_config.get("chat_mode", "audio")
        self._enable_bid_reasoning = agent_config.get("enable_bid_reasoning", False)
        self._cost_tracker = LLMCostTracker(model_name=model_name)

        self._model_name = model_name
        self._system_prompt = system_prompt
        self._prompt_template = prompt_template
        self._is_vertex_ai = "vertex_ai" in self._model_name

        # storing all events including internal and external
        self._event_logs: List[EventLogItem] = []

        # This new attribute will track how much history to include for each retry attempt
        self._event_log_items_to_keep = 0

        if self._is_vertex_ai:
            self._decoding_kwargs.update(
                {
                    "vertex_ai_project": os.environ.get("VERTEXAI_PROJECT", ""),
                    "vertex_ai_location": os.environ.get("VERTEXAI_LOCATION", ""),
                }
            )

    @property
    def cost_tracker(self) -> LLMCostTracker:
        return self._cost_tracker

    def log_token_usage(self):
        cost_history = self._cost_tracker.query_token_cost.cost_history_usd
        query_cost = cost_history[-1] if cost_history else None
        logger.info(
            ", ".join(
                [
                    f"*** Total prompt tokens: {self._cost_tracker.prompt_token_cost.total_tokens}",
                    f"total completion_tokens: {self._cost_tracker.completion_token_cost.total_tokens}",
                    f"total query cost: $ {self._cost_tracker.query_token_cost.total_costs_usd}",
                    f"current query cost: $ {query_cost}",
                ]
            )
        )

    def __del__(self):
        logger.info(
            f"Instance '{self._model_name}' is being deleted. "
            f"Prompt tokens: '{self._cost_tracker.prompt_token_cost.total_tokens}' "
            f"completion_tokens: '{self._cost_tracker.completion_token_cost.total_tokens}'."
        )

    @tenacity.retry(
        retry=tenacity.retry_if_exception(lambda e: isinstance(e, Exception)),
        stop=tenacity.stop_after_attempt(10),
        wait=tenacity.wait_random_exponential(multiplier=1, min=2, max=300),
        reraise=True,
    )
    def query(self, prompt):
        logger.info(f"prompt for {self._model_name}: {prompt}")
        response = completion(
            model=self._model_name, messages=[{"content": prompt, "role": "user"}], **self._decoding_kwargs
        )
        msgs = []
        for item in response.get("choices", []):
            content = item.get("message", {}).get("content", "")
            msgs.append(content)
        msg = "".join(msgs)

        self._cost_tracker.update(response)
        logger.info(f"message from {self._model_name}: {msg}")
        if msg == "":
            raise ValueError(f"Response returned no msg. response={response}")
        return msg

    def parse(self, out: str) -> dict:
        """
        Parses the string output from an LLM into a dictionary.

        This method implements best practices for parsing potentially-malformed
        JSON output from a large language model.
        1. It looks for JSON within Markdown code blocks (```json ... ```).
        2. It attempts to clean the extracted string to fix common LLM mistakes.
        3. It uses a robust JSON parser.
        4. If standard parsing fails, it falls back to a regular expression search
           for the most critical fields as a last resort.

        Args:
            out: The raw string output from the LLM.

        Returns:
            A dictionary parsed from the JSON, or an empty dictionary if all parsing attempts fail.
        """
        try:
            # 1. Extract JSON string from Markdown code blocks
            if "```json" in out:
                # Find the start and end of the json block
                start = out.find("```json") + len("```json")
                end = out.find("```", start)
                json_str = out[start:end].strip()
            elif "```" in out:
                start = out.find("```") + len("```")
                end = out.find("```", start)
                json_str = out[start:end].strip()
            else:
                # If no code block, assume the whole output might be JSON
                json_str = out

            # 2. Clean the JSON string
            # Remove trailing commas from objects and arrays which is a common mistake
            json_str = re.sub(r",\s*([\}\]])", r"\1", json_str)

            # 3. Parse the cleaned string
            return pyjson5.loads(json_str)
        except Exception:
            # Catch any other unexpected errors during string manipulation or parsing
            error_trace = traceback.format_exc()
            logger.error("An error occurred:\n%s", error_trace)
            logger.error(f'The model out failed to parse is model_name="{self._model_name}".')
            logger.error(f"Failed to parse out={out}")
            # reraise the error
            raise

    def render_prompt(self, instruction: str, obs, max_log_items: int = -1, error_stack_trace=None, error_prompt=None):
        """
        Renders the final prompt, optionally truncating the event log
        to include only the last 'max_log_items' events.
        """
        current_state = self.current_state(obs)

        # Greedily take the last n items from the event log if a limit is set
        if 0 <= max_log_items < len(self._event_logs):
            event_logs = self._event_logs[-max_log_items:]
        else:
            event_logs = self._event_logs

        # Build the unified, tagged event logs
        log_parts = []
        day_phase = (None, None)
        for log_key, day, phase, log_item in event_logs:
            if (day, phase) != day_phase:
                day_phase = (day, phase)
                log_parts.append(f"**--- {phase} {day} ---**")
            if log_key == EventLogKeys.PUBLIC_EVENT:
                log_parts.append(f"[EVENT] {log_item.description}")
            elif log_key == EventLogKeys.PRIVATE_ACTION:
                text_parts = [f"[YOUR ACTION & REASONING] You decided to use {type(log_item).__name__} "]
                # account for NOOP
                if log_item.action_field:
                    action_field_item = (
                        f" - {log_item.action_field.capitalize()}: {getattr(log_item, log_item.action_field)}"
                    )
                    text_parts.append(action_field_item)
                text_parts.append(f" - Reasoning: {log_item.reasoning}")
                text_parts.append(f" - Perceived threat level: {log_item.perceived_threat_level}")
                log_parts.append("\n".join(text_parts))

        event_log = "\n\n".join(log_parts)

        error_instruction = ""
        if error_stack_trace:
            error_instruction = (
                f"\n\nYour previous attempt resulted in the following error:\n{error_stack_trace}\n\n{error_prompt}"
            )

        content = {
            "system_prompt": self._system_prompt,
            "current_state": json.dumps(current_state, sort_keys=True),
            "event_log": event_log,
            "instruction": instruction,
            "error_instruction": error_instruction,
        }
        return self._prompt_template.format(**content)

    @staticmethod
    def current_state(obs):
        obs_model = get_raw_observation(obs)
        content = {
            "your_name": obs_model.player_id,
            "your_team": obs_model.team,
            "your_role_name": obs_model.role,
            "all_player_ids": obs_model.all_player_ids,
            "alive_players": obs_model.alive_players,
            "revealed_players": obs_model.revealed_players,
        }
        return content

    @tenacity.retry(
        retry=tenacity.retry_if_exception(_is_context_window_exceeded_error),
        stop=tenacity.stop_after_attempt(5),
        wait=tenacity.wait_random_exponential(multiplier=1, min=2, max=10),
        before_sleep=_truncate_and_log_on_retry,
        reraise=True,
    )
    def render_prompt_query(self, instruction, obs, error_stack_trace=None, error_prompt=None):
        prompt = self.render_prompt(
            instruction=instruction,
            obs=obs,
            max_log_items=self._event_log_items_to_keep,
            error_stack_trace=error_stack_trace,
            error_prompt=error_prompt,
        )
        out = self.query(prompt)
        return out, prompt

    @tenacity.retry(
        retry=tenacity.retry_if_exception(_is_json_parsing_error),
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_random_exponential(multiplier=1, min=2, max=10),
        before_sleep=_add_error_entry_on_retry,
        reraise=True,
    )
    def query_parse(self, instruction, obs, error_stack_trace=None, error_prompt=None):
        raw_out, prompt = self.render_prompt_query(instruction, obs, error_stack_trace, error_prompt)
        try:
            parsed_out = self.parse(raw_out)
            # Add the raw_out and prompt to the output dict
            parsed_out["raw_prompt"] = prompt
            parsed_out["raw_completion"] = raw_out
            return parsed_out
        except pyjson5.Json5Exception as e:
            # Catch the parsing error, wrap it with context, and re-raise.
            # Tenacity will catch this and decide whether to retry.
            raise LLMActionException(
                message="Failed to parse LLM output.", original_exception=e, raw_out=raw_out, prompt=prompt
            )

    @action_registry.register(DetailedPhase.NIGHT_AWAIT_ACTIONS, RoleConst.WEREWOLF)
    def _night_werewolf_vote(self, entries, obs, common_args):
        # Werewolves target other alive players.
        history_entry = get_last_action_request(entries, EventName.VOTE_REQUEST)
        action = NoOpAction(**common_args, reasoning="There's nothing to be done.")
        if history_entry:
            valid_targets = history_entry.data.get("valid_targets")
            instruction = INSTRUCTION_TEMPLATE.format(
                **{
                    "role": "You are a Werewolf.",
                    "task": "Vote for a player to eliminate.",
                    "additional_constraints": f"- Valid targets are: `{valid_targets}`.",
                    "json_schema": json.dumps(TARGETED_ACTION_SCHEMA),
                    "exemplar": TARGETED_ACTION_EXEMPLAR,
                }
            )
            parsed_out = self.query_parse(
                instruction, obs, error_prompt="Your previous attempt failed. Please vote again."
            )
            action = EliminateProposalAction(**common_args, **parsed_out)
        return action

    @action_registry.register(DetailedPhase.NIGHT_AWAIT_ACTIONS, RoleConst.SEER)
    def _night_seer_inspect(self, entries, obs, common_args):
        # Seers can inspect any alive player.
        history_entry = get_last_action_request(entries, EventName.INSPECT_REQUEST)
        action = NoOpAction(**common_args, reasoning="There's nothing to be done.")
        if history_entry:
            valid_targets = history_entry.data["valid_candidates"]
            instruction = INSTRUCTION_TEMPLATE.format(
                **{
                    "role": "You are a Seer.",
                    "task": "Choose a player to inspect and reveal their role.",
                    "additional_constraints": f'- The "target_id" must be in this list: `{valid_targets}`.',
                    "json_schema": json.dumps(TARGETED_ACTION_SCHEMA),
                    "exemplar": TARGETED_ACTION_EXEMPLAR,
                }
            )
            parsed_out = self.query_parse(
                instruction,
                obs,
                error_prompt="Your previous attempt failed. Please choose one player to inspect again.",
            )
            action = InspectAction(**common_args, **parsed_out)
        return action

    @action_registry.register(DetailedPhase.NIGHT_AWAIT_ACTIONS, RoleConst.DOCTOR)
    def _night_doctor_heal(self, entries, obs, common_args):
        action = NoOpAction(**common_args, reasoning="There's nothing to be done.")
        history_entry = get_last_action_request(entries, EventName.HEAL_REQUEST)
        if history_entry:
            valid_targets = history_entry.data["valid_candidates"]
            instruction = INSTRUCTION_TEMPLATE.format(
                **{
                    "role": "You are a Doctor.",
                    "task": "Choose a player to save from the werewolf attack.",
                    "additional_constraints": f'- The "target_id" must be in this list: `{valid_targets}`.',
                    "json_schema": json.dumps(TARGETED_ACTION_SCHEMA),
                    "exemplar": TARGETED_ACTION_EXEMPLAR,
                }
            )
            parsed_out = self.query_parse(
                instruction, obs, error_prompt="Your previous attempt failed. Please choose one player to heal again."
            )
            action = HealAction(**common_args, **parsed_out)
        return action

    @action_registry.register(DetailedPhase.DAY_BIDDING_AWAIT)
    def _day_bid(self, entries, obs, common_args):
        instruction = INSTRUCTION_TEMPLATE.format(
            **{
                "role": "It is bidding time. You can bid to get a chance to speak.",
                "task": "Decide how much to bid for a speaking turn. A higher bid increases your chance of speaking. You can bid from 0 to 4.",
                "additional_constraints": "- The 'amount' must be an integer between 0 and 4.",
                "json_schema": json.dumps(BID_ACTION_SCHEMA),
                "exemplar": BID_ACTION_EXEMPLAR_REASONING if self._enable_bid_reasoning else BID_ACTION_EXEMPLAR,
            }
        )
        parsed_out = self.query_parse(
            instruction, obs, error_prompt="Your previous attempt failed. Please place your bid again."
        )
        action = BidAction(**common_args, **parsed_out)
        return action

    @action_registry.register(DetailedPhase.DAY_CHAT_AWAIT)
    def _day_chat(self, entries, obs, common_args):
        # All alive players can discuss.
        if self._chat_mode == "text":
            constraints = CHAT_ACTION_ADDITIONAL_CONSTRAINTS_TEXT
            exemplar = CHAT_ACTION_EXEMPLAR_TEXT
        elif self._chat_mode == "audio":  # audio mode
            constraints = CHAT_ACTION_ADDITIONAL_CONSTRAINTS_AUDIO
            exemplar = CHAT_ACTION_EXEMPLAR
        else:
            raise ValueError(
                f'Can only select between "text" mode and "audio" mode to prompt the LLM. "{self._chat_mode}" mode detected.'
            )
        instruction = INSTRUCTION_TEMPLATE.format(
            **{
                "role": "It is day time. Participate in the discussion.",
                "task": 'Discuss with other players to decide who to vote out. Formulate a "message" to persuade others.',
                "additional_constraints": "\n".join(constraints),
                "json_schema": json.dumps(CHAT_ACTION_SCHEMA),
                "exemplar": exemplar,
            }
        )
        parsed_out = self.query_parse(
            instruction, obs, error_prompt="Your previous attempt failed. Please prepare your message again."
        )
        action = ChatAction(**common_args, **parsed_out)
        return action

    @action_registry.register(DetailedPhase.DAY_VOTING_AWAIT)
    def _day_vote(self, entries, obs, common_args):
        raw_obs = get_raw_observation(obs)
        alive_players = raw_obs.alive_players
        my_id = raw_obs.player_id
        valid_targets = [p for p in alive_players if p != my_id]
        instruction = INSTRUCTION_TEMPLATE.format(
            **{
                "role": "It is day time. It is time to vote.",
                "task": "Choose a player to exile.",
                "additional_constraints": f'- The "target_id" must be in this list: `{valid_targets}`.',
                "json_schema": json.dumps(TARGETED_ACTION_SCHEMA),
                "exemplar": TARGETED_ACTION_EXEMPLAR,
            }
        )
        parsed_out = self.query_parse(
            instruction, obs, error_prompt="Your previous attempt failed. Please cast your vote again."
        )
        action = VoteAction(**common_args, **parsed_out)
        return action

    def __call__(self, obs):
        raw_obs = get_raw_observation(obs)
        entries = raw_obs.new_player_event_views

        for entry in entries:
            self._event_logs.append(
                EventLogItem(EventLogKeys.PUBLIC_EVENT, day=entry.day, phase=entry.phase, log_item=entry)
            )

        # Default to NO_OP if observation is missing or agent cannot act
        if not raw_obs or not entries:
            return {"action_type": ActionType.NO_OP.value, "target_idx": None, "message": None}

        self._event_log_items_to_keep = len(self._event_logs)

        current_phase = DetailedPhase(raw_obs.detailed_phase)
        my_role = RoleConst(raw_obs.role)

        common_args = {"day": raw_obs.day, "phase": raw_obs.game_state_phase, "actor_id": raw_obs.player_id}

        handler = self.action_registry.get(phase=current_phase, role=my_role)

        start_cost = self._cost_tracker.query_token_cost.total_costs_usd
        start_prompt = self._cost_tracker.prompt_token_cost.total_tokens
        start_completion = self._cost_tracker.completion_token_cost.total_tokens

        try:
            action = handler(self, entries, obs, common_args)
        except LLMActionException as e:
            # Catch the specific exception after all retries have failed
            error_trace = traceback.format_exc()
            logger.error("An LLMActionException occurred after all retries:\n%s", error_trace)
            logger.error(f'The model failed to act is model_name="{self._model_name}".')

            # Now you can access the preserved data!
            action = NoOpAction(
                **common_args,
                reasoning="Fell back to NoOp after multiple parsing failures.",
                error=error_trace,
                raw_completion=e.raw_out,  # <-- Preserved data
                raw_prompt=e.prompt,  # <-- Preserved data
            )

        end_cost = self._cost_tracker.query_token_cost.total_costs_usd
        end_prompt = self._cost_tracker.prompt_token_cost.total_tokens
        end_completion = self._cost_tracker.completion_token_cost.total_tokens

        action.cost = end_cost - start_cost
        action.prompt_tokens = end_prompt - start_prompt
        action.completion_tokens = end_completion - start_completion

        self.log_token_usage()
        # record self action
        self._event_logs.append(
            EventLogItem(EventLogKeys.PRIVATE_ACTION, day=raw_obs.day, phase=raw_obs.game_state_phase, log_item=action)
        )
        return action.serialize()
