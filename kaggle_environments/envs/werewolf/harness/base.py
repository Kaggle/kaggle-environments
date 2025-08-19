from abc import ABC, abstractmethod
import os
import re
import json
import traceback
import ast
import datetime
import logging

import litellm
from litellm import completion
from dotenv import load_dotenv
from pydantic import create_model
import tenacity

from kaggle_environments.envs.werewolf.game.records import WerewolfObservationModel
from kaggle_environments.envs.werewolf.game.states import HistoryEntry
from kaggle_environments.envs.werewolf.game.engine import DetailedPhase
from kaggle_environments.envs.werewolf.game.consts import RoleConst, ActionType
from kaggle_environments.envs.werewolf.game.actions import (
    NoOpAction, EliminateProposalAction, HealAction, InspectAction, ChatAction, VoteAction, TargetedAction, BidAction
)

logger = logging.getLogger(__name__)

litellm.drop_params = True

# Load environment variables from a .env file in the same directory
load_dotenv()


def _log_retry_warning(retry_state: tenacity.RetryCallState):
  assert retry_state.outcome is not None
  exception = retry_state.outcome.exception()
  traceback_str = ''.join(traceback.format_exception(exception))
  logging.warning(
      'Attempting retry # %d. Traceback: %s. Retry state: %s',
      retry_state.attempt_number,
      traceback_str,
      retry_state,
  )


def _is_rate_limit_error(exception) -> bool:
    """
    Checks if an exception is a RateLimitError that warrants a context reduction retry.
    This checks for both OpenAI's specific error and the generic HTTP 429 status code.
    """
    is_openai_rate_limit = "RateLimitError" in str(type(exception))
    is_http_429 = hasattr(exception, 'status_code') and exception.status_code == 429
    return is_openai_rate_limit or is_http_429


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
            'RateLimitError detected. Retrying with smaller context. '
            'Reducing event log from %d to %d items.',
            original_count,
            agent_instance._event_log_items_to_keep,
        )

    # Also call the original logging function for general retry logging
    _log_retry_warning(retry_state)


_retry_decorator = tenacity.retry(
    wait=tenacity.wait_random_exponential(min=1, max=60),
    stop=tenacity.stop_after_delay(datetime.timedelta(minutes=15)),
    before_sleep=_log_retry_warning,
    reraise=True,
)


def get_action_subset_fields_schema(model_cls, new_cls_name, fields):
    """
    Creates a new Pydantic model with a subset of fields from an existing model,
    preserving all field metadata (like descriptions, constraints, etc.).
    """
    field_definitions = {
        field: (
            model_cls.model_fields[field].annotation,
            # Pass the entire FieldInfo object, not just the default value
            model_cls.model_fields[field]
        )
        for field in fields
        if field in model_cls.model_fields
    }
    sub_cls = create_model(new_cls_name, **field_definitions)
    subset_schema = sub_cls.model_json_schema()
    return subset_schema


TARGETED_ACTION_SCHEMA = get_action_subset_fields_schema(
    TargetedAction, "TargetedLLMAction", fields=['target_id', 'reasoning', 'perceived_threat_level'])
CHAT_ACTION_SCHEMA = get_action_subset_fields_schema(
    ChatAction, "ChatLLMAction", fields=['message', 'reasoning', 'perceived_threat_level'])
BID_ACTION_SCHEMA = get_action_subset_fields_schema(
    BidAction, "BidLLMAction", fields=['amount', 'reasoning', 'perceived_threat_level'])


TARGETED_ACTION_EXEMPLAR = f"""```json
{json.dumps(dict(reasoning="I chose this target randomly.", target_id="some_player_id", perceived_threat_level="SAFE"))}
```"""

BID_ACTION_EXEMPLAR = f"""```json
{json.dumps(dict(reasoning="I have important information to share, so I am bidding high.", amount=4, perceived_threat_level="UNEASY"))}
```"""

AUDIO_EXAMPLE = 'Say in an spooky whisper: "By the pricking of my thumbs... Something wicked this way comes!"'
AUDIO_EXAMPLE_2 = 'Deliver in a thoughtful tone: "I was stunned. I really suspect John\'s intent of bringing up Tim."' 
AUDIO_EXAMPLE_3 = 'Read this in as fast as possible while remaining intelligible: "My nomination for Jack was purely incidental."' 
AUDIO_EXAMPLE_4 = 'Sound amused and relaxed: "that was a very keen observation, AND a classic wolf play.\n(voice: curious)\nI\'m wondering what the seer might say."' 
CHAT_AUDIO_DICT = {"message": AUDIO_EXAMPLE, "reasoning": "To draw attention to other players ...", "perceived_threat_level": "SAFE"}
CHAT_AUDIO_DICT_2 = {"message": AUDIO_EXAMPLE_2, "reasoning": "This accusation is uncalled for ...", "perceived_threat_level": "DANGER"}
CHAT_AUDIO_DICT_3 = {"message": AUDIO_EXAMPLE_3, "reasoning": "I sense there are some suspicion directed towards me ...", "perceived_threat_level": "UNEASY"}
CHAT_AUDIO_DICT_4 = {"message": AUDIO_EXAMPLE_4, "reasoning": "I am redirecting the attention to other leads ...", "perceived_threat_level": "UNEASY"}
CHAT_ACTION_EXEMPLAR_2 = f"```json\n{json.dumps(CHAT_AUDIO_DICT)}\n```"
CHAT_ACTION_EXEMPLAR_3 = f"```json\n{json.dumps(CHAT_AUDIO_DICT_2)}\n```"
CHAT_ACTION_EXEMPLAR = f"```json\n{json.dumps(CHAT_AUDIO_DICT_3)}\n```"
CHAT_ACTION_EXEMPLAR_4 = f"```json\n{json.dumps(CHAT_AUDIO_DICT_4)}\n```"


CHAT_ACTION_ADDITIONAL_CONSTRAINTS_AUDIO = [
    f'- The "message" will be rendered to TTS and shown to other players, so make sure to control the style, tone, ' 
    f'accent and pace of your message using natural language prompt. e.g.\n{CHAT_ACTION_EXEMPLAR_2}',
    "- Since this is a social game, the script in the message should sound conversational.",
    '- Be Informal: Use contractions (like "it\'s," "gonna"), and simple language.',
    '- Be Spontaneous: Vary your sentence length. It\'s okay to have short, incomplete thoughts or to restart a sentence.',
    '- [Optional] If appropriate, you could add natural sounds in (sound: ...) e.g. (sound: chuckles), or (sound: laughs), etc.',
    '- [Optional] Be Dynamic: A real chat is never monotonous. Use (voice: ...) instructions to constantly and subtly shift the tone to match the words.',
    # f'- Be Expressive: Use a variety of descriptive tones. Don\'t just use happy or sad. Try tones like amused, ' 
    # f'thoughtful, curious, energetic, sarcastic, or conspiratorial. e.g. \n{CHAT_ACTION_EXEMPLAR_4}'
]


CHAT_TEXT_DICT = {"reasoning": "I want to put pressure on Player3 and see how they react. A quiet player is often a werewolf.", "message": "I'm suspicious of Player3. They've been too quiet. What do you all think?", "perceived_threat_level": "UNEASY"}
CHAT_ACTION_EXEMPLAR_TEXT = f"```json\n{json.dumps(CHAT_TEXT_DICT)}\n```"


CHAT_ACTION_ADDITIONAL_CONSTRAINTS_TEXT = [
    '- The "message" will be displayed as text to other players. Focus on being clear, persuasive, and strategic.',
    '- Your goal is to convince others to vote with you. Use logic, point out inconsistencies, or form alliances.',
    '- Refer to players by their ID (e.g., "Player1", "Player3") to avoid ambiguity.',
    '- Keep your messages concise and to the point.'
]


class WerewolfAgentBase(ABC):
    @abstractmethod
    def __call__(self, obs):
        """The instance is meant to be used as callable for kaggle environments."""


DEFAULT_PROMPT_TEMPLATE = """{system_prompt}

### Current Game State
This is the current state of the game. Use this information to make your decision.
{current_state}

### Game Event Log
This is the history of events that have happened so far. You can see what actions were taken and what was said.
{event_log}

### Your Instruction
Based on the game state and event log, please respond to the following instruction.
{instruction}
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




class LLMWerewolfAgent(WerewolfAgentBase):

    def __init__(
            self, model_name: str, agent_config: dict = None, system_prompt: str = "",
            prompt_template: str = DEFAULT_PROMPT_TEMPLATE, kaggle_config=None
    ):
        """This wrapper only support 1 LLM.
        """
        agent_config = agent_config or {}
        decoding_kwargs = agent_config.get("llms", [{}])[0].get('parameters')
        self._decoding_kwargs = decoding_kwargs or {}
        self._kaggle_config = kaggle_config or {}
        self._chat_mode = agent_config.get("chat_mode", "audio")
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._total_cost = 0.
        self._cur_response_cost = 0.
        self._history_entries = []
        self._model_name = model_name
        self._system_prompt = system_prompt
        self._prompt_template = prompt_template
        self._is_vertex_ai = "vertex_ai" in self._model_name

        # This new attribute will track how much history to include for each retry attempt
        self._event_log_items_to_keep = 0

        if self._is_vertex_ai:
            self._decoding_kwargs.update({
                "vertex_ai_project": os.environ.get("VERTEXAI_PROJECT",""),
                "vertex_ai_location": os.environ.get("VERTEXAI_LOCATION",""),
            })

    @property
    def prompt_tokens(self):
        return self._prompt_tokens

    @property
    def completion_tokens(self):
        return self._completion_tokens

    @property
    def total_cost(self):
        return self._total_cost

    def log_token_usage(self) :
        logger.info(
            ", ".join([
                f"*** Total prompt tokens: {self._prompt_tokens}",
                f"total completion_tokens: {self._completion_tokens}",
                f"total query cost: $ {self._total_cost}",
                f"current query cost: $ {self._cur_response_cost}"
            ])
        )

    def __del__(self):
        logger.info(
            f"Instance '{self._model_name}' is being deleted. "
            f"Prompt tokens: '{self._prompt_tokens}' completion_tokens: '{self._completion_tokens}'."
        )

    def query(self, prompt):
        response = completion(
            model=self._model_name,
            messages=[{"content": prompt, "role": "user"}],
            **self._decoding_kwargs
        )
        msg = response["choices"][0]["message"]["content"]
        self._completion_tokens += response['usage']['completion_tokens']
        self._prompt_tokens += response['usage']['prompt_tokens']
        self._cur_response_cost = response._hidden_params["response_cost"]
        self._total_cost += self._cur_response_cost
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
            if '```json' in out:
                # Find the start and end of the json block
                start = out.find('```json') + len('```json')
                end = out.find('```', start)
                json_str = out[start:end].strip()
            elif '```' in out:
                start = out.find('```') + len('```')
                end = out.find('```', start)
                json_str = out[start:end].strip()
            else:
                # If no code block, assume the whole output might be JSON
                json_str = out

            # 2. Clean the JSON string
            # Remove trailing commas from objects and arrays which is a common mistake
            json_str = re.sub(r',\s*([\}\]])', r'\1', json_str)

            # 3. Parse the cleaned string
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                # Fallback for models that output python dicts with single quotes.
                return ast.literal_eval(json_str)
        except Exception:
            # Catch any other unexpected errors during string manipulation or parsing
            error_trace = traceback.format_exc()
            logger.error("An error occurred:\n%s", error_trace)
            logger.error(f"The model out failed to parse is model_name=\"{self._model_name}\".")
            logger.error(f"Failed to parse out={out}")
        return {}


    def render_prompt(self, instruction: str, obs, max_log_items: int = -1):
        """
        Renders the final prompt, optionally truncating the event log
        to include only the last 'max_log_items' events.
        """
        current_state = self.current_state(obs)
        all_descriptions = [
            entry.description
            for entry_list in self._history_entries
            for entry in entry_list
        ]

        # Greedily take the last n items from the event log if a limit is set
        if max_log_items >= 0 and len(all_descriptions) > max_log_items:
            event_log = "\n\n".join(all_descriptions[-max_log_items:])
        else:
            event_log = "\n\n".join(all_descriptions)

        content = {
            "system_prompt": self._system_prompt,
            "current_state": json.dumps(current_state, indent=2, sort_keys=True),
            "event_log": event_log,
            "instruction": instruction
        }
        return self._prompt_template.format(**content)

    @staticmethod
    def current_state(obs):
        raw_obs = obs.get('raw_observation')
        obs_model = WerewolfObservationModel(**raw_obs)
        content = {
            "your_name": obs_model.role,
            "your_team": obs_model.team,
            "your_role_name": obs_model.role,
            "all_player_ids": obs_model.all_player_ids,
            "alive_players": obs_model.alive_players,
            "revealed_players_by_role": obs_model.revealed_players_by_role,
        }
        return content

    @tenacity.retry(
        retry=tenacity.retry_if_exception(_is_rate_limit_error),
        stop=tenacity.stop_after_attempt(5),
        wait=tenacity.wait_random_exponential(min=1, max=60),
        before_sleep=_truncate_and_log_on_retry,
        reraise=True,
    )
    def query_parse(self, instruction, obs):
        prompt = self.render_prompt(instruction=instruction, obs=obs, max_log_items=self._event_log_items_to_keep)
        out = self.query(prompt)
        parsed_out = self.parse(out)
        return parsed_out

    def __call__(self, obs):
        raw_obs = obs.get('raw_observation')
        entries = [HistoryEntry(**entry) for entry in obs.get('new_history_entries_json')]

        self._history_entries.append(entries)

        # Default to NO_OP if observation is missing or agent cannot act
        if not raw_obs or not entries:
            return {"action_type": ActionType.NO_OP.value, "target_idx": None, "message": None}

        self._event_log_items_to_keep = sum(len(entry_list) for entry_list in self._history_entries)

        phase = raw_obs['game_state_phase']
        current_phase = DetailedPhase(raw_obs['phase'])
        my_role = RoleConst(raw_obs['role'])

        my_id = raw_obs['player_id']
        alive_players = raw_obs['alive_players']

        day = raw_obs['day']
        common_args = {"day": day, "phase": phase, "actor_id": my_id}

        action = NoOpAction(**common_args, reasoning="There's nothing to be done.")  # Default action
        instruction = None
        parsed_out = None
        try:
            if current_phase == DetailedPhase.NIGHT_AWAIT_ACTIONS:
                if my_role == RoleConst.WEREWOLF:
                    # Werewolves target other alive players.
                    history_entry = next((entry for entry in entries
                                          if entry.data and entry.data.get('valid_targets')), None)
                    if history_entry:
                        valid_targets = history_entry.data.get('valid_targets')
                        instruction = INSTRUCTION_TEMPLATE.format(**{
                            "role": "You are a Werewolf.",
                            "task": "Vote for a player to eliminate.",
                            "additional_constraints": f"- Valid targets are: `{valid_targets}`.",
                            "json_schema": json.dumps(TARGETED_ACTION_SCHEMA),
                            "exemplar": TARGETED_ACTION_EXEMPLAR
                        })
                        parsed_out = self.query_parse(instruction, obs)
                        action = EliminateProposalAction(**common_args, **parsed_out)

                elif my_role == RoleConst.DOCTOR:
                    # Doctors can save any alive player (including themselves).
                    history_entry = next((entry for entry in entries if entry.data and entry.data.get('valid_candidates')),
                                         None)
                    if history_entry:
                        valid_targets = history_entry.data['valid_candidates']
                        instruction = INSTRUCTION_TEMPLATE.format(**{
                            "role": "You are a Doctor.",
                            "task": "Choose a player to save from the werewolf attack.",
                            "additional_constraints": f'- The "target_id" must be in this list: `{valid_targets}`.',
                            "json_schema": json.dumps(TARGETED_ACTION_SCHEMA),
                            "exemplar": TARGETED_ACTION_EXEMPLAR
                        })
                        parsed_out = self.query_parse(instruction, obs)
                        action = HealAction(**common_args, **parsed_out)

                elif my_role == RoleConst.SEER:
                    # Seers can inspect any alive player.
                    history_entry = next((entry for entry in entries if entry.data and entry.data.get('valid_candidates')),
                                         None)
                    if history_entry:
                        valid_targets = history_entry.data['valid_candidates']
                        instruction = INSTRUCTION_TEMPLATE.format(**{
                            "role": "You are a Seer.",
                            "task": "Choose a player to inspect and reveal their role.",
                            "additional_constraints": f'- The "target_id" must be in this list: `{valid_targets}`.',
                            "json_schema": json.dumps(TARGETED_ACTION_SCHEMA),
                            "exemplar": TARGETED_ACTION_EXEMPLAR
                        })
                        parsed_out = self.query_parse(instruction, obs)
                        action = InspectAction(**common_args, **parsed_out)

            elif current_phase == DetailedPhase.DAY_BIDDING_AWAIT:
                if my_id in alive_players:
                    instruction = INSTRUCTION_TEMPLATE.format(**{
                        "role": "It is bidding time. You can bid to get a chance to speak.",
                        "task": 'Decide how much to bid for a speaking turn. A higher bid increases your chance of speaking. You can bid from 0 to 4.',
                        "additional_constraints": "- The 'amount' must be an integer between 0 and 4.",
                        "json_schema": json.dumps(BID_ACTION_SCHEMA),
                        "exemplar": BID_ACTION_EXEMPLAR
                    })
                    parsed_out = self.query_parse(instruction, obs)
                    action = BidAction(**common_args, **parsed_out)

            elif current_phase == DetailedPhase.DAY_CHAT_AWAIT:
                # All alive players can discuss.
                if my_id in alive_players:
                    if self._chat_mode == 'text':
                        constraints = CHAT_ACTION_ADDITIONAL_CONSTRAINTS_TEXT
                        exemplar = CHAT_ACTION_EXEMPLAR_TEXT
                    elif self._chat_mode == 'audio':  # audio mode
                        constraints = CHAT_ACTION_ADDITIONAL_CONSTRAINTS_AUDIO
                        exemplar = CHAT_ACTION_EXEMPLAR
                    else:
                        raise ValueError(f'Can only select between "text" mode and "audio" mode to prompt the LLM. "{self._chat_mode}" mode detected.')
                    instruction = INSTRUCTION_TEMPLATE.format(**{
                        "role": "It is day time. Participate in the discussion.",
                        "task": 'Discuss with other players to decide who to vote out. Formulate a "message" to persuade others.',
                        "additional_constraints": "\n".join(constraints),
                        "json_schema": json.dumps(CHAT_ACTION_SCHEMA),
                        "exemplar": exemplar
                    })
                    parsed_out = self.query_parse(instruction, obs)
                    action = ChatAction(**common_args, **parsed_out)

            elif current_phase == DetailedPhase.DAY_VOTING_AWAIT:
                # Only alive players can vote. They cannot vote for themselves.
                if my_id in alive_players:
                    valid_targets = [p for p in alive_players if p != my_id]
                    instruction = INSTRUCTION_TEMPLATE.format(**{
                        "role": "It is day time. It is time to vote.",
                        "task": 'Choose a player to exile.',
                        "additional_constraints": f'- The "target_id" must be in this list: `{valid_targets}`.',
                        "json_schema": json.dumps(TARGETED_ACTION_SCHEMA),
                        "exemplar": TARGETED_ACTION_EXEMPLAR
                    })
                    parsed_out = self.query_parse(instruction, obs)
                    action = VoteAction(**common_args, **parsed_out)

            elif current_phase == DetailedPhase.GAME_OVER:
                # No action needed when the game is over.
                action = NoOpAction(**common_args, reasoning="Game over.")
        except Exception:
            error_trace = traceback.format_exc()
            logger.error("An error occurred:\n%s", error_trace)
            logger.error(f"The model failed to act is model_name=\"{self._model_name}\".")
            logger.error(f"instruction=\"{instruction}\"")
            logger.error(f'parsed_out="{parsed_out}"')
        logger.info(f"model_name={self._model_name}")
        logger.info(f'instruction="{instruction}"')
        logger.info(f'action="{action.model_dump()}"')
        self.log_token_usage()
        return action.serialize()
