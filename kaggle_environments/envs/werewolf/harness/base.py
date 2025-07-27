
from abc import ABC, abstractmethod
import os
import re
import json
import traceback

import litellm
from litellm import completion
from dotenv import load_dotenv
from pydantic import create_model

from kaggle_environments.envs.werewolf.game.records import WerewolfObservationModel
from kaggle_environments.envs.werewolf.game.states import HistoryEntry
from kaggle_environments.envs.werewolf.game.engine import DetailedPhase
from kaggle_environments.envs.werewolf.game.consts import RoleConst, ActionType
from kaggle_environments.envs.werewolf.game.actions import (
    NoOpAction, EliminateProposalAction, HealAction, InspectAction, ChatAction, VoteAction, TargetedAction
)

litellm.drop_params = True

# Load environment variables from a .env file in the same directory
load_dotenv()


def get_action_subset_fields_schema(model_cls, new_cls_name, fields):
    field_definitions = {field: (model_cls.model_fields[field].annotation, model_cls.model_fields[field].default)
                  for field in fields}
    sub_cls = create_model(new_cls_name, **field_definitions)
    subset_schema = sub_cls.model_json_schema()
    return subset_schema


TARGETED_ACTION_SCHEMA = get_action_subset_fields_schema(
    TargetedAction, "TargetedLLMAction", fields=['target_id', 'reasoning'])
CHAT_ACTION_SCHEMA = get_action_subset_fields_schema(
    ChatAction, "ChatLLMAction", fields=['message', 'reasoning'])

TARGETED_ACTION_EXEMPLAR = f"```json\n{dict(reasoning="I chose this target randomly.", target_id="some_player_id")}\n```"
CHAT_ACTION_EXEMPLAR = f"```json\n{dict(reasoning='I need to show that I am a helpful villager.', 
                                        message='I am only a helpful villager. Anyone has any information to share?')}\n```"


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
Here is an example of a valid response format:
```json
{exemplar}
```
"""




class LLMWerewolfAgent(WerewolfAgentBase):

    def __init__(
            self, model_name: str, agent_config: dict = None, system_prompt: str = "",
            prompt_template: str = DEFAULT_PROMPT_TEMPLATE, kaggle_config=None
    ):
        """This wrapper only support 1 LLM.
        """
        decoding_kwargs = agent_config.get("llms", [{}])[0].get('parameters')
        self._decoding_kwargs = decoding_kwargs or {}
        self._kaggle_config = kaggle_config or {}

        self._history_entries = []
        self._model_name = model_name
        self._system_prompt = system_prompt
        self._prompt_template = prompt_template
        self._is_vertex_ai = "vertex_ai" in self._model_name

        if self._is_vertex_ai:
            file_path = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
            with open(file_path, 'r') as file:
                vertex_credentials = json.load(file)
            vertex_credentials_json = json.dumps(vertex_credentials)
            self._decoding_kwargs.update({
                "vertex_ai_project": os.environ["VERTEXAI_PROJECT"],
                "vertex_ai_location": os.environ["VERTEXAI_LOCATION"],
                "vertex_credentials": vertex_credentials_json
            })

    def query(self, prompt):
        response = completion(
            model=self._model_name,
            messages=[{"content": prompt, "role": "user"}],
            **self._decoding_kwargs
        )
        msg = response["choices"][0]["message"]["content"]
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
        json_str = ""
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
            json_str = re.sub(r',\s*([}\]])', r'\1', json_str)

            # 3. Parse the cleaned string
            parsed_json = json.loads(json_str)
            return parsed_json
        except Exception:
            # Catch any other unexpected errors during string manipulation
            traceback.print_exc()
            print(f"out={out}")
        return {}


    def render_prompt(self, instruction: str, obs):
        """
        Renders the final prompt string by injecting context into a template.

        Args:
            instruction: The specific instruction for the current action.
            current_state: A dictionary representing the current game state.
        """
        current_state = self.current_state(obs)
        # Flatten the list of lists of history entries into a single list of descriptions
        all_descriptions = [
            entry.description
            for entry_list in self._history_entries
            for entry in entry_list
        ]
        event_log = "\n\n".join(all_descriptions)

        # The content dictionary holds the values for the template
        content = {
            "system_prompt": self._system_prompt,
            "current_state": json.dumps(current_state, indent=2, sort_keys=True),
            "event_log": event_log,
            "instruction": instruction
        }

        # Render the template with the content
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

    def query_parse(self, instruction, obs):
        prompt = self.render_prompt(instruction=instruction, obs=obs)
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
                        # instruction = (f'You are a Werewolf. Vote for a player to eliminate. Valid targets are: `{valid_targets}`. '
                        #                f'Respond in this JSON schema: `{json.dumps(TARGETED_ACTION_SCHEMA)}`, '
                        #                f'e.g. {TARGETED_ACTION_EXEMPLAR}')
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
                        # instruction = (f'You are a Doctor. Choose a player to save. Valid targets are: `{valid_targets}`. '
                        #                f'Respond in this JSON schema: `{json.dumps(TARGETED_ACTION_SCHEMA)}`, '
                        #                f'e.g. {TARGETED_ACTION_EXEMPLAR}')
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
                        # instruction = (f'You are a Seer. Choose a player to inspect and reveal their role. Valid targets are: `{valid_targets}`. '
                        #                f'Respond in this JSON schema: `{json.dumps(TARGETED_ACTION_SCHEMA)}`, '
                        #                f'e.g. {TARGETED_ACTION_EXEMPLAR}')
                        parsed_out = self.query_parse(instruction, obs)
                        action = InspectAction(**common_args, **parsed_out)

            elif current_phase == DetailedPhase.DAY_DISCUSSION_AWAIT_CHAT:
                # All alive players can discuss.
                if my_id in alive_players:
                    instruction = INSTRUCTION_TEMPLATE.format(**{
                        "role": "It is day time. Participate in the discussion.",
                        "task": 'Discuss with other players to decide who to vote out. Formulate a "message" to persuade others.',
                        "additional_constraints": "",
                        "json_schema": json.dumps(CHAT_ACTION_SCHEMA),
                        "exemplar": CHAT_ACTION_EXEMPLAR
                    })
                    # instruction = (f'It is day. Discuss with other players to decide who to vote out. '
                    #                f'Formulate a message to persuade others. Respond in this JSON schema: `{json.dumps(CHAT_ACTION_SCHEMA)}`,'
                    #                f' e.g. {CHAT_ACTION_EXEMPLAR}')
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
                    # instruction = (f'It is time to vote. Choose a player to exile. Valid targets are: `{valid_targets}`. '
                    #                f'Respond in this JSON schema: `{json.dumps(TARGETED_ACTION_SCHEMA)}`'
                    #                f'e.g. {TARGETED_ACTION_EXEMPLAR}')
                    parsed_out = self.query_parse(instruction, obs)
                    action = VoteAction(**common_args, **parsed_out)

            elif current_phase == DetailedPhase.GAME_OVER:
                # No action needed when the game is over.
                action = NoOpAction(**common_args, reasoning="Game over.")
        except Exception:
            traceback.print_exc()
            print(f"instruction={instruction}")
            print(f"parsed_out={parsed_out}")
        print(action.model_dump())
        return action.serialize()