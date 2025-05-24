import random
from enum import Enum, auto
import math  # Added for math.ceil
from typing import Dict, List, Any, Optional, Tuple, Union
import json

from pydantic import BaseModel, ValidationError, field_validator
# To avoid conflict with PettingZoo's Optional
from typing import Optional as TypingOptional

from pettingzoo.utils.env import AECEnv
from pettingzoo.utils import AgentSelector
from gymnasium import spaces  # type: ignore





# Define Roles


class Role(Enum):
    VILLAGER = auto()
    WEREWOLF = auto()
    DOCTOR = auto()
    SEER = auto()

# Define Game Phases


class Phase(Enum):
    NIGHT_WEREWOLF_VOTE = auto()
    NIGHT_DOCTOR_SAVE = auto()
    NIGHT_SEER_INSPECT = auto()
    DAY_DISCUSSION = auto()
    DAY_VOTING = auto()
    GAME_OVER = auto()

# Define Action Types for the Dict action space


class ActionType(Enum):
    NIGHT_KILL_VOTE = auto()    # Werewolf action
    NIGHT_SAVE_TARGET = auto()  # Doctor action
    NIGHT_INSPECT_TARGET = auto()  # Seer action
    DAY_DISCUSS = auto()        # All alive, during DAY_DISCUSSION
    DAY_LYNCH_VOTE = auto()     # All alive, during DAY_VOTING
    NO_OP = auto()              # For those who don't act in a phase

# Pydantic model for action validation


class ActionModel(BaseModel):
    action_type: int
    target_idx: TypingOptional[int] = None
    message: TypingOptional[str] = None

    @field_validator('action_type')
    def action_type_must_be_valid_enum_value(cls, v_action_type):
        try:
            # Check if it's a valid ActionType enum value
            ActionType(v_action_type)
        except ValueError:
            raise ValueError(
                f"Unknown 'action_type' value: {v_action_type}. Not a member of ActionType enum.")
        return v_action_type


class WerewolfObservationModel(BaseModel):
    role: int  # The agent's own role (e.g., Villager, Werewolf), as an enum value.
    phase: int  # The current game phase (e.g., NIGHT_WEREWOLF_VOTE, DAY_DISCUSSION), as an enum value.
    alive_players: List[int]  # A list indicating the status of each player (1 if alive, 0 if dead), indexed by player_idx.
    known_werewolves: List[int]  # For Werewolves: a list indicating other Werewolves (1 if Werewolf, 0 otherwise), indexed by player_idx. For others: all zeros.
    seer_last_inspection: Tuple[int, int]  # For Seers: (target_player_idx, target_player_role_value) from their last inspection. Defaults to (num_players, 0) if no inspection or invalid.
    discussion_log: str  # A JSON string representing a list of discussion messages from the current day phase. Each message is a dict: {"speaker": "player_X", "message": "text"}.
    last_lynched: int  # The player_idx of the player most recently lynched. Defaults to num_players if no one was lynched.
    last_lynched_player_role: int # The role enum value of the player most recently lynched. Defaults to 0 if no one was lynched or role not revealed.
    last_killed_by_werewolf: int  # The player_idx of the player most recently killed by werewolves. Defaults to num_players if no one was killed.
    last_killed_by_werewolf_role: int # The role enum value of the player most recently killed by werewolves. Defaults to 0 if no one was killed or role not revealed.
    my_unique_name: str  # The unique string identifier for this agent (e.g., "player_0").
    all_player_unique_names: str  # A JSON string representing a list of all player unique names, in order of their player_idx.
    last_day_vote_details: str # A JSON string detailing votes from the *previous* day's lynch voting phase. Format: {"voter_agent_id_str": target_idx_int}.
    current_day_vote_details: str # A JSON string detailing votes cast *so far* in the *current* day's lynch voting phase. Format: {"voter_agent_id_str": target_idx_int}.
    current_night_werewolf_votes: str # For Werewolves: A JSON string detailing votes cast *so far* by werewolves in the *current* night's kill vote. Format: {"voter_agent_id_str": target_idx_int}.
    last_action_feedback: str # A message from the environment indicating the status/validity of the agent's last submitted action.

    def get_human_readable(self) -> Dict[str, Any]:
        """
        Returns a more human-readable version of the observation.
        Enum values are converted to names, JSON strings are parsed,
        and player indices are mapped to names where appropriate.
        """
        player_names = json.loads(self.all_player_unique_names)
        num_players_from_names = len(player_names)

        def get_player_name_or_special(idx: int, special_val_str: str = "N/A") -> str:
            if 0 <= idx < num_players_from_names:
                return player_names[idx]
            # The convention in the environment is to use num_players as an index for "no one" or "invalid".
            elif idx == num_players_from_names:
                return special_val_str
            return f"Invalid_Index_{idx}"

        def get_role_name_or_special(role_val: int, special_val_str: str = "Unknown/None") -> str:
            try:
                if role_val == 0: # Convention for invalid/unknown role in observation
                    return special_val_str
                return Role(role_val).name
            except ValueError: # If role_val is not a valid Role enum member
                return f"Invalid_Role_Value_{role_val}"

        readable_obs = {}

        readable_obs["role"] = Role(self.role).name
        readable_obs["phase"] = Phase(self.phase).name

        readable_obs["alive_players"] = [
            player_names[i] for i, status in enumerate(self.alive_players) if status == 1 and i < num_players_from_names
        ]
        readable_obs["known_werewolves"] = [
            player_names[i] for i, status in enumerate(self.known_werewolves) if status == 1 and i < num_players_from_names
        ]

        seer_target_idx, seer_role_val = self.seer_last_inspection
        readable_obs["seer_last_inspection"] = (
            get_player_name_or_special(seer_target_idx, "No Inspection/Invalid Target"),
            get_role_name_or_special(seer_role_val, "Role Not Revealed/Invalid")
        )

        try:
            readable_obs["discussion_log"] = json.loads(self.discussion_log)
        except json.JSONDecodeError:
            readable_obs["discussion_log"] = f"Error: Could not parse discussion log: '{self.discussion_log}'"

        readable_obs["last_lynched"] = get_player_name_or_special(self.last_lynched, "No One Lynched")
        readable_obs["last_lynched_player_role"] = get_role_name_or_special(self.last_lynched_player_role)

        readable_obs["last_killed_by_werewolf"] = get_player_name_or_special(self.last_killed_by_werewolf, "No One Killed by WW")
        readable_obs["last_killed_by_werewolf_role"] = get_role_name_or_special(self.last_killed_by_werewolf_role)

        readable_obs["my_unique_name"] = self.my_unique_name
        readable_obs["all_player_unique_names"] = player_names # Use the parsed list

        readable_obs["last_day_vote_details"] = self._parse_vote_details_human_readable(self.last_day_vote_details, player_names, num_players_from_names)
        readable_obs["current_day_vote_details"] = self._parse_vote_details_human_readable(self.current_day_vote_details, player_names, num_players_from_names)
        readable_obs["current_night_werewolf_votes"] = self._parse_vote_details_human_readable(self.current_night_werewolf_votes, player_names, num_players_from_names)

        readable_obs["last_action_feedback"] = self.last_action_feedback
        return readable_obs

    def _parse_vote_details_human_readable(self, vote_json_str: str, player_names_list: List[str], num_actual_players: int) -> Union[Dict[str, str], str]:
        """Helper to parse vote JSON strings into human-readable dicts."""
        try:
            # The environment stores votes as {voter_agent_id_str: target_idx_int}
            votes_dict_idx = json.loads(vote_json_str)
            readable_votes = {}
            for voter_name, target_idx_int in votes_dict_idx.items():
                # voter_name is already the player's unique name (agent_id)
                if 0 <= target_idx_int < num_actual_players:
                    target_name = player_names_list[target_idx_int]
                elif target_idx_int == num_actual_players: # Convention for "No one" or "Invalid"
                    target_name = "Invalid Vote Target/No Vote"
                else:
                    target_name = f"Invalid_Target_Index_{target_idx_int}"
                readable_votes[voter_name] = target_name
            return readable_votes
        except json.JSONDecodeError:
            return f"Error: Could not parse vote details: '{vote_json_str}'"
        except Exception as e:
            return f"Error processing vote details: {str(e)}"


class WerewolfEnv(AECEnv):
    metadata = {
        "name": "werewolf_v1.2",  # Updated version to reflect role reveal on lynch
        "is_parallelizable": False,
        "render_modes": ["human"],
    }

    def __init__(self,
                 num_doctors: int = 1, num_seers: int = 1, render_mode: Optional[str] = None,
                 ):
        super().__init__()

        # num_players and related attributes will be initialized in reset()
        self.num_players: int = 0
        self.num_werewolves: int = 0
        self._agent_ids: List[str] = []
        self.index_to_agent_id: Dict[int, str] = {}
        self.agent_id_to_index: Dict[str, int] = {}

        self.action_spaces: Dict[str, spaces.Space] = {}
        self.observation_spaces: Dict[str, spaces.Space] = {}

        assert num_doctors >= 0, f"Number of doctors cannot be negative, got {num_doctors}."
        assert num_seers >= 0, f"Number of seers cannot be negative, got {num_seers}."

        self.num_doctors = num_doctors
        self.num_seers = num_seers
        self.render_mode = render_mode
        self.agents: List[str] = []

        self.player_roles: Dict[str, Role] = {}
        self.alive_agents: List[str] = []
        self.alive_agents_at_night_start: List[str] = []
        self.current_phase = Phase.NIGHT_WEREWOLF_VOTE

        self._werewolf_kill_votes: Dict[str, int] = {}
        self._doctor_save_choices: Dict[str, int] = {}
        self._seer_inspect_choices: Dict[str, int] = {}

        self._seer_inspection_results_for_obs: Dict[str, Tuple[int, int]] = {}
        self._doctor_save_outcomes_for_obs: Dict[str, Tuple[int, int]] = {}

        self._discussion_log_this_round: List[Dict[str, str]] = []
        self._day_lynch_votes: Dict[str, int] = {}

        self._last_lynched_player_idx: Optional[int] = None
        self._last_killed_by_werewolf_idx: Optional[int] = None
        # Store the role value of the lynched player
        self._last_lynched_player_role_val: Optional[int] = None
        # Store the role value of the player killed by WW
        self._last_killed_by_werewolf_role_val: Optional[int] = None
        self._last_day_vote_details: str = json.dumps({})
        self.game_winner_team: Optional[str] = None

        # These will be properly initialized in reset() once possible_agents is known
        self.rewards: Dict[str, float] = {}
        self._cumulative_rewards: Dict[str, float] = {}
        self.terminations: Dict[str, bool] = {}
        self.truncations: Dict[str, bool] = {}
        # infos is initialized by AECEnv's __init__ based on possible_agents.
        # We will re-initialize it in reset.
        self.infos = {
            agent_id: {
                "initial_unique_name": agent_id,
                "last_action_feedback": "No action taken yet in this episode."
            }
            for agent_id in self.agent_ids
        }
        # _agent_selector is initialized by AECEnv's __init__ based on possible_agents.
        # It will be re-initialized in reset with phase-specific actors.

        self.active_player_indices_history: List[Union[int, None]] = [None] # start with one None to align with kaggle environment step parsing
        self.render_step_ind = -1

    @property
    def agent_ids(self) -> List[str]:
        return self._agent_ids

    def _assign_roles(self):
        roles_list = ([Role.WEREWOLF] * self.num_werewolves +
                      [Role.DOCTOR] * self.num_doctors +
                      [Role.SEER] * self.num_seers)
        num_villagers = self.num_players - len(roles_list)
        roles_list.extend([Role.VILLAGER] * num_villagers)
        random.shuffle(roles_list)
        self.player_roles = {agent_id: role for agent_id,
                             role in zip(self.agent_ids, roles_list)}

    def _get_agents_by_role(self, role: Role, only_alive: bool = True) -> List[str]:
        agents_with_role = [
            agent_id for agent_id, r in self.player_roles.items() if r == role
        ]
        if only_alive:
            return [agent_id for agent_id in agents_with_role if agent_id in self.alive_agents]
        return agents_with_role

    def _check_win_conditions(self) -> bool:
        alive_wws = self._get_agents_by_role(Role.WEREWOLF)
        alive_non_wws = [
            p for p in self.alive_agents if self.player_roles[p] != Role.WEREWOLF
        ]
        if not alive_wws:
            self.game_winner_team = "VILLAGE"
            return True
        if len(alive_wws) >= len(alive_non_wws):
            self.game_winner_team = "WEREWOLF"
            return True
        return False

    def _resolve_night_actions(self):
        ww_target_idx: Optional[int] = None
        if self._werewolf_kill_votes:
            counts = {}
            for target_idx in self._werewolf_kill_votes.values():
                counts[target_idx] = counts.get(target_idx, 0) + 1
            if counts:
                max_votes = max(counts.values())
                candidates = [
                    idx for idx, num_votes in counts.items() if num_votes == max_votes]
                if candidates:
                    ww_target_idx = random.choice(candidates)

        actual_doctor_save_idx: Optional[int] = None
        if self._doctor_save_choices:
            first_doctor_id = next(iter(self._doctor_save_choices))
            actual_doctor_save_idx = self._doctor_save_choices[first_doctor_id]

        self._last_killed_by_werewolf_idx = None
        player_killed_this_night_idx: Optional[int] = None
        if ww_target_idx is not None:
            if ww_target_idx != actual_doctor_save_idx:
                player_killed_this_night_idx = ww_target_idx
                self._last_killed_by_werewolf_idx = ww_target_idx

        self._doctor_save_outcomes_for_obs.clear()
        for doc_id, saved_idx in self._doctor_save_choices.items():
            was_targeted = 1 if saved_idx == ww_target_idx and ww_target_idx is not None else 0
            self._doctor_save_outcomes_for_obs[doc_id] = (
                saved_idx, was_targeted)

        self._seer_inspection_results_for_obs.clear()
        for seer_id, inspect_idx in self._seer_inspect_choices.items():
            target_agent_id = self.index_to_agent_id.get(inspect_idx)
            if target_agent_id and target_agent_id in self.player_roles:
                inspected_role_val = self.player_roles[target_agent_id].value
                self._seer_inspection_results_for_obs[seer_id] = (
                    inspect_idx, inspected_role_val)
            else:
                # Use 0 for invalid/unknown role consistently
                self._seer_inspection_results_for_obs[seer_id] = (
                    self.num_players, 0)

        victim_agent_id_for_message: Optional[str] = None
        if player_killed_this_night_idx is not None:
            killed_agent_id = self.index_to_agent_id.get(
                player_killed_this_night_idx)
            if killed_agent_id and killed_agent_id in self.alive_agents:
                victim_agent_id_for_message = killed_agent_id
                self.alive_agents.remove(killed_agent_id)
                self.terminations[killed_agent_id] = True
                self._last_killed_by_werewolf_role_val = self.player_roles[killed_agent_id].value

        if self.render_mode == "human":
            if victim_agent_id_for_message:
                killed_role_name = self.player_roles[victim_agent_id_for_message].name
                print(
                    f"Night victim: {victim_agent_id_for_message} (Role: {killed_role_name})")
            elif ww_target_idx is not None:
                saved_player_name = self.index_to_agent_id.get(
                    ww_target_idx, f"player_{ww_target_idx}")
                print(
                    f"Player {saved_player_name} was targeted by werewolves but SAVED by a Doctor!")
            else:
                print("No one was killed by werewolves this night.")

        self._werewolf_kill_votes.clear()
        self._doctor_save_choices.clear()
        self._seer_inspect_choices.clear()

    def _resolve_day_vote(self):
        self._last_lynched_player_idx = None
        self._last_lynched_player_role_val = None  # Reset for this resolution attempt
        self._last_day_vote_details = json.dumps(self._day_lynch_votes)
        if self._day_lynch_votes:
            counts = {}
            for target_idx in self._day_lynch_votes.values():
                counts[target_idx] = counts.get(target_idx, 0) + 1

            if counts:
                max_votes = 0
                for count_val in counts.values():
                    if count_val > max_votes:
                        max_votes = count_val

                candidates = [
                    idx for idx, num_votes in counts.items() if num_votes == max_votes]
                if len(candidates) == 1:
                    self._last_lynched_player_idx = candidates[0]
                    lynched_agent_id = self.index_to_agent_id.get(
                        self._last_lynched_player_idx)
                    if lynched_agent_id and lynched_agent_id in self.alive_agents:
                        self.alive_agents.remove(lynched_agent_id)
                        self.terminations[lynched_agent_id] = True
                        # Store the role of the lynched player
                        self._last_lynched_player_role_val = self.player_roles[lynched_agent_id].value
                        if self.render_mode == "human":
                            lynched_role_name = self.player_roles[lynched_agent_id].name
                            print(
                                f"Day lynch: {lynched_agent_id} (Role: {lynched_role_name})")
                elif self.render_mode == "human" and candidates:
                    print(
                        f"No one lynched due to a tie in votes (candidates: {[self.index_to_agent_id.get(c,c) for c in candidates]}).")
            elif self.render_mode == "human":
                print("No votes cast during the day, no one lynched.")
        elif self.render_mode == "human":
            print("No votes recorded, no one lynched.")
        self._day_lynch_votes.clear()

    def _transition_phase(self):
        if self.current_phase == Phase.GAME_OVER:
            return

        if self._check_win_conditions():
            self.current_phase = Phase.GAME_OVER
            win_reason = "Unknown win condition."
            if self.game_winner_team == "VILLAGE":
                win_reason = "All werewolves eliminated."
            elif self.game_winner_team == "WEREWOLF":
                win_reason = "Werewolves equal or outnumber non-werewolves."

            if self.render_mode == "human":
                print(f"Termination Condition: {win_reason}")

            for agent_id in self.agent_ids:
                self.terminations[agent_id] = True
                player_role_type = self.player_roles[agent_id]
                is_werewolf = player_role_type == Role.WEREWOLF
                final_reward_value = 0.0
                if (self.game_winner_team == "WEREWOLF" and is_werewolf) or \
                   (self.game_winner_team == "VILLAGE" and not is_werewolf):
                    final_reward_value = 1.0
                else:
                    final_reward_value = -1.0
                self.rewards[agent_id] = final_reward_value
                self._cumulative_rewards[agent_id] += final_reward_value

            self.agents = list(self.agent_ids)
            self._agent_selector.reinit(self.agents)
            if self.agents:
                self.agent_selection = self._agent_selector.reset()
            else:
                self.agent_selection = None
            if self.render_mode == "human":
                print(f"GAME OVER! Winners: {self.game_winner_team}")
            return

        next_actors = []
        if self.current_phase == Phase.NIGHT_WEREWOLF_VOTE:
            self.current_phase = Phase.NIGHT_DOCTOR_SAVE
            next_actors = self._get_agents_by_role(Role.DOCTOR)
            if self.render_mode == "human":
                print("Transitioning to NIGHT_DOCTOR_SAVE.")
        elif self.current_phase == Phase.NIGHT_DOCTOR_SAVE:
            self.current_phase = Phase.NIGHT_SEER_INSPECT
            next_actors = self._get_agents_by_role(Role.SEER)
            if self.render_mode == "human":
                print("Transitioning to NIGHT_SEER_INSPECT.")
        elif self.current_phase == Phase.NIGHT_SEER_INSPECT:
            self._resolve_night_actions()
            if self._check_win_conditions():
                self._transition_phase()
                return
            self.current_phase = Phase.DAY_DISCUSSION
            self._discussion_log_this_round.clear()
            self.alive_agents_at_night_start = list(
                self.alive_agents)  # Update for next night
            next_actors = list(self.alive_agents)
            if self.render_mode == "human":
                print("Transitioning to DAY_DISCUSSION.")
        elif self.current_phase == Phase.DAY_DISCUSSION:
            self.current_phase = Phase.DAY_VOTING
            next_actors = list(self.alive_agents)
            if self.render_mode == "human":
                print("Transitioning to DAY_VOTING.")
        elif self.current_phase == Phase.DAY_VOTING:
            self._resolve_day_vote()
            if self._check_win_conditions():
                self._transition_phase()
                return
            self.current_phase = Phase.NIGHT_WEREWOLF_VOTE
            self.alive_agents_at_night_start = list(
                self.alive_agents)  # Update for this night
            next_actors = self._get_agents_by_role(Role.WEREWOLF)
            if self.render_mode == "human":
                print("Transitioning to NIGHT_WEREWOLF_VOTE.")

        self.agents = list(next_actors)
        self._agent_selector.reinit(self.agents)
        if self.agents:
            self.agent_selection = self._agent_selector.reset()
        else:
            self.agent_selection = None
            self._transition_phase()

    def observe(self, agent_id: str) -> Dict[str, Any]:
        role_val = self.player_roles[agent_id].value

        source_alive_list = self.alive_agents
        if self.current_phase in [Phase.NIGHT_DOCTOR_SAVE, Phase.NIGHT_SEER_INSPECT] and \
           self.player_roles.get(agent_id) in [Role.DOCTOR, Role.SEER]:
            source_alive_list = self.alive_agents_at_night_start

        alive_arr_for_obs = [
            1 if p_id in source_alive_list else 0 for p_id in self.agent_ids]

        known_ww_arr = [0] * self.num_players
        if self.player_roles[agent_id] == Role.WEREWOLF:
            for i, p_id_possible in enumerate(self.agent_ids):
                # Check current alive agents
                if self.player_roles.get(p_id_possible) == Role.WEREWOLF and p_id_possible in self.alive_agents:
                    known_ww_arr[i] = 1

        seer_obs = self._seer_inspection_results_for_obs.get(
            agent_id, (self.num_players, 0))  # Default to (invalid_player_idx, invalid_role_value=0)
        discussion_log_str = json.dumps(self._discussion_log_this_round)

        current_ww_votes_str = json.dumps({})
        if self.player_roles[agent_id] == Role.WEREWOLF and self.current_phase == Phase.NIGHT_WEREWOLF_VOTE:
            current_ww_votes_str = json.dumps(
                {v_id: t_idx for v_id, t_idx in self._werewolf_kill_votes.items()})

        current_day_votes_str = json.dumps({})
        if self.current_phase == Phase.DAY_VOTING:
            current_day_votes_str = json.dumps(
                {v_id: t_idx for v_id, t_idx in self._day_lynch_votes.items()})

        obs_data = {
            "role": role_val,
            "phase": self.current_phase.value,
            "alive_players": alive_arr_for_obs,
            "known_werewolves": known_ww_arr,
            "seer_last_inspection": seer_obs,
            "discussion_log": discussion_log_str,
            "last_lynched": self._last_lynched_player_idx if self._last_lynched_player_idx is not None else self.num_players,
            "last_killed_by_werewolf": self._last_killed_by_werewolf_idx if self._last_killed_by_werewolf_idx is not None else self.num_players,
            "last_killed_by_werewolf_role": self._last_killed_by_werewolf_role_val if self._last_killed_by_werewolf_role_val is not None else 0,
            "my_unique_name": agent_id,
            "all_player_unique_names": json.dumps(self.agent_ids),
            "last_day_vote_details": self._last_day_vote_details,
            "current_day_vote_details": current_day_votes_str,
            "current_night_werewolf_votes": current_ww_votes_str,
            "last_action_feedback": self.infos[agent_id].get("last_action_feedback", "Feedback unavailable."),
            "last_lynched_player_role": self._last_lynched_player_role_val if self._last_lynched_player_role_val is not None else 0,
        }
        # Validate with Pydantic model and return as dict
        # This ensures the structure is correct according to WerewolfObservationModel
        return WerewolfObservationModel.model_validate(obs_data).model_dump(mode="python")

    def step(self, action: Dict[str, Any]):
        acting_agent_id = self.agent_selection
        if acting_agent_id is not None and acting_agent_id in self.agent_id_to_index:
            self.active_player_indices_history.append(
                self.agent_id_to_index[acting_agent_id])
        else:
            self.active_player_indices_history.append(None)

        self.rewards[acting_agent_id] = 0.0
        parsed_action_data: TypingOptional[ActionModel] = None
        # Default, will be updated
        self.infos[acting_agent_id]["last_action_feedback"] = "Action processed."

        if self.terminations[acting_agent_id] or self.truncations[acting_agent_id]:
            self._was_dead_step(action)
            return
        else:
            is_valid_action = True
            feedback_message = "Action processed."
            current_role = self.player_roles[acting_agent_id]
            action_description_for_log = f"Attempted: {str(action)[:100]}"
            try:
                if not isinstance(action, dict):
                    raise ValidationError.from_exception_data(
                        title="ActionFormatError",
                        line_errors=[{"type": "dict_type", "loc": ( # Changed error type
                            "action",), "msg": "Input should be a valid dictionary", "input": action}], # Removed ctx
                    )
                parsed_action_data = ActionModel.model_validate(action)
                action_type_val = parsed_action_data.action_type
                target_idx = parsed_action_data.target_idx
                message_text = parsed_action_data.message
            except ValidationError as e:
                is_valid_action = False
                error_details = []
                for error in e.errors():
                    loc_str = ".".join(
                        map(str, error['loc'])) if error['loc'] else "action"
                    input_val_str = str(error.get('input', 'N/A'))[:50]
                    error_details.append(
                        f"Field '{loc_str}': {error['msg']} (input was: '{input_val_str}')")
                feedback_message = "Error: Invalid action structure. Details: " + \
                    "; ".join(error_details)
                # Use original for logging if parse failed
                action_type_val = action.get("action_type", "N/A_STRUCT_ERROR") if isinstance(action, dict) else "N/A_ACTION_NOT_DICT"
                target_idx = action.get("target_idx", "N/A_STRUCT_ERROR") if isinstance(action, dict) else "N/A_ACTION_NOT_DICT"
                message_text = action.get("message", "N/A_STRUCT_ERROR") if isinstance(action, dict) else "N/A_ACTION_NOT_DICT"

            if is_valid_action and parsed_action_data:
                current_action_enum = ActionType(action_type_val)
                action_description_for_log = f"Attempted: type={current_action_enum.name}, target={target_idx}, msg='{str(message_text)[:20]}...'"

                required_target = [
                    ActionType.NIGHT_KILL_VOTE, ActionType.NIGHT_SAVE_TARGET,
                    ActionType.NIGHT_INSPECT_TARGET, ActionType.DAY_LYNCH_VOTE
                ]
                if current_action_enum in required_target:
                    if target_idx is None:
                        is_valid_action = False
                        feedback_message = f"Error: 'target_idx' is required for {current_action_enum.name}."
                    elif not (0 <= target_idx < self.num_players):
                        is_valid_action = False
                        feedback_message = f"Error: 'target_idx' {target_idx} out of bounds [0, {self.num_players - 1}]."

                if current_action_enum == ActionType.DAY_DISCUSS and message_text is None:
                    is_valid_action = False
                    feedback_message = "Error: 'message' is required for DAY_DISCUSS."

                if not is_valid_action:  # Failed conditional field checks
                    pass  # feedback_message already set
                elif current_action_enum == ActionType.NO_OP:
                    action_description_for_log = "NO_OP"
                    feedback_message = "Action: NO_OP processed."
                elif self.current_phase == Phase.NIGHT_WEREWOLF_VOTE and current_action_enum == ActionType.NIGHT_KILL_VOTE:
                    if current_role == Role.WEREWOLF:  # target_idx already validated for range
                        target_agent_id = self.index_to_agent_id[target_idx]
                        if target_agent_id in self.alive_agents_at_night_start and self.player_roles[target_agent_id] != Role.WEREWOLF:
                            self._werewolf_kill_votes[acting_agent_id] = target_idx
                            action_description_for_log = f"WW_VOTE_KILL: {target_agent_id}"
                            feedback_message = f"Action: Voted to kill {target_agent_id}."
                        else:
                            is_valid_action = False
                            feedback_message = f"Invalid target for NIGHT_KILL_VOTE: {target_agent_id}. Target must be alive (at night start) and not a Werewolf."
                    else:
                        is_valid_action = False
                        feedback_message = "Invalid action: Role must be Werewolf for NIGHT_KILL_VOTE."
                elif self.current_phase == Phase.NIGHT_DOCTOR_SAVE and current_action_enum == ActionType.NIGHT_SAVE_TARGET:
                    if current_role == Role.DOCTOR:  # target_idx already validated
                        target_agent_id = self.index_to_agent_id[target_idx]
                        if target_agent_id in self.alive_agents_at_night_start:
                            self._doctor_save_choices[acting_agent_id] = target_idx
                            action_description_for_log = f"DOC_SAVE: {target_agent_id}"
                            feedback_message = f"Action: Chose to save {target_agent_id}."
                        else:
                            is_valid_action = False
                            feedback_message = f"Invalid target for NIGHT_SAVE_TARGET: {target_agent_id}. Target must be alive (at night start)."
                    else:
                        is_valid_action = False
                        feedback_message = "Invalid action: Role must be Doctor for NIGHT_SAVE_TARGET."
                elif self.current_phase == Phase.NIGHT_SEER_INSPECT and current_action_enum == ActionType.NIGHT_INSPECT_TARGET:
                    if current_role == Role.SEER:  # target_idx already validated
                        target_agent_id = self.index_to_agent_id[target_idx]
                        if target_agent_id in self.alive_agents_at_night_start:
                            self._seer_inspect_choices[acting_agent_id] = target_idx
                            action_description_for_log = f"SEER_INSPECT: {target_agent_id}"
                            feedback_message = f"Action: Chose to inspect {target_agent_id}."
                        else:
                            is_valid_action = False
                            feedback_message = f"Invalid target for NIGHT_INSPECT_TARGET: {target_agent_id}. Target must be alive (at night start)."
                    else:
                        is_valid_action = False
                        feedback_message = "Invalid action: Role must be Seer for NIGHT_INSPECT_TARGET."
                elif self.current_phase == Phase.DAY_DISCUSSION and current_action_enum == ActionType.DAY_DISCUSS:
                    self._discussion_log_this_round.append(
                        {"speaker": acting_agent_id, "message": message_text})  # message_text validated
                    action_description_for_log = f"DISCUSS: \"{message_text[:30]}...\""
                    feedback_message = "Action: Discussion message sent."
                elif self.current_phase == Phase.DAY_VOTING and current_action_enum == ActionType.DAY_LYNCH_VOTE:
                    # target_idx validated
                    target_agent_id = self.index_to_agent_id[target_idx]
                    if target_agent_id in self.alive_agents and target_agent_id != acting_agent_id:
                        self._day_lynch_votes[acting_agent_id] = target_idx
                        action_description_for_log = f"LYNCH_VOTE: {target_agent_id}"
                        feedback_message = f"Action: Voted to lynch {target_agent_id}."
                    else:
                        is_valid_action = False
                        feedback_message = f"Invalid target for DAY_LYNCH_VOTE: {target_agent_id}. Target must be alive and not self."
                else:  # Action type not appropriate for phase
                    is_valid_action = False
                    feedback_message = f"Action type '{current_action_enum.name}' is not appropriate for phase '{self.current_phase.name}'."

            if not is_valid_action:
                self.rewards[acting_agent_id] = -1.0
                action_type_str = ActionType(action_type_val).name if isinstance(
                    action_type_val, int) and action_type_val in ActionType._value2member_map_ else str(action_type_val)
                if parsed_action_data is None:  # Pydantic failed
                    action_description_for_log = f"INVALID_ACTION (Pydantic Error): {str(action)[:100]}"
                else:  # Contextual validation failed
                    action_description_for_log = f"INVALID_ACTION (Contextual Error): type={action_type_str}, target={target_idx}, msg='{str(message_text)[:20]}...'"
                self.infos[acting_agent_id]["last_attempted_action_raw"] = str(
                    action)

            self.infos[acting_agent_id]["last_action_feedback"] = feedback_message
            self.infos[acting_agent_id]["last_action_valid"] = is_valid_action
            self.infos[acting_agent_id]["action_description_for_log"] = action_description_for_log

            if self.render_mode == "human":
                print(
                    f"Agent {acting_agent_id} ({current_role.name}) in {self.current_phase.name} action: {action_description_for_log}")
                if not is_valid_action:
                    print(f"    Feedback to agent: {feedback_message}")

        self._cumulative_rewards[acting_agent_id] += self.rewards[acting_agent_id]

        if self._agent_selector.is_last():
            if self.current_phase != Phase.GAME_OVER and self._check_win_conditions():
                self._transition_phase()

            if self.current_phase != Phase.GAME_OVER:
                self._transition_phase()
            else:  # Game is over
                if not all(self.terminations.values()):
                    self.agents = list(self.agent_ids)
                    self._agent_selector.reinit(self.agents)
                    self.agent_selection = self._agent_selector.next()
                elif self.agent_selection is not None and not self._agent_selector.is_last():
                    self.agent_selection = self._agent_selector.next()
        else:
            self.agent_selection = self._agent_selector.next()

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        self.render_step_ind = -1
        if seed is not None:
            random.seed(seed)

        if options and "num_players" in options:
            num_players = options["num_players"]
        else:
            raise ValueError(
                "WerewolfEnv.reset() called without num_players in options.")

        if not isinstance(num_players, int) or num_players < 3:
            raise ValueError(
                f"Werewolf game requires at least 3 players, got {num_players}.")
        self.num_players = num_players

        # Initialize num_players-dependent attributes
        self.num_werewolves = int(math.ceil(self.num_players / 4.0))

        # Validate num_doctors and num_seers against the now known num_players
        if self.num_werewolves + self.num_doctors + self.num_seers > self.num_players:
            raise ValueError(
                f"Too many special roles ({self.num_werewolves} WW, {self.num_doctors} Dr, {self.num_seers} Seer) for {self.num_players} players.")

        self._agent_ids = [
            f"player_{i}" for i in range(self.num_players)]
        self.index_to_agent_id = {i: agent_id for i,
                                  agent_id in enumerate(self.agent_ids)}
        self.agent_id_to_index = {agent_id: i for i,
                                  agent_id in enumerate(self.agent_ids)}

        # Initialize action and observation spaces
        self.action_spaces = {
            agent_id: spaces.Dict({
                "action_type": spaces.Discrete(len(ActionType)),
                "target_idx": spaces.Discrete(self.num_players),
                "message": spaces.Text(max_length=256)
            }) for agent_id in self.agent_ids
        }

        max_name_len = 128
        max_all_names_len = self.num_players * (max_name_len + 5) + 20
        self.observation_spaces = {
            agent_id: spaces.Dict({
                "role": spaces.Discrete(len(Role)),
                "phase": spaces.Discrete(len(Phase)),
                "alive_players": spaces.MultiBinary(self.num_players),
                "known_werewolves": spaces.MultiBinary(self.num_players),
                "seer_last_inspection": spaces.Tuple((
                    spaces.Discrete(self.num_players +
                                    1), spaces.Discrete(len(Role) + 1)
                )),
                "discussion_log": spaces.Text(max_length=4096),
                "last_lynched": spaces.Discrete(self.num_players + 1),
                "last_lynched_player_role": spaces.Discrete(len(Role) + 1),
                "last_killed_by_werewolf": spaces.Discrete(self.num_players + 1),
                "last_killed_by_werewolf_role": spaces.Discrete(len(Role) + 1),
                "my_unique_name": spaces.Text(max_length=max_name_len),
                "all_player_unique_names": spaces.Text(max_length=max_all_names_len),
                "last_day_vote_details": spaces.Text(max_length=1024),
                "current_day_vote_details": spaces.Text(max_length=1024),
                "current_night_werewolf_votes": spaces.Text(max_length=1024),
                "last_action_feedback": spaces.Text(max_length=512),
            }) for agent_id in self.agent_ids
        }

        # Initialize PettingZoo specific agent-keyed dictionaries
        self._assign_roles()
        self.rewards = {agent_id: 0.0 for agent_id in self.agent_ids}
        self._cumulative_rewards = {
            agent_id: 0.0 for agent_id in self.agent_ids}
        self.terminations = {
            agent_id: False for agent_id in self.agent_ids}
        self.truncations = {
            agent_id: False for agent_id in self.agent_ids}
        self.infos = {
            agent_id: {
                "role": self.player_roles[agent_id].name,
                "initial_unique_name": agent_id,
                "last_action_feedback": "No action taken yet in this episode."
            }
            for agent_id in self.agent_ids
        }

        # Initialize game state
        # Start with all agents for role assignment etc.
        self.agents = list(self.agent_ids)
        self.alive_agents = list(self.agent_ids)
        self.alive_agents_at_night_start = list(self.agent_ids)
        self.current_phase = Phase.NIGHT_WEREWOLF_VOTE
        self._werewolf_kill_votes.clear()
        self._doctor_save_choices.clear()
        self._seer_inspect_choices.clear()
        self._seer_inspection_results_for_obs.clear()
        self._doctor_save_outcomes_for_obs.clear()
        self._discussion_log_this_round.clear()
        self._day_lynch_votes.clear()
        self._last_lynched_player_idx = None
        self._last_lynched_player_role_val = None
        self._last_killed_by_werewolf_idx = None
        self._last_killed_by_werewolf_role_val = None
        self._last_day_vote_details = json.dumps({})
        self.game_winner_team = None
        self.active_player_indices_history = [None]

        # Set up agent selector for turn management.
        # The AECEnv's self._agent_selector is already created in super().__init__
        initial_actors = self._get_agents_by_role(Role.WEREWOLF)
        self.agents = list(initial_actors)
        self._agent_selector = AgentSelector(self.agents)
        self._agent_selector.reinit(self.agents)

        if self.agents:
            self.agent_selection = self._agent_selector.reset()
        else:
            self.agent_selection = None
            self._transition_phase()

        if self.render_mode == "human":
            print("\n--- New Werewolf Game Reset ---")
            print(
                f"Player Count: {self.num_players}, WW: {self.num_werewolves}, Doc: {self.num_doctors}, Seer: {self.num_seers}")
            print(f"Player Unique IDs: {self.agent_ids}")
            # print(f"Roles: {{p: r.name for p, r in self.player_roles.items()}}") # For debugging
            print(f"Initial Phase: {self.current_phase.name}")
            if self.agent_selection:
                print(
                    f"First to act: {self.agent_selection} ({self.player_roles.get(self.agent_selection).name})")

    def render(self):
        if self.render_mode == "human":
            print(f"\n--- State: Phase {self.current_phase.name} ---")
            print(f"Alive Agents: {self.alive_agents}")
            if self.agent_selection and not self.terminations.get(self.agent_selection, True):
                print(
                    f"Next to act: {self.agent_selection} ({self.player_roles.get(self.agent_selection, Role.VILLAGER).name})")

            if self._last_killed_by_werewolf_idx is not None and self._last_killed_by_werewolf_role_val is not None and self._last_killed_by_werewolf_role_val > 0:
                killed_name = self.index_to_agent_id.get(
                    self._last_killed_by_werewolf_idx, f"player_{self._last_killed_by_werewolf_idx}")
                killed_role_name = Role(
                    self._last_killed_by_werewolf_role_val).name
                print(
                    f"Last night kill: {killed_name} (Role: {killed_role_name})")
            if self._last_lynched_player_idx is not None and self._last_lynched_player_role_val is not None and self._last_lynched_player_role_val > 0:
                lynched_name = self.index_to_agent_id.get(
                    self._last_lynched_player_idx, f"player_{self._last_lynched_player_idx}")
                lynched_role_name = Role(
                    self._last_lynched_player_role_val).name
                print(
                    f"Last day lynch: {lynched_name} (Role: {lynched_role_name})")

            if self.current_phase == Phase.GAME_OVER:
                print(f"GAME OVER! Winner Team: {self.game_winner_team}")
                final_roles_str = {p_id: r.name for p_id,
                                   r in self.player_roles.items()}
                print(f"Final Roles: {final_roles_str}")
                print(f"Cumulative Rewards: {self._cumulative_rewards}")
                print(f"Terminations: {self.terminations}")

    def close(self):
        pass
