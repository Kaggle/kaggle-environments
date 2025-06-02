"""Kaggle environment wrapper for OpenSpiel games."""

import copy
import random
from typing import Any

from kaggle_environments import core
from kaggle_environments import utils
import numpy as np
import pyspiel


DEFAULT_ACT_TIMEOUT = 5
DEFAULT_RUN_TIMEOUT = 1200
DEFAULT_EPISODE_STEP_BUFFER = 100  # To account for timeouts, retrys, etc...

BASE_SPEC_TEMPLATE = {
    "name": "PLACEHOLDER_NAME",
    "title": "PLACEHOLDER_TITLE",
    "description": "PLACEHOLDER_DESCRIPTION",
    "version": "0.1.0",
    "agents": ["PLACEHOLDER_NUM_AGENTS"],

    "configuration": {
        "episodeSteps": -1,
        "actTimeout": DEFAULT_ACT_TIMEOUT,
        "runTimeout": DEFAULT_RUN_TIMEOUT,
        "openSpielGameString": {
            "description": "The full game string including parameters.",
            "type": "string",
            "default": "PLACEHOLDER_GAME_STRING"
        },
        "openSpielGameName": {
            "description": "The short_name of the OpenSpiel game to load.",
            "type": "string",
            "default": "PLACEHOLDER_GAME_SHORT_NAME"
        },
    },
    "observation": {
        "properties": {
            "openSpielGameString": {
                "description": "Full game string including parameters.",
                "type": "string"
            },
            "openSpielGameName": {
                "description": "Short name of the OpenSpiel game.",
                "type": "string"
            },
            "observation_string": {
                "description": "String representation of state.",
                "type": "string"
            },
            # TODO(jhtschultz): add legal action strings
            "legal_actions": {
                "description": "List of OpenSpiel legal actions.",
                "type": "array",
                "items": {
                    "type": "integer"
                }
            },
            "chance_outcome_probs": {
                "description": "List of probabilities for chance outcomes.",
                "type": "array",
                "items": {
                    "type": "float"
                }
            },
            "current_player": {
                "description": "ID of player whose turn it is.",
                "type": "integer"
            },
            "is_terminal": {
                "description": "Boolean indicating game end.",
                "type": "boolean"
            },
            "player_id": {
                "description": "ID of the agent receiving this observation.",
                "type": "integer"
            },
            "remainingOverageTime": 60,
            "step": 0
        },
        "default": {}
    },
    "action": {
        "type": ["integer"],
        "minimum": -1,
        "default": -1
    },
    "reward": {
        "type": ["number"],
        "default": 0.0
    },
}


_OS_GLOBAL_GAME = None
_OS_GLOBAL_STATE = None


def _get_open_spiel_game(env_config: utils.Struct) -> pyspiel.Game:
  global _OS_GLOBAL_GAME
  game_string = env_config.get("openSpielGameString")
  if game_string == str(_OS_GLOBAL_GAME):
    return _OS_GLOBAL_GAME
  if _OS_GLOBAL_GAME is not None:
    print(
      f"WARNING: Overwriting game. Old: {_OS_GLOBAL_GAME}. New {game_string}"
    )
  _OS_GLOBAL_GAME = pyspiel.load_game(game_string)
  return _OS_GLOBAL_GAME


# Currently we make the assumption that there will always be an extra "game
# master" agent in the final agent position. This agent corresponds to the
# OpenSpiel chance player, and is responsible for handling chance actions, as
# well as acting as an observer with full access to the game state. These are
# the observations used to create the game replay.
def _os_player_to_kaggle_agent(player: int, num_players: int) -> int:
  if player == pyspiel.PlayerId.CHANCE:
    return num_players
  return player


# Unused but potentially helpful.
def _reconstruct_os_state(
    game: pyspiel.Game,
    kaggle_history: list,
    num_expected_agents: int,
) -> pyspiel.State:
  os_state = game.new_initial_state()
  for i, kaggle_state in enumerate(kaggle_history):
    if i == 0:
      continue  # kaggle_history[0] is the initial dummy state from core.py
    os_current_player = os_state.current_player()
    current_agent = _os_player_to_kaggle_agent(os_current_player,
                                               game.num_players())
    if 0 <= current_agent < num_expected_agents:
      action = kaggle_state[current_agent].info.get("action_applied")
      if action is not None:
        legal_actions = list(os_state.legal_actions())
        if action in legal_actions:
          os_state.apply_action(action)
        else:
          raise ValueError(
              f"_reconstruct_os_state failed to find action {action} "
              f"in legal actions: {legal_actions}"
          )
      else:
        raise ValueError(
            "_reconstruct_os_state found None action for "
            f"current player {os_current_player} in state: {os_state}"
        )
    elif os_current_player == pyspiel.PlayerId.SIMULTANEOUS:
      raise NotImplementedError
    elif os_current_player == pyspiel.PlayerId.TERMINAL:
      continue
    else:
      raise ValueError(f"Invalid player: {os_current_player}")
  return os_state


def interpreter(
  state: list[utils.Struct],
  env: core.Environment,
) -> list[utils.Struct]:
  """Updates environment using player responses and returns new observations."""
  global _OS_GLOBAL_GAME, _OS_GLOBAL_STATE
  kaggle_state = state
  del state

  if env.done:
    return kaggle_state

  # --- Get Game Info ---
  game = _get_open_spiel_game(env.configuration)
  num_players = game.num_players()  # Actual number of players
  num_agents = len(kaggle_state)
  if num_agents != num_players + 1:
    raise ValueError(
        f"Invalid num_agents: {num_agents}. Open Spiel must always include a "
        "game master in the final agent position."
    )

  statuses = [
      kaggle_state[os_current_player].status
      for os_current_player in range(num_agents)
  ]
  if not any(status == "ACTIVE" for status in statuses):
    raise ValueError("Environment not done and no active agents.")

  # --- Initialization / Reset ---
  # TODO(jhtschultz): test this behavior.
  is_initial_step = len(env.steps) == 1
  if _OS_GLOBAL_STATE is None or (not is_initial_step and env.done):
    _OS_GLOBAL_STATE = game.new_initial_state()
  # Alternatively can reconstruct the state
  # os_state = _reconstruct_os_state(game, env.steps, num_agents)

  # --- Maybe apply agent action ---
  os_current_player = _OS_GLOBAL_STATE.current_player()
  current_agent = _os_player_to_kaggle_agent(os_current_player, num_players)
  action_applied = None
  if is_initial_step:
    pass
  elif kaggle_state[current_agent].status != "ACTIVE":
    pass
  elif 0 <= current_agent < num_agents:
    action_submitted = kaggle_state[current_agent].action
    legal = _OS_GLOBAL_STATE.legal_actions()
    if action_submitted in legal:
      try:
        _OS_GLOBAL_STATE.apply_action(action_submitted)
        action_applied = action_submitted
      except Exception:  # pylint: disable=broad-exception-caught
        kaggle_state[current_agent].status = "ERROR"
    else:
      kaggle_state[current_agent].status = "INVALID"
  elif os_current_player == pyspiel.PlayerId.SIMULTANEOUS:
    raise NotImplementedError
  elif os_current_player == pyspiel.PlayerId.TERMINAL:
    pass
  else:
    raise ValueError(f"Unknown OpenSpiel player ID: {os_current_player}")

  # --- Update state info ---
  is_terminal = _OS_GLOBAL_STATE.is_terminal()
  agent_returns = _OS_GLOBAL_STATE.returns() + [None]
  os_next_player = _OS_GLOBAL_STATE.current_player()
  next_agent = _os_player_to_kaggle_agent(os_next_player, num_players)

  for i, agent_state in enumerate(kaggle_state):
    input_status = agent_state.status
    status = ""
    reward = None

    if input_status in ["TIMEOUT", "ERROR", "INVALID"]:
      status = input_status
      reward = None
    elif is_terminal:
      status = "DONE"
      reward = agent_returns[i]
    elif next_agent == i:
      status = "ACTIVE"
      reward = agent_returns[i]
    else:
      status = "INACTIVE"
      reward = agent_returns[i]

    info_dict = {}
    # Store the applied action in info for potential debugging/analysis
    if current_agent == i and action_applied is not None:
      info_dict["action_applied"] = action_applied

    legal_actions = []
    chance_outcome_probs = []
    if i == num_agents - 1:  # Game master agent
      obs_str = str(_OS_GLOBAL_STATE)
      if _OS_GLOBAL_STATE.is_chance_node():
        outcomes = _OS_GLOBAL_STATE.chance_outcomes()
        legal_actions, chance_outcome_probs = zip(*outcomes)
    else:
      game_type = _OS_GLOBAL_GAME.get_type()
      if game_type.provides_information_state_string:
        obs_str = _OS_GLOBAL_STATE.information_state_string(i)
      elif game_type.provides_observation_string:
        obs_str = _OS_GLOBAL_STATE.observation_string(i)
      else:
        raise ValueError(
          "Must provide either information state or observation string"
        )
      legal_actions = _OS_GLOBAL_STATE.legal_actions(i)

    if status == "ACTIVE" and not legal_actions:
      raise ValueError(
        f"Active agent {i} has no legal actions in state {_OS_GLOBAL_STATE}."
      )

    # Apply updates
    obs_update_dict = {
      "observation_string": obs_str,
      "legal_actions": legal_actions,
      "chance_outcome_probs": chance_outcome_probs,
      "current_player": os_next_player,
      "is_terminal": is_terminal,
      "player_id": i,
    }
    for k, v in obs_update_dict.items():
      setattr(agent_state.observation, k, v)
    agent_state.reward = reward
    agent_state.info = info_dict
    agent_state.status = status

  return kaggle_state


def renderer(state: list[utils.Struct], env: core.Environment) -> str:
  """Kaggle renderer function."""
  try:
    obs_str = state[-1].observation["observation_string"]
    return obs_str if obs_str else "<Empty observation string>"
  except Exception as e:  # pylint: disable=broad-exception-caught
    print(f"Error rendering {env.name} at state: {state}.")
    raise e


def html_renderer():
  """Provides the simplest possible HTML/JS renderer for OpenSpiel text observations."""
  return """
function renderer(context) {
    const { parent, environment, step } = context;
    parent.innerHTML = ''; // Clear previous rendering

    // Get the current step's data
    const currentStepData = environment.steps[step];
    const numAgents = currentStepData.length
    const gameMasterIndex = numAgents - 1
    let obsString = "Observation not available for this step.";

    // Try to get the raw observation string from the game master agent.
    if (currentStepData && currentStepData[gameMasterIndex] && currentStepData[gameMasterIndex].observation && currentStepData[gameMasterIndex].observation.observation_string !== undefined) {
        obsString = currentStepData[gameMasterIndex].observation.observation_string;
    } else if (step === 0 && environment.steps[0] && environment.steps[0][gameMasterIndex] && environment.steps[0][gameMasterIndex].observation && environment.steps[0][gameMasterIndex].observation.observation_string !== undefined) {
        // Fallback for initial state if current step data is missing
        obsString = environment.steps[0][gameMasterIndex].observation.observation_string;
    }

    // Create a <pre> element to preserve formatting
    const pre = document.createElement("pre");
    pre.style.fontFamily = "monospace"; // Ensure monospace font
    pre.style.margin = "10px";        // Add some padding
    pre.style.border = "1px solid #ccc";
    pre.style.padding = "5px";
    pre.style.backgroundColor = "#f0f0f0";

    // Set the text content (safer than innerHTML for plain text)
    pre.textContent = `Step: ${step}\\n\\n${obsString}`; // Add step number for context

    parent.appendChild(pre);
}
"""


# --- Agents ---
def random_agent(
  observation: dict[str, Any],
  configuration: dict[str, Any],
) -> int:
  """A built-in random agent specifically for OpenSpiel environments."""
  del configuration
  legal_actions = observation.get("legal_actions")
  if not legal_actions:
    return None
  action = random.choice(legal_actions)
  return int(action)


def game_master_agent(
  observation: dict[str, Any],
  configuration: dict[str, Any],
) -> int:
  """Agent for handling chance nodes and recording full game state."""
  del configuration
  legal_actions = observation.get("legal_actions")
  chance_outcome_probs = observation.get("chance_outcome_probs")
  if not legal_actions:
    return None
  if not chance_outcome_probs:
    raise ValueError("Game master received legal actions without probs.")
  action = np.random.choice(legal_actions, p=chance_outcome_probs)
  return int(action)


agents = {
  "game_master": game_master_agent,
  "random": random_agent,
}

def _register_open_spiel_envs(
  games_list: list[str] | None = None,
) -> dict[str, Any]:
  successfully_loaded_games = []
  skipped_games = []
  registered_envs = {}
  if games_list is None:
    games_list = pyspiel.registered_names()
  for short_name in games_list:
    try:
      game = pyspiel.load_game(short_name)
      game_type = game.get_type()
      if not any([
        game_type.provides_information_state_string,
        game_type.provides_observation_string,
      ]):
        continue
      game_spec = copy.deepcopy(BASE_SPEC_TEMPLATE)
      env_name = f"open_spiel_{short_name.replace('-', '_').replace('.', '_')}"
      game_spec["name"] = env_name
      game_spec["title"] = f"Open Spiel: {short_name}"
      game_spec["description"] = """
Kaggle environment wrapper for OpenSpiel games.
For game implementation details see:
https://github.com/google-deepmind/open_spiel/tree/master/open_spiel/games
""".strip()
      # Extra game master agent for handling chance nodes.
      game_spec["agents"] = [game.num_players() + 1]
      game_spec["configuration"]["episodeSteps"] = (
          game.max_history_length() + DEFAULT_EPISODE_STEP_BUFFER
      )
      game_spec["configuration"]["openSpielGameString"]["default"] = str(game)
      game_spec["configuration"]["openSpielGameName"]["default"] = short_name
      game_spec["observation"]["properties"]["openSpielGameString"][
          "default"] = str(game)
      game_spec["observation"]["properties"]["openSpielGameName"][
          "default"] = short_name

      registered_envs[env_name] = {
          "specification": game_spec,
          "interpreter": interpreter,
          "renderer": renderer,
          "html_renderer": html_renderer,
          "agents": agents,
      }
      successfully_loaded_games.append(short_name)

    except Exception:  # pylint: disable=broad-exception-caught
      skipped_games.append(short_name)
      continue

  print(f"""
Successfully loaded OpenSpiel environments: {len(successfully_loaded_games)}.
OpenSpiel games skipped: {len(skipped_games)}.
""".strip())

  return registered_envs


registered_open_spiel_envs = _register_open_spiel_envs()
