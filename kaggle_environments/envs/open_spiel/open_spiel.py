"""Kaggle environment wrapper for OpenSpiel games."""

import copy
import os
import pathlib
import random
from typing import Any, Callable

from kaggle_environments import core
from kaggle_environments import utils
import numpy as np
import pyspiel
from .games.connect_four import connect_four_proxy

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
  num_players = game.num_players()
  statuses = [
      kaggle_state[os_current_player].status
      for os_current_player in range(num_players)
  ]
  if not any(status == "ACTIVE" for status in statuses):
    raise ValueError("Environment not done and no active agents.")

  # --- Initialization / Reset ---
  # TODO(jhtschultz): test this behavior.
  is_initial_step = len(env.steps) == 1
  if _OS_GLOBAL_STATE is None or (not is_initial_step and env.done):
    _OS_GLOBAL_STATE = game.new_initial_state()

  # --- Maybe apply agent action ---
  os_current_player = _OS_GLOBAL_STATE.current_player()
  action_applied = None
  if is_initial_step:
    pass
  elif 0 <= os_current_player < num_players:
    if kaggle_state[os_current_player].status != "ACTIVE":
      pass
    else:
      action_submitted = kaggle_state[os_current_player].action
      legal = _OS_GLOBAL_STATE.legal_actions()
      if action_submitted in legal:
        try:
          _OS_GLOBAL_STATE.apply_action(action_submitted)
          action_applied = action_submitted
        except Exception:  # pylint: disable=broad-exception-caught
          kaggle_state[os_current_player].status = "ERROR"
      else:
        kaggle_state[os_current_player].status = "INVALID"
  elif os_current_player == pyspiel.PlayerId.SIMULTANEOUS:
    raise NotImplementedError
  elif os_current_player == pyspiel.PlayerId.TERMINAL:
    pass
  elif os_current_player == pyspiel.PlayerId.CHANCE:
    raise ValueError("Interpreter should not be called at chance nodes.")
  else:
    raise ValueError(f"Unknown OpenSpiel player ID: {os_current_player}")

  # --- Update state info ---
  while _OS_GLOBAL_STATE.is_chance_node():
    chance_outcomes = _OS_GLOBAL_STATE.chance_outcomes
    outcomes = _OS_GLOBAL_STATE.chance_outcomes()
    legal_actions, chance_outcome_probs = zip(*outcomes)
    action = np.random.choice(legal_actions, p=chance_outcome_probs)
    _OS_GLOBAL_STATE.apply_action(action)
  is_terminal = _OS_GLOBAL_STATE.is_terminal()
  agent_returns = _OS_GLOBAL_STATE.returns() + [None]
  next_agent = _OS_GLOBAL_STATE.current_player()

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
    if os_current_player == i and action_applied is not None:
      info_dict["action_applied"] = action_applied

    game_type = _OS_GLOBAL_GAME.get_type()
    obs_str = str(_OS_GLOBAL_STATE)
    legal_actions = _OS_GLOBAL_STATE.legal_actions(i)

    if status == "ACTIVE" and not legal_actions:
      raise ValueError(
        f"Active agent {i} has no legal actions in state {_OS_GLOBAL_STATE}."
      )

    # Apply updates
    obs_update_dict = {
      "observation_string": obs_str,
      "legal_actions": legal_actions,
      "current_player": next_agent,
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

# --- HTML Renderer Logic ---

def _default_html_renderer() -> str:
  """Provides the JavaScript string for the default HTML renderer."""
  return """
function renderer(context) {
    const { parent, environment, step } = context;
    parent.innerHTML = ''; // Clear previous rendering

    const currentStepData = environment.steps[step];
    if (!currentStepData) {
        parent.textContent = "Waiting for step data...";
        return;
    }
    const numAgents = currentStepData.length;
    const gameMasterIndex = numAgents - 1;
    let obsString = "Observation not available for this step.";
    let title = `Step: ${step}`;

    if (environment.configuration && environment.configuration.openSpielGameName) {
        title = `${environment.configuration.openSpielGameName} - Step: ${step}`;
    }

    // Try to get obs_string from game_master of current step
    if (currentStepData[gameMasterIndex] && 
        currentStepData[gameMasterIndex].observation && 
        typeof currentStepData[gameMasterIndex].observation.observation_string === 'string') {
        obsString = currentStepData[gameMasterIndex].observation.observation_string;
    } 
    // Fallback to initial step if current is unavailable (e.g. very first render call)
    else if (step === 0 && environment.steps[0] && environment.steps[0][gameMasterIndex] && 
             environment.steps[0][gameMasterIndex].observation &&
             typeof environment.steps[0][gameMasterIndex].observation.observation_string === 'string') {
        obsString = environment.steps[0][gameMasterIndex].observation.observation_string;
    }

    const pre = document.createElement("pre");
    pre.style.fontFamily = "monospace";
    pre.style.margin = "10px";
    pre.style.border = "1px solid #ccc";
    pre.style.padding = "10px";
    pre.style.backgroundColor = "#f9f9f9";
    pre.style.whiteSpace = "pre-wrap";
    pre.style.wordBreak = "break-all";

    pre.textContent = `${title}\\n\\n${obsString}`;
    parent.appendChild(pre);
}
"""

def _get_html_renderer_content(
    open_spiel_short_name: str,
    base_path_for_custom_renderers: pathlib.Path,
    default_renderer_func: Callable[[], str]
) -> str:
  """
  Tries to load a custom JS renderer for the game.
  Falls back to the default renderer if not found or on error.
  """
  if "proxy" not in open_spiel_short_name:
    return default_renderer_func()
  sanitized_game_name = open_spiel_short_name.replace('-', '_').replace('.', '_')
  sanitized_game_name = sanitized_game_name.removesuffix("_proxy")
  custom_renderer_js_path = (
      base_path_for_custom_renderers /
      sanitized_game_name /
      f"{sanitized_game_name}.js"
  )
  if custom_renderer_js_path.is_file():
    try:
      with open(custom_renderer_js_path, "r", encoding="utf-8") as f:
        content = f.read()
      print(f"INFO: Using custom HTML renderer for {open_spiel_short_name} from {custom_renderer_js_path}")
      return content
    except Exception as e_render:
      pass
  return default_renderer_func()


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


agents = {
  "random": random_agent,
}


def _register_open_spiel_envs(
  games_list: list[str] | None = None,
) -> dict[str, Any]:
  successfully_loaded_games = []
  skipped_games = []
  registered_envs = {}
  current_file_dir = pathlib.Path(__file__).parent.resolve()
  custom_renderers_base = current_file_dir / "games"
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
      game_spec["agents"] = [game.num_players()]
      game_spec["configuration"]["episodeSteps"] = (
          game.max_history_length() + DEFAULT_EPISODE_STEP_BUFFER
      )
      game_spec["configuration"]["openSpielGameString"]["default"] = str(game)
      game_spec["configuration"]["openSpielGameName"]["default"] = short_name
      game_spec["observation"]["properties"]["openSpielGameString"][
          "default"] = str(game)
      game_spec["observation"]["properties"]["openSpielGameName"][
          "default"] = short_name

      # Building html_renderer_callable is a bit convoluted but other approaches
      # failed for a variety of reasons. Returning a simple lambda function
      # doesn't work because of late-binding. The last env registered will
      # overwrite all previous renderers.
      js_string_content = _get_html_renderer_content(
          open_spiel_short_name=short_name,
          base_path_for_custom_renderers=custom_renderers_base,
          default_renderer_func=_default_html_renderer,
      )

      def create_html_renderer_closure(captured_content):
          def html_renderer_callable_no_args():
              return captured_content
          return html_renderer_callable_no_args

      html_renderer_callable = create_html_renderer_closure(js_string_content)

      registered_envs[env_name] = {
          "specification": game_spec,
          "interpreter": interpreter,
          "renderer": renderer,
          "html_renderer": html_renderer_callable,
          "agents": agents,
      }
      successfully_loaded_games.append(short_name)

    except Exception as e:  # pylint: disable=broad-exception-caught
      skipped_games.append(short_name)
      continue

  print(f"""
Successfully loaded OpenSpiel environments: {len(successfully_loaded_games)}.
OpenSpiel games skipped: {len(skipped_games)}.
""".strip())

  return registered_envs


registered_open_spiel_envs = _register_open_spiel_envs()
