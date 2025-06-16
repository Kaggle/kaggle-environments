"""Kaggle environment wrapper for OpenSpiel games."""

import copy
import importlib
import logging
import os
import pathlib
import random
import sys
from typing import Any, Callable

from kaggle_environments import core
from kaggle_environments import utils
import numpy as np
import pyspiel

_log = logging.getLogger(__name__) 
_log.setLevel(logging.INFO) 
_handler = logging.StreamHandler(sys.stdout) 
_formatter = logging.Formatter('[%(name)s] %(levelname)s: %(message)s')
_handler.setFormatter(_formatter)
_log.addHandler(_handler)

# --- Import proxy games ---
_log.debug("Auto-importing OpenSpiel game proxies...")
GAMES_DIR = pathlib.Path(__file__).parent / "games"
for proxy_file in GAMES_DIR.glob("**/*_proxy.py"):
  try:
    relative_path = proxy_file.relative_to(GAMES_DIR.parent)
    module_path = str(relative_path.with_suffix("")).replace(os.path.sep, ".")
    importlib.import_module("." + module_path, package=__package__)
    _log.debug(f"  - Imported: {module_path}")
  except Exception as e:  # pylint: disable=broad-exception-caught
    _log.debug(f"  - FAILED to import proxy from {proxy_file.name}: {e}")


# --- Constants ---
DEFAULT_ACT_TIMEOUT = 5
DEFAULT_RUN_TIMEOUT = 1200
# Buffer in addition to max game length to account for timeouts, retrys, etc.
DEFAULT_STEP_BUFFER = 100
# TODO(jhtschultz): Add individual game descriptions.
DEFAULT_DESCRIPTION = """
Kaggle environment wrapper for OpenSpiel games.
For game implementation details see:
https://github.com/google-deepmind/open_spiel/tree/master/open_spiel/games
""".strip()

CONFIGURATION_SPEC_TEMPLATE = {
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
    "openSpielGameParameters": {
        "description": "Game parameters for Open Spiel game.",
        "type": "object",
        "default": {}
    },
}

OBSERVATION_SPEC_TEMPLATE = {
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
        # TODO(jhtschultz): Use camel case for consistency with spec, or snake
        # case for consistency with pyspiel?
        "legal_actions": {
            "description": "List of OpenSpiel legal action integers.",
            "type": "array",
            "items": {
                "type": "integer"
            }
        },
        "legal_action_strings": {
            "description": "List of OpenSpiel legal actions strings.",
            "type": "array",
            "items": {
                "type": "string"
            }
        },
        "current_player": {
            "description": "ID of player whose turn it is.",
            "type": "integer"
        },
        "player_id": {
            "description": "ID of the agent receiving this observation.",
            "type": "integer"
        },
        "is_terminal": {
            "description": "Boolean indicating game end.",
            "type": "boolean"
        },
        "remainingOverageTime": 60,
        "step": 0
    },
    "default": {}
}


ENV_SPEC_TEMPLATE = {
    "name": "PLACEHOLDER_NAME",
    "title": "PLACEHOLDER_TITLE",
    "description": DEFAULT_DESCRIPTION,
    "version": "0.1.0",
    "agents": ["PLACEHOLDER_NUM_AGENTS"],
    "configuration": CONFIGURATION_SPEC_TEMPLATE,
    "observation": OBSERVATION_SPEC_TEMPLATE,
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


# --- Core step logic ---

def interpreter(
  state: list[utils.Struct],
  env: core.Environment,
) -> list[utils.Struct]:
  """Updates environment using player responses and returns new observations."""
  kaggle_state = state  # Not to be confused with OpenSpiel state.
  del state

  # TODO(jhtschultz): Test reset behavior. Currently containers are restarted
  # after each episode.
  if env.done:
    return kaggle_state

  # --- Get and maybe initialize game and state on the env object ---
  if not hasattr(env, 'os_game'):
    game_string = env.configuration.get("openSpielGameString")
    env.os_game = pyspiel.load_game(game_string)
  if not hasattr(env, 'os_state'):
    env.os_state = env.os_game.new_initial_state()
  if "state_history" not in env.info:
    env.info['state_history'] = [str(env.os_state)]
    env.info['action_history'] = []
  
  os_game = env.os_game
  os_state = env.os_state
  num_players = os_game.num_players()
  statuses = [
      kaggle_state[player_id].status for player_id in range(num_players)
  ]
  if not any(status == "ACTIVE" for status in statuses):
    raise ValueError("Environment not done and no active agents.")

  # TODO(jhtschultz): Test reset behavior.
  is_initial_step = len(env.steps) == 1
  if is_initial_step and os_state.is_terminal():
    env.os_state = os_game.new_initial_state()
    os_state = env.os_state

  # --- Apply agent action ---
  acting_agent = os_state.current_player()
  action_submitted = None
  action_applied = None
  if is_initial_step:
    pass
  elif 0 <= acting_agent < num_players:
    if kaggle_state[acting_agent].status != "ACTIVE":
      pass
    else:
      action_submitted = kaggle_state[acting_agent].action
      if action_submitted in os_state.legal_actions():
        try:
          os_state.apply_action(action_submitted)
          action_applied = action_submitted
          env.info['action_history'].append(str(action_applied))
          env.info['state_history'].append(str(os_state))
        except Exception as e:  # pylint: disable=broad-exception-caught
          _log.debug(e)
          kaggle_state[acting_agent].status = "ERROR"
      else:
        kaggle_state[acting_agent].status = "INVALID"
  elif acting_agent == pyspiel.PlayerId.SIMULTANEOUS:
    raise NotImplementedError
  elif acting_agent == pyspiel.PlayerId.TERMINAL:
    pass
  elif acting_agent == pyspiel.PlayerId.CHANCE:
    raise ValueError("Interpreter should not be called at chance nodes.")
  else:
    raise ValueError(f"Unknown OpenSpiel player ID: {acting_agent}")

  # --- Step chance nodes ---
  while os_state.is_chance_node():
    outcomes, probs = zip(*os_state.chance_outcomes())
    chance_action = np.random.choice(outcomes, p=probs)
    os_state.apply_action(chance_action)
    env.info['action_history'].append(str(chance_action))
    env.info['state_history'].append(str(os_state))

  # --- Update agent states ---
  for player_id, agent_state in enumerate(kaggle_state):
    reward = None
    if agent_state.status in ["TIMEOUT", "ERROR", "INVALID"]:
      status = agent_state.status
    elif os_state.is_terminal():
      status = "DONE"
      reward = os_state.returns()[player_id]
    elif os_state.current_player() == player_id:
      status = "ACTIVE"
      if not os_state.legal_actions(player_id):
        raise ValueError(
          f"Active agent {i} has no legal actions in state {os_state}."
        )
    else:
      status = "INACTIVE"

    info_dict = {}
    if acting_agent == player_id:
      info_dict["action_submitted"] = action_submitted
      info_dict["action_applied"] = action_applied

    obs_update_dict = {
      "observation_string": os_state.observation_string(player_id),
      "legal_actions": os_state.legal_actions(player_id),
      "legal_action_strings": [
          os_state.action_to_string(action) for action
          in os_state.legal_actions(player_id)
      ],
      "current_player": os_state.current_player(),
      "is_terminal": os_state.is_terminal(),
      "player_id": player_id,
    }

    # Apply updates
    for k, v in obs_update_dict.items():
      setattr(agent_state.observation, k, v)
    agent_state.reward = reward
    agent_state.info = info_dict
    agent_state.status = status

  return kaggle_state


# --- Rendering ---

def renderer(state: list[utils.Struct], env: core.Environment) -> str:
  """Kaggle environment text renderer."""
  if hasattr(env, 'os_state'):
    return str(env.os_state)
  else:
    return "Game state uninitialized."


# TODO(jhtschultz): Use custom player.html that replays from env.info instead
# of player steps. The full game state is stored in env.info, player steps only
# contain player observations.
def _default_html_renderer() -> str:
  """Provides the JavaScript string for the default HTML renderer."""
  return """
function renderer(context) {
    const { parent, environment, step } = context;
    parent.innerHTML = '';  // Clear previous rendering

    const currentStepData = environment.steps[step];
    if (!currentStepData) {
        parent.textContent = "Waiting for step data...";
        return;
    }
    const agentObsIndex = 0
    let obsString = "Observation not available for this step.";
    let title = `Step: ${step}`;

    if (environment.configuration && environment.configuration.openSpielGameName) {
        title = `${environment.configuration.openSpielGameName} - Step: ${step}`;
    }

    // Try to get obs_string from game_master of current step
    if (currentStepData[agentObsIndex] && 
        currentStepData[agentObsIndex].observation && 
        typeof currentStepData[agentObsIndex].observation.observation_string === 'string') {
        obsString = currentStepData[agentObsIndex].observation.observation_string;
    } 
    // Fallback to initial step if current is unavailable (e.g. very first render call)
    else if (step === 0 && environment.steps[0] && environment.steps[0][agentObsIndex] && 
             environment.steps[0][agentObsIndex].observation &&
             typeof environment.steps[0][agentObsIndex].observation.observation_string === 'string') {
        obsString = environment.steps[0][agentObsIndex].observation.observation_string;
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
  """Tries to load a custom JS renderer for the game, falls back to default."""
  custom_renderer_js_path = pathlib.Path(
      base_path_for_custom_renderers,
      open_spiel_short_name,
      f"{open_spiel_short_name}.js",
  )
  if custom_renderer_js_path.is_file():
    try:
      with open(custom_renderer_js_path, "r", encoding="utf-8") as f:
        content = f.read()
      _log.debug(f"Using custom HTML renderer for {open_spiel_short_name} from {custom_renderer_js_path}")
      return content
    except Exception as e:  # pylint: disable=broad-exception-caught
      _log.debug(e)
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


AGENT_REGISTRY = {
  "random": random_agent,
}


# --- Build and register environments --- 

def _build_env(game_string: str) -> dict[str, Any]:
  game = pyspiel.load_game(game_string)
  short_name = game.get_type().short_name

  proxy_path = GAMES_DIR / short_name / f"{short_name}_proxy.py"
  if proxy_path.is_file():
    game = pyspiel.load_game(short_name + "_proxy", game.get_parameters())

  game_type = game.get_type()
  if not game_type.provides_observation_string:
    raise ValueError(f"No observation string for game: {game_string}")

  env_spec = copy.deepcopy(ENV_SPEC_TEMPLATE)
  env_spec["name"] = f"open_spiel_{short_name}"
  env_spec["title"] = f"Open Spiel: {short_name}"
  env_spec["agents"] = [game.num_players()]

  env_config = copy.deepcopy(CONFIGURATION_SPEC_TEMPLATE)
  env_spec["configuration"] = env_config
  env_config["episodeSteps"] = game.max_history_length() + DEFAULT_STEP_BUFFER
  env_config["openSpielGameString"]["default"] = str(game)
  env_config["openSpielGameName"]["default"] = short_name

  env_obs = copy.deepcopy(OBSERVATION_SPEC_TEMPLATE)
  env_spec["observation"] = env_obs
  env_obs["properties"]["openSpielGameString"]["default"] = str(game)
  env_obs["properties"]["openSpielGameName"]["default"] = short_name

  # Building html_renderer_callable is a bit convoluted but other approaches
  # fail for a variety of reasons. Returning a simple lambda function
  # doesn't work because of late-binding -- the last env registered will
  # overwrite all previous renderers.
  js_string_content = _get_html_renderer_content(
      open_spiel_short_name=short_name,
      base_path_for_custom_renderers=GAMES_DIR,
      default_renderer_func=_default_html_renderer,
  )

  def create_html_renderer_closure(captured_content):
      def html_renderer_callable_no_args():
          return captured_content
      return html_renderer_callable_no_args

  html_renderer_callable = create_html_renderer_closure(js_string_content)

  return {
      "specification": env_spec,
      "interpreter": interpreter,
      "renderer": renderer,
      "html_renderer": html_renderer_callable,
      "agents": AGENT_REGISTRY,
  }


def _register_game_envs(games_list: list[str]) -> dict[str, Any]:
  skipped_games = []
  registered_envs = {}
  for game_string in games_list:
    try:
      env_config = _build_env(game_string)
      if env_config is None:
        continue
      env_name = env_config["specification"]["name"]
      if env_name in registered_envs:
        raise ValueError(f"Attempting to overwrite existing env: {env_name}")
      registered_envs[env_name] = env_config
    except Exception as e:  # pylint: disable=broad-exception-caught
      _log.debug(e)
      skipped_games.append(game_string)

  _log.info(f"Successfully loaded OpenSpiel environments: {len(registered_envs)}.")
  for env_name in registered_envs:
    _log.info(f"   {env_name}")
  _log.info(f"OpenSpiel games skipped: {len(skipped_games)}.")
  for game_string in skipped_games:
    _log.info(f"   {game_string}")

  return registered_envs


GAMES_LIST = [
    "chess",
    "connect_four",
    "gin_rummy",
    "go(board_size=9)",
    "tic_tac_toe",
    "universal_poker(betting=nolimit,bettingAbstraction=fullgame,blind=1 2,firstPlayer=2 1 1 1,numBoardCards=0 3 1 1,numHoleCards=2,numPlayers=2,numRanks=13,numRounds=4,numSuits=4,stack=400 400)",
]

ENV_REGISTRY = _register_game_envs(GAMES_LIST)
