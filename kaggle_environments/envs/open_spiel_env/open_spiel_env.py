"""Kaggle environment wrapper for OpenSpiel games."""

import copy
import importlib
import json
import logging
import os
import pathlib
import random
import re
import sys
import warnings
from typing import Any, Callable

import numpy as np
import pokerkit  # noqa: F401
import pyspiel
from open_spiel.python.games import pokerkit_wrapper  # noqa: F401

from kaggle_environments import core, utils

ERROR = "ERROR"
DONE = "DONE"
INACTIVE = "INACTIVE"
ACTIVE = "ACTIVE"
TIMEOUT = "TIMEOUT"
INVALID = "INVALID"

_log = logging.getLogger(__name__)
_log.setLevel(logging.INFO)
_handler = logging.StreamHandler(sys.stdout)
_formatter = logging.Formatter("[%(name)s] %(levelname)s: %(message)s")
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
# TODO(jhtschultz): Make this configurable per-game. For instance, in poker, a
# invalid action would likely result in a fold, forfeiting the player's
# contribution to the pot.
DEFAULT_INVALID_ACTION_REWARD = -1

# Can be used by agents to signal an internal error to the environement.
AGENT_ERROR_ACTION = -2

DEFAULT_ACT_TIMEOUT = 60 * 60  # sixty minutes
DEFAULT_RUN_TIMEOUT = 60 * 60 * 48  # thirty hours
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
        "default": "PLACEHOLDER_GAME_STRING",
    },
    "openSpielGameName": {
        "description": "The short_name of the OpenSpiel game to load.",
        "type": "string",
        "default": "PLACEHOLDER_GAME_SHORT_NAME",
    },
    "openSpielGameParameters": {"description": "Game parameters for Open Spiel game.", "type": "object", "default": {}},
    "observationType": {
        "description": "Type of observation string: 'observation' or 'information_state'.",
        "type": "string",
        "default": "observation",
    },
    "useOpenings": {
        "description": "Whether to start from a position in an opening book.",
        "type": "boolean",
        "default": False,
    },
    "useImage": {
        "description": (
            "If true, indicates the observation is intended to be rendered as"
            " an image. Note that currently the agent harness is responsible"
            " for the actual rendering; no image is passed in the observation."
        ),
        "type": "boolean",
        "default": False,
    },
    "includeLegalActions": {
        "description": (
            "If true, include legalActions and legalActionStrings in observations. "
            "Defaults to false since these can be derived from serializedGameAndState."
        ),
        "type": "boolean",
        "default": False,
    },
    "seed": {
        "description": "Integer currently only used for selecting starting position.",
        "type": "number",
    },
    "initialActions": {
        "description": "Actions applied to initial state before play begins to set up starting position.",
        "type": "array",
        "items": {"type": "integer"},
    },
    "loadPresetHands": {
        "description": "Repeated poker only. Load preset hand chance actions from preset_hands.jsonl.",
        "type": "boolean",
        "default": False,
    },
    "presetHands": {
        "description": (
            "Repeated poker only. List of per-hand chance action sequences to use instead of random chance."
        ),
        "type": "array",
        "items": {
            "type": "array",
            "items": {"type": "integer"},
        },
    },
    "metadata": {"description": "Arbitrary metadata.", "type": "object", "default": {}},
}

OBSERVATION_SPEC_TEMPLATE = {
    "properties": {
        "openSpielGameString": {"description": "Full game string including parameters.", "type": "string"},
        "openSpielGameName": {"description": "Short name of the OpenSpiel game.", "type": "string"},
        "observationString": {"description": "String representation of state.", "type": "string"},
        "legalActions": {
            "description": "List of OpenSpiel legal action integers. Only included if includeLegalActions is true.",
            "type": "array",
            "items": {"type": "integer"},
        },
        "legalActionStrings": {
            "description": "List of OpenSpiel legal action strings. Only included if includeLegalActions is true.",
            "type": "array",
            "items": {"type": "string"},
        },
        "currentPlayer": {"description": "ID of player whose turn it is.", "type": "integer"},
        "playerId": {"description": "ID of the agent receiving this observation.", "type": "integer"},
        "isTerminal": {"description": "Boolean indicating game end.", "type": "boolean"},
        "serializedGameAndState": {
            "description": "Enables reconstructing the Game and State objects.",
            "type": "string",
        },
        "remainingOverageTime": 60,
        "step": 0,
    },
    "default": {},
}

ACTION_SPEC_TEMPLATE = {
    "description": "Action object MUST contain a field `submission`, and MAY contain arbitrary additional information.",
    "type": "object",
    "default": {"submission": -1},
}

ENV_SPEC_TEMPLATE = {
    "name": "PLACEHOLDER_NAME",
    "title": "PLACEHOLDER_TITLE",
    "description": DEFAULT_DESCRIPTION,
    "version": "0.1.0",
    "agents": ["PLACEHOLDER_NUM_AGENTS"],
    "configuration": CONFIGURATION_SPEC_TEMPLATE,
    "observation": OBSERVATION_SPEC_TEMPLATE,
    "action": ACTION_SPEC_TEMPLATE,
    "reward": {"type": ["number"], "default": 0.0},
}


def _get_initial_actions(
    configuration: dict[str, Any],
) -> tuple[list[int], dict[str, Any]]:
    initial_actions = configuration.get("initialActions", [])
    if initial_actions:
        if configuration.get("useOpenings"):
            raise ValueError("Cannot set both useOpenings and initialActions.")
        else:
            return initial_actions, {}
    if not configuration.get("useOpenings"):
        return [], {}
    seed = configuration.get("seed", None)
    if seed is None:
        raise ValueError("Must provide seed if useOpenings is True.")
    openings_path = pathlib.Path(
        GAMES_DIR,
        configuration.get("openSpielGameName"),
        "openings.jsonl",
    )
    if not openings_path.is_file():
        raise ValueError(f"No opening file found at {openings_path}")
    with open(openings_path, "r", encoding="utf-8") as f:
        openings = f.readlines()
        opening = json.loads(openings[seed % len(openings)])
        initial_actions = opening.pop("initialActions")
        return initial_actions, opening


# TODO(jhtschultz): General method for handling state setup functions.
def _get_preset_hands(configuration: dict[str, Any]) -> list[list[int]]:
    preset_hands = configuration.get("presetHands")
    if not preset_hands:
        return []
    if configuration.get("useOpenings"):
        raise ValueError("Cannot set both useOpenings and presetHands.")
    if (
        configuration.get("loadPresetHands")
        and "presetHands" in configuration
        and not configuration.get("_presetHandsLoaded")
    ):
        raise ValueError("Cannot set both loadPresetHands and presetHands.")
    if configuration.get("initialActions"):
        raise ValueError("Cannot set both initialActions and presetHands.")
    game_name = configuration.get("openSpielGameName")
    if game_name != "repeated_poker":
        raise ValueError(f"presetHands only supported for repeated_poker, not {game_name}.")
    validated: list[list[int]] = []
    for hand_index, hand in enumerate(preset_hands):
        if not isinstance(hand, list):
            raise ValueError(f"presetHands[{hand_index}] must be a list of integers.")
        if any(not isinstance(action, int) for action in hand):
            raise ValueError(f"presetHands[{hand_index}] must contain only integers.")
        validated.append(list(hand))
    return validated


def _get_image_config(configuration: dict[str, Any]) -> dict[str, Any]:
    use_image = configuration.get("useImage", None)
    if use_image is None:
        raise ValueError("_get_image_config called but useImage missing from env config.")
    if not use_image:
        raise ValueError("_get_image_config called but useImage is False.")
    seed = configuration.get("seed", None)
    if seed is None:
        raise ValueError("Must provide seed if useImage is True.")
    image_config_path = pathlib.Path(
        GAMES_DIR,
        configuration.get("openSpielGameName"),
        "image_config.jsonl",
    )
    if not image_config_path.is_file():
        raise ValueError(f"No image config file found at {image_config_path}")
    with open(image_config_path, "r", encoding="utf-8") as f:
        image_configs = f.readlines()
        image_config = json.loads(image_configs[seed % len(image_configs)])
        return image_config


def _load_preset_hands_from_file(configuration: dict[str, Any]) -> list[list[int]]:
    if configuration.get("openSpielGameName") != "repeated_poker":
        raise ValueError("loadPresetHands only supported for repeated_poker.")
    seed = configuration.get("seed", None)
    if seed is None:
        raise ValueError("Must provide seed if loadPresetHands is True.")
    preset_path = pathlib.Path(
        GAMES_DIR,
        configuration.get("openSpielGameName"),
        "preset_hands.jsonl",
    )
    if not preset_path.is_file():
        raise ValueError(f"No preset hands file found at {preset_path}")
    with open(preset_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    if not lines:
        raise ValueError(f"Preset hands file at {preset_path} is empty.")
    entry = json.loads(lines[seed % len(lines)])
    preset_hands = entry.get("presetHands")
    if not preset_hands:
        raise ValueError("Preset hands entry missing presetHands data.")
    return preset_hands


def _get_preset_chance_action(
    env: core.Environment,
    os_state: pyspiel.State,
    outcomes: tuple[int, ...],
) -> int | None:
    preset_state = env.info.get("presetHandsState")
    if not preset_state:
        return None
    hand_idx = len(os_state.acpc_hand_histories())
    current_hand_index: int = preset_state["current_hand_index"]
    hands: list[tuple[int, ...]] = preset_state["hands"]
    next_index: list[int] = preset_state["next_index"]
    if hand_idx > current_hand_index:
        preset_state["current_hand_index"] = hand_idx
    if hand_idx >= len(hands):
        raise ValueError(f"Ran out of presetHands entries while attempting to start hand {hand_idx}.")
    hand_actions = hands[hand_idx]
    action_pos = next_index[hand_idx]
    if action_pos >= len(hand_actions):
        raise ValueError(f"presetHands[{hand_idx}] does not contain enough chance actions for the hand.")
    next_action = hand_actions[action_pos]
    if next_action not in outcomes:
        raise ValueError(
            f"presetHands[{hand_idx}] specified chance action {next_action} which is not available in the current chance outcomes."
        )
    next_index[hand_idx] = action_pos + 1
    return next_action


# --- Core step logic ---


def interpreter(
    state: list[utils.Struct],
    env: core.Environment,
    logs: list[dict[str, Any]],
) -> list[utils.Struct]:
    """Updates environment using player responses and returns new observations."""
    kaggle_state = state  # Not to be confused with OpenSpiel state.
    del state

    # TODO(jhtschultz): Test reset behavior. Currently containers are restarted
    # after each episode.
    if env.done:
        return kaggle_state

    # --- Get and maybe initialize game and state on the env object ---
    if not hasattr(env, "os_game"):
        game_string = env.configuration.get("openSpielGameString")
        game_name = env.configuration.get("openSpielGameName")

        # Load base game from string to get its parameters
        base_game = pyspiel.load_game(game_string)
        base_params = base_game.get_parameters()

        # Find user-provided params by comparing config to spec defaults
        config_params = env.configuration.get("openSpielGameParameters", {})
        default_params = env.specification.configuration.openSpielGameParameters.get("default", {})
        user_params = {k: v for k, v in config_params.items() if config_params.get(k) != default_params.get(k)}

        # Deprecated: use openSpielGameParameters.max_num_hands instead
        if env.configuration.get("setNumHands", None):
            warnings.warn(
                "setNumHands is deprecated. Use openSpielGameParameters={'max_num_hands': N} instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if "repeated_poker" not in game_name:
                raise ValueError(f"setNumHands only supported for repeated_poker, not {game_name}")
            user_params["max_num_hands"] = env.configuration.get("setNumHands")

        # Merge: base params from string, then user params override
        merged_params = {**base_params, **user_params}

        # Load the game with merged parameters
        env.os_game = pyspiel.load_game(game_name, merged_params)

        # Check if a proxy exists for this game and use it instead
        proxy_path = GAMES_DIR / game_name / f"{game_name}_proxy.py"
        if proxy_path.is_file():
            env.os_game = pyspiel.load_game(game_name + "_proxy", env.os_game.get_parameters())

        # Store the resolved game string (after merging parameters)
        env.info["openSpielGameStringResolved"] = str(env.os_game)
    if not hasattr(env, "os_state"):
        env.os_state = env.os_game.new_initial_state()
    if "actionHistory" not in env.info:
        env.info["actionHistory"] = []
        env.info["moveDurations"] = []
        initial_actions, metadata = _get_initial_actions(env.configuration)
        if env.configuration.get("loadPresetHands", False):
            if env.configuration.get("presetHands"):
                raise ValueError("Cannot provide presetHands when loadPresetHands is True.")
            preset_hands_from_file = _load_preset_hands_from_file(env.configuration)
            env.configuration["presetHands"] = preset_hands_from_file
            env.configuration["_presetHandsLoaded"] = True
        preset_hands = _get_preset_hands(env.configuration)
        env.configuration.pop("_presetHandsLoaded", None)
        if initial_actions:
            env.info["initialActions"] = initial_actions
            env.info["openingMetadata"] = metadata
            for action in initial_actions:
                env.os_state.apply_action(action)
                env.info["actionHistory"].append(str(action))
        if preset_hands:
            env.info["presetHands"] = copy.deepcopy(preset_hands)
            env.info["presetHandsState"] = {
                "hands": [tuple(hand) for hand in preset_hands],
                "next_index": [0 for _ in preset_hands],
                "current_hand_index": 0,
            }
    if env.configuration.get("useImage", False):
        env.configuration["imageConfig"] = _get_image_config(env.configuration)

    os_game = env.os_game
    os_state = env.os_state
    num_players = os_game.num_players()

    # TODO(jhtschultz): Test reset behavior.
    is_initial_step = len(env.steps) == 1
    if is_initial_step and os_state.is_terminal():
        env.os_state = os_game.new_initial_state()
        os_state = env.os_state

    # --- Apply agent action ---
    acting_agent = os_state.current_player()
    action_submitted: int | None = None
    action_submitted_to_string: str | None = None
    action_applied: int | None = None
    move_duration: float | None = None
    if is_initial_step:
        pass
    elif 0 <= acting_agent < num_players:
        if kaggle_state[acting_agent]["status"] != "ACTIVE":
            pass
        else:
            action_submitted = kaggle_state[acting_agent]["action"]["submission"]
            if action_submitted in os_state.legal_actions():
                action_submitted_to_string = os_state.action_to_string(action_submitted)
                os_state.apply_action(action_submitted)
                action_applied = action_submitted
                env.info["actionHistory"].append(str(action_applied))
            elif action_submitted == AGENT_ERROR_ACTION:
                kaggle_state[acting_agent]["status"] = "ERROR"
            else:
                kaggle_state[acting_agent]["status"] = "INVALID"
            try:
                if "duration" in logs[acting_agent]:
                    move_duration = round(logs[acting_agent]["duration"], 3)
                    env.info["moveDurations"].append(move_duration)
                else:
                    env.info["moveDurations"].append(None)
            except Exception:
                pass  # No logs when stepping the env manually.

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
        preset_action = _get_preset_chance_action(env, os_state, outcomes)
        if preset_action is not None:
            chance_action = preset_action
        else:
            chance_action = np.random.choice(outcomes, p=probs)
        os_state.apply_action(chance_action)
        env.info["actionHistory"].append(str(chance_action))

    # --- Update agent states ---
    agent_error = any(kaggle_state[player_id]["status"] in ["TIMEOUT", "ERROR"] for player_id in range(num_players))
    if agent_error:
        _log.info("AGENT ERROR DETECTED")

    invalid_action = any(kaggle_state[player_id]["status"] == "INVALID" for player_id in range(num_players))
    if invalid_action:
        _log.info("INVALID ACTION DETECTED")

    status: str | None = None
    for player_id, agent_state in enumerate(kaggle_state):
        reward = None
        if agent_error:
            # Set all agent statuses to ERROR in order not to score episode. Preserve
            # TIMEOUT which has the same effect.
            if agent_state["status"] == "TIMEOUT":
                status = "TIMEOUT"
            else:
                status = "ERROR"
        elif invalid_action:
            if agent_state["status"] == "INVALID":
                reward = DEFAULT_INVALID_ACTION_REWARD
            else:
                reward = -DEFAULT_INVALID_ACTION_REWARD
            status = "DONE"
        elif os_state.is_terminal():
            status = "DONE"
            reward = os_state.returns()[player_id]
        elif os_state.current_player() == player_id:
            status = "ACTIVE"
            if not os_state.legal_actions(player_id):
                raise ValueError(f"Active agent {player_id} has no legal actions in state {os_state}.")
        else:
            status = "INACTIVE"
        assert status is not None

        info_dict = {}
        if acting_agent == player_id:
            info_dict["actionSubmitted"] = action_submitted
            info_dict["actionSubmittedToString"] = action_submitted_to_string
            info_dict["actionApplied"] = action_applied
            info_dict["timeTaken"] = move_duration
            info_dict["agentSelfReportedStatus"] = (
                kaggle_state[acting_agent]["action"].get("status")
                if kaggle_state[acting_agent]["action"]
                else "unknown"
            )

        # Get observation string based on game's observation type
        if env.configuration.get("observationType") == "information_state":
            obs_string = os_state.information_state_string(player_id)
        else:
            obs_string = os_state.observation_string(player_id)

        obs_update_dict = {
            "observationString": obs_string,
            "currentPlayer": os_state.current_player(),
            "playerId": player_id,
            "isTerminal": os_state.is_terminal(),
            "serializedGameAndState": pyspiel.serialize_game_and_state(os_game, os_state),
        }
        if env.configuration.get("includeLegalActions", False):
            obs_update_dict["legalActions"] = os_state.legal_actions(player_id)
            obs_update_dict["legalActionStrings"] = [
                os_state.action_to_string(action) for action in os_state.legal_actions(player_id)
            ]
        if "imageConfig" in env.configuration:
            obs_update_dict["imageConfig"] = env.configuration["imageConfig"]

        # Apply updates
        for k, v in obs_update_dict.items():
            setattr(agent_state.observation, k, v)
        agent_state["reward"] = reward
        agent_state["info"] = info_dict
        agent_state["status"] = status

    return kaggle_state


# --- Rendering ---


def renderer(state: list[utils.Struct], env: core.Environment) -> str:
    """Kaggle environment text renderer."""
    if hasattr(env, "os_state"):
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
        typeof currentStepData[agentObsIndex].observation.observationString === 'string') {
        obsString = currentStepData[agentObsIndex].observation.observationString;
    } 
    // Fallback to initial step if current is unavailable (e.g. very first render call)
    else if (step === 0 && environment.steps[0] && environment.steps[0][agentObsIndex] && 
             environment.steps[0][agentObsIndex].observation &&
             typeof environment.steps[0][agentObsIndex].observation.observationString === 'string') {
        obsString = environment.steps[0][agentObsIndex].observation.observationString;
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
    open_spiel_short_name: str, base_path_for_custom_renderers: pathlib.Path, default_renderer_func: Callable[[], str]
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
    legal_actions = observation.get("legalActions")
    if not legal_actions:
        return None
    action = random.choice(legal_actions)
    return {"submission": int(action)}


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
    if game_type.provides_observation_string:
        observation_type = "observation"
    elif game_type.provides_information_state_string:
        observation_type = "information_state"
    else:
        raise ValueError(f"No observation or information state string for game: {game_string}")

    env_spec = copy.deepcopy(ENV_SPEC_TEMPLATE)
    env_spec["name"] = f"open_spiel_{short_name}"
    env_spec["title"] = f"Open Spiel: {short_name}"
    env_spec["agents"] = [game.num_players()]

    env_config = env_spec["configuration"]
    env_config["episodeSteps"] = game.max_history_length() + DEFAULT_STEP_BUFFER
    env_config["openSpielGameString"]["default"] = str(game)
    env_config["openSpielGameName"]["default"] = short_name
    env_config["openSpielGameParameters"]["default"] = game.get_parameters()
    env_config["observationType"]["default"] = observation_type

    env_obs = env_spec["observation"]
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


DEFAULT_UNIVERSAL_POKER_GAME_STRING = (
    "universal_poker("
    "betting=nolimit,"
    "bettingAbstraction=fullgame,"
    "blind=2 1,"
    "firstPlayer=2 1 1 1,"
    "numBoardCards=0 3 1 1,"
    "numHoleCards=2,"
    "numPlayers=2,"
    "numRanks=13,"
    "numRounds=4,"
    "numSuits=4,"
    "stack=200 200,"
    "calcOddsNumSims=1000000)"
)

DEFAULT_REPEATED_POKER_GAME_STRING = (
    "repeated_poker("
    "max_num_hands=100,"
    "reset_stacks=True,"
    "rotate_dealer=True,"
    f"universal_poker_game_string={DEFAULT_UNIVERSAL_POKER_GAME_STRING})"
)

DEFAULT_REPEATED_POKERKIT_GAME_STRING = (
    "python_repeated_pokerkit(bet_size_schedule=,blind_schedule=,"
    "bring_in_schedule=,first_button_player=-1,max_num_hands=100,"
    "pokerkit_game_params=python_pokerkit_wrapper(blinds=1 2,num_players=2,"
    "stack_sizes=200 200,variant=NoLimitTexasHoldem),reset_stacks=True,"
    "rotate_dealer=True)"
)

GAMES_LIST = [
    "backgammon",
    "checkers",
    "chess",
    "connect_four",
    "gin_rummy",
    "go(board_size=9)",
    "hearts",
    "hex",
    "othello",
    "tic_tac_toe",
    DEFAULT_UNIVERSAL_POKER_GAME_STRING,
    DEFAULT_REPEATED_POKER_GAME_STRING,
    DEFAULT_REPEATED_POKERKIT_GAME_STRING,
]

ENV_REGISTRY = _register_game_envs(GAMES_LIST)
