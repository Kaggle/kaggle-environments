"""Kaggle adapter for the versioned ``battlecast-engine arena kaggle-step`` protocol."""

import json
import subprocess
from os import path

from kaggle_environments.utils import resolve_episode_seed


BRIDGE_PROTOCOL_VERSION = 1
BRIDGE_COMMAND = ("battlecast-engine", "arena")


class BridgeError(RuntimeError):
    """Raised when the BattleCast engine cannot produce a valid bridge response."""


class PlayerError(BridgeError):
    """Raised only for a player-supplied invalid party or action."""


def interpreter(state, env):
    if env.done and state[0].observation.phase != "setup":
        return state
    invalid = next((i for i, player in enumerate(state) if player.status in {"INVALID", "TIMEOUT", "ERROR"}), None)
    if invalid is not None:
        return _forfeit(state, invalid, "Invalid setup or combat action.")
    try:
        if state[0].observation.phase == "setup":
            episode_seed = resolve_episode_seed(env)
            if not all(isinstance(player.action, dict) and "party" in player.action for player in state):
                return _setup_observation(state)
            for index, team in enumerate(("red", "blue")):
                try:
                    _validate_party(team, state[index].action["party"], env.configuration)
                except PlayerError as error:
                    return _forfeit(state, index, str(error))
            response = _call_bridge(
                {
                    "version": BRIDGE_PROTOCOL_VERSION,
                    "mode": "init",
                    "seed": episode_seed,
                    "mapId": env.configuration.mapId,
                    "roundCap": env.configuration.roundCap,
                    "redParty": state[0].action["party"],
                    "blueParty": state[1].action["party"],
                },
                env.configuration.bridgeTimeout,
            )
        else:
            active = _active_index(state)
            response = _call_bridge(
                {
                    "version": BRIDGE_PROTOCOL_VERSION,
                    "mode": "step",
                    "state": state[0].observation.engineState,
                    "team": "red" if active == 0 else "blue",
                    "action": state[active].action,
                },
                env.configuration.bridgeTimeout,
                player_error=True,
            )
        return _apply_response(state, response)
    except PlayerError as error:
        return _forfeit(state, _active_index(state), str(error))
    except BridgeError as error:
        return _bridge_error(state, str(error))


def _setup_observation(state):
    for player in state:
        player.observation = {
            "remainingOverageTime": player.observation.remainingOverageTime,
            "phase": "setup",
            "round": 0,
            "activeCreatureIds": [],
            "legalActions": [],
            "setup": {"action": {"party": "four validated level-5 heroes"}},
        }
        player.status = "ACTIVE"
        player.reward = 0
    return state


def _validate_party(team, party, configuration):
    payload = {"team": team, "party": party}
    _call_bridge(payload, configuration.bridgeTimeout, command=BRIDGE_COMMAND + ("validate-party",), player_error=True, validate_only=True)


def _active_index(state):
    return next((index for index, player in enumerate(state) if player.status == "ACTIVE"), 0)


def _call_bridge(payload, timeout, command=None, player_error=False, validate_only=False):
    if command is None:
        command = BRIDGE_COMMAND + ("kaggle-step",)
    try:
        result = subprocess.run(
            command,
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise BridgeError(f"BattleCast bridge unavailable: {error}") from error
    if result.returncode != 0:
        message = result.stderr.strip() or "BattleCast bridge failed."
        if player_error and message.startswith("INVALID_REQUEST:"):
            raise PlayerError(message.removeprefix("INVALID_REQUEST:").strip())
        raise BridgeError(message)
    try:
        response = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise BridgeError("BattleCast bridge returned invalid JSON.") from error
    if validate_only:
        if response != {"valid": True}:
            raise BridgeError("BattleCast bridge returned an invalid validation response.")
        return response
    if (
        not isinstance(response, dict)
        or not isinstance(response.get("state"), dict)
        or response["state"].get("version") != BRIDGE_PROTOCOL_VERSION
        or not all(isinstance(response.get(key), dict) for key in ("observations", "statuses", "rewards"))
    ):
        raise BridgeError("BattleCast bridge returned an incomplete response.")
    return response


def _apply_response(state, response):
    updates = []
    for index, team in enumerate(("red", "blue")):
        observation = response["observations"].get(team)
        status = response["statuses"].get(team)
        reward = response["rewards"].get(team)
        if not isinstance(observation, dict) or status not in {"ACTIVE", "INACTIVE", "DONE"} or reward not in {-1, 0, 1}:
            raise BridgeError("BattleCast bridge returned an invalid team response.")
        updates.append((index, observation, status, reward))
    for index, observation, status, reward in updates:
        state[index].observation = {"remainingOverageTime": state[index].observation.remainingOverageTime, **observation}
        state[index].status = status
        state[index].reward = reward
    state[0].observation["engineState"] = response["state"]
    return state


def _forfeit(state, loser, message):
    for index, player in enumerate(state):
        player.status = "DONE"
        player.reward = -1 if index == loser else 1
        player.info["bridgeError"] = message
    return state


def _bridge_error(state, message):
    for player in state:
        player.status = "ERROR"
        player.reward = 0
        player.info["bridgeError"] = message
    return state


dirpath = path.dirname(__file__)
with open(path.join(dirpath, "battlecast_arena.json")) as json_file:
    specification = json.load(json_file)


def renderer(state, env):
    observation = state[0].observation
    return f"BattleCast Arena: {observation.phase}, round {observation.get('round', 0)}"


def html_renderer():
    return ""


agents = {}
