import json
import shutil
import subprocess

import pytest

from kaggle_environments import make
from kaggle_environments.envs.battlecast_arena import battlecast_arena


PARTY = {
    "characters": [
        {"slot": 1},
        {"slot": 2},
        {"slot": 3},
        {"slot": 4},
    ],
}


def party_agent(observation, configuration):
    if observation.get("phase") != "combat":
        return {"party": PARTY}
    return "end_turn"


def test_adapter_keeps_state_hidden_and_translates_setup(monkeypatch):
    calls = []

    def fake_bridge(payload, timeout, **kwargs):
        calls.append(payload)
        if "mode" not in payload:
            return {"valid": True}
        if payload["mode"] == "init":
            return {
                "state": {"version": 1, "turn": 0},
                "observations": {
                    "red": {"phase": "combat", "round": 1, "activeCreatureIds": ["r"], "legalActions": ["end_turn"]},
                    "blue": {"phase": "combat", "round": 1, "activeCreatureIds": [], "legalActions": []},
                },
                "statuses": {"red": "ACTIVE", "blue": "INACTIVE"},
                "rewards": {"red": 0, "blue": 0},
            }
        return {
            "state": {"version": 1, "turn": 1},
            "observations": {"red": {"phase": "complete", "round": 1}, "blue": {"phase": "complete", "round": 1}},
            "statuses": {"red": "DONE", "blue": "DONE"},
            "rewards": {"red": 1, "blue": -1},
        }

    monkeypatch.setattr(battlecast_arena, "_call_bridge", fake_bridge)
    env = make("battlecast_arena")
    env.run([party_agent, party_agent])

    assert [call["mode"] for call in calls if "mode" in call] == ["init", "step"]
    assert "engineState" not in env._Environment__get_shared_state(0).observation
    assert env.state[0].observation.engineState == {"version": 1, "turn": 1}
    assert env.configuration.seed is None
    assert calls[2]["seed"] == 1


def test_invalid_party_forfeits_only_its_submitter(monkeypatch):
    def fake_bridge(payload, timeout, **kwargs):
        if payload.get("team") == "blue":
            raise battlecast_arena.PlayerError("bad party")
        return {"valid": True}

    monkeypatch.setattr(battlecast_arena, "_call_bridge", fake_bridge)
    env = make("battlecast_arena")
    env.run([party_agent, party_agent])
    assert [state.status for state in env.state] == ["DONE", "DONE"]
    assert [state.reward for state in env.state] == [1, -1]


def test_bridge_failure_marks_both_players_error(monkeypatch):
    def unavailable(*args, **kwargs):
        raise battlecast_arena.BridgeError("missing battlecast-engine")

    monkeypatch.setattr(battlecast_arena, "_call_bridge", unavailable)
    env = make("battlecast_arena")
    env.run([party_agent, party_agent])
    assert [state.status for state in env.state] == ["ERROR", "ERROR"]


def test_incomplete_bridge_response_is_rejected(monkeypatch):
    class Result:
        returncode = 0
        stdout = '{"state": []}'
        stderr = ""

    monkeypatch.setattr(battlecast_arena.subprocess, "run", lambda *args, **kwargs: Result())
    with pytest.raises(battlecast_arena.BridgeError, match="incomplete response"):
        battlecast_arena._call_bridge({"mode": "init"}, 1)


def test_arena_configuration_locks_the_round_cap():
    assert battlecast_arena.specification["configuration"]["roundCap"]["enum"] == [20]


def test_installed_bridge_initializes_a_non_overlapping_party():
    command = shutil.which("battlecast-engine")
    if command is None:
        pytest.skip("battlecast-engine is installed in the competition image")
    payload = {
        "version": 1,
        "mode": "init",
        "seed": 1,
        "mapId": "open-arena",
        "roundCap": 20,
        "redParty": PARTY,
        "blueParty": PARTY,
    }
    result = subprocess.run(
        [command, "arena", "kaggle-step"],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    response = json.loads(result.stdout)
    creatures = response["state"]["battleState"]["creatures"]
    assert len({(creature["position"]["x"], creature["position"]["y"]) for creature in creatures}) == 8

    active_team = next(team for team in ("red", "blue") if response["statuses"][team] == "ACTIVE")
    step = subprocess.run(
        [command, "arena", "kaggle-step"],
        input=json.dumps({
            "version": 1,
            "mode": "step",
            "state": response["state"],
            "team": active_team,
            "action": "end_turn",
        }),
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert step.returncode == 0, step.stderr
    stepped = json.loads(step.stdout)
    assert stepped["state"]["version"] == 1
    assert stepped["state"] != response["state"]
