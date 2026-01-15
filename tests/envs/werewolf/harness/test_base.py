import json
import os
from unittest.mock import MagicMock, patch

import litellm
import pytest
from dotenv import load_dotenv

from kaggle_environments.envs.werewolf.harness.base import LLMWerewolfAgent

load_dotenv()


@pytest.mark.skip("Require the key to run test.")
def test_vertex_ai():
    model = "vertex_ai/deepseek-ai/deepseek-r1-0528-maas"
    file_path = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
    with open(file_path, "r") as file:
        vertex_credentials = json.load(file)

    vertex_credentials_json = json.dumps(vertex_credentials)

    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.7,
        vertex_ai_project=os.environ["VERTEXAI_PROJECT"],
        vertex_ai_location=os.environ["VERTEXAI_LOCATION"],
        vertex_credentials=vertex_credentials_json,
    )
    print(response)


@pytest.mark.skip("Require the key to run test.")
def test_together():
    model = "together_ai/deepseek-ai/DeepSeek-R1"
    response = litellm.completion(model=model, messages=[{"role": "user", "content": "hi"}])
    print(response)


def test_agent_propagates_query_error():
    # Setup
    agent = LLMWerewolfAgent(model_name="fake-model")
    LLMWerewolfAgent.query.retry.sleep = lambda x: None

    with (
        patch("kaggle_environments.envs.werewolf.harness.base.get_raw_observation") as mock_get_raw_obs,
        patch("kaggle_environments.envs.werewolf.harness.base.completion") as mock_completion,
        patch("kaggle_environments.envs.werewolf.harness.base.LLMWerewolfAgent.log_token_usage"),
    ):
        # Mock Observation
        mock_obs_model = MagicMock()
        mock_obs_model.player_id = "player_0"
        mock_obs_model.role = "Villager"
        mock_obs_model.detailed_phase = "DAY_CHAT_AWAIT"
        mock_obs_model.game_state_phase = "Day"
        mock_obs_model.day = 1
        mock_obs_model.team = "Villagers"
        mock_obs_model.all_player_ids = ["player_0", "player_1"]
        mock_obs_model.alive_players = ["player_0", "player_1"]
        mock_obs_model.revealed_players = []

        # Mock Entry to trigger logic
        mock_entry = MagicMock()
        mock_entry.day = 1
        mock_entry.phase = "Day"
        mock_entry.description = "Game started"
        mock_entry.action_field = None
        mock_obs_model.new_player_event_views = [mock_entry]

        mock_get_raw_obs.return_value = mock_obs_model

        # Mock completion to raise exception
        mock_completion.side_effect = Exception("Network Failure")

        # Assert exception is raised
        with pytest.raises(Exception, match="Network Failure"):
            agent("dummy_obs")

        assert mock_completion.call_count == 10


def test_agent_handles_parsing_error():
    # Setup
    agent = LLMWerewolfAgent(model_name="fake-model")

    with (
        patch("kaggle_environments.envs.werewolf.harness.base.get_raw_observation") as mock_get_raw_obs,
        patch("kaggle_environments.envs.werewolf.harness.base.completion") as mock_completion,
        patch("kaggle_environments.envs.werewolf.harness.base.cost_per_token") as mock_cost,
        patch("kaggle_environments.envs.werewolf.harness.base.LLMWerewolfAgent.log_token_usage"),
    ):
        # Mock Observation (same as above)
        mock_obs_model = MagicMock()
        mock_obs_model.player_id = "player_0"
        mock_obs_model.role = "Villager"
        mock_obs_model.detailed_phase = "DAY_CHAT_AWAIT"
        mock_obs_model.game_state_phase = "Day"
        mock_obs_model.day = 1
        mock_obs_model.team = "Villagers"
        mock_obs_model.all_player_ids = ["player_0", "player_1"]
        mock_obs_model.alive_players = ["player_0", "player_1"]
        mock_obs_model.revealed_players = []

        mock_entry = MagicMock()
        mock_entry.day = 1
        mock_entry.phase = "Day"
        mock_entry.description = "Game started"
        mock_entry.action_field = None
        mock_obs_model.new_player_event_views = [mock_entry]

        mock_get_raw_obs.return_value = mock_obs_model
        mock_cost.return_value = (0.0, 0.0)

        # Mock completion to return invalid JSON
        mock_response = {
            "choices": [{"message": {"content": "Invalid JSON"}}],
            "usage": {"completion_tokens": 10, "prompt_tokens": 10},
        }
        mock_model_resp = MagicMock()
        mock_model_resp.__getitem__.side_effect = lambda key: mock_response[key]
        mock_model_resp.get.side_effect = mock_response.get
        mock_model_resp.usage = mock_response["usage"]
        mock_model_resp._hidden_params = {"response_cost": 0.001}
        mock_completion.return_value = mock_model_resp

        # Assert action is NoOp with correct reasoning
        action = agent("dummy_obs")
        # When an action is serialized, it returns a dict.
        # For NoOpAction, it should have a 'reasoning' field if included in serialization.
        # However, the serialize() method on BaseAction might exclude some fields or rename them.
        # Let's check the keys first.
        print(f"Action keys: {action.keys()}")

        assert action["action_type"] == "NoOpAction"
        assert "Fell back to NoOp after multiple parsing failures" in action["kwargs"]["reasoning"]


def test_cost_and_token_tracking():
    agent = LLMWerewolfAgent(model_name="test-model")

    with (
        patch("kaggle_environments.envs.werewolf.harness.base.get_raw_observation") as mock_get_raw_obs,
        patch("kaggle_environments.envs.werewolf.harness.base.completion") as mock_completion,
        patch("kaggle_environments.envs.werewolf.harness.base.cost_per_token") as mock_cost,
        patch("kaggle_environments.envs.werewolf.harness.base.LLMWerewolfAgent.log_token_usage"),
    ):
        # Mock Observation
        mock_obs_model = MagicMock()
        mock_obs_model.player_id = "player_0"
        mock_obs_model.role = "Villager"
        mock_obs_model.detailed_phase = "DAY_CHAT_AWAIT"
        mock_obs_model.game_state_phase = "Day"
        mock_obs_model.day = 1
        mock_obs_model.team = "Villagers"
        mock_obs_model.all_player_ids = ["player_0", "player_1"]
        mock_obs_model.alive_players = ["player_0", "player_1"]
        mock_obs_model.revealed_players = []

        mock_entry = MagicMock()
        mock_entry.day = 1
        mock_entry.phase = "Day"
        mock_entry.description = "Day 1 starts"
        mock_entry.action_field = None
        mock_obs_model.new_player_event_views = [mock_entry]

        mock_get_raw_obs.return_value = mock_obs_model

        # Setup Mocks
        mock_cost.return_value = (0.001, 0.002)

        # Mock Response Object
        mock_response = MagicMock()
        usage_dict = {"prompt_tokens": 150, "completion_tokens": 50}
        mock_response.__getitem__.side_effect = lambda k: usage_dict if k == "usage" else MagicMock()
        mock_response._hidden_params = {"response_cost": 0.05}

        chat_action_dict = {"perceived_threat_level": "SAFE", "reasoning": "Just testing.", "message": "Hello world"}
        mock_choice = MagicMock()
        mock_choice.get.return_value = {"content": f"```json\n{json.dumps(chat_action_dict)}\n```"}
        mock_response.get.return_value = [mock_choice]

        mock_completion.return_value = mock_response

        # --- Execution ---
        action_serialized = agent("dummy_obs")

        # --- Verification ---
        kwargs = action_serialized.get("kwargs", {})

        assert kwargs.get("cost") == 0.05
        assert kwargs.get("prompt_tokens") == 150
        assert kwargs.get("completion_tokens") == 50

        # Verify Tracker State
        tracker = agent.cost_tracker
        assert tracker.query_token_cost.total_costs_usd == 0.05
        assert tracker.prompt_token_cost.total_tokens == 150
        assert tracker.completion_token_cost.total_tokens == 50

        # --- Second Call (Accumulation Check) ---
        mock_response_2 = MagicMock()
        usage_dict_2 = {"prompt_tokens": 10, "completion_tokens": 10}
        mock_response_2.__getitem__.side_effect = lambda k: usage_dict_2 if k == "usage" else MagicMock()
        mock_response_2._hidden_params = {"response_cost": 0.01}
        mock_choice_2 = MagicMock()
        mock_choice_2.get.return_value = {"content": f"```json\n{json.dumps(chat_action_dict)}\n```"}
        mock_response_2.get.return_value = [mock_choice_2]

        mock_completion.return_value = mock_response_2

        action_serialized_2 = agent("dummy_obs")
        kwargs_2 = action_serialized_2.get("kwargs", {})

        # Use almost equal for float comparison
        assert abs(kwargs_2.get("cost") - 0.01) < 1e-9
        assert kwargs_2.get("prompt_tokens") == 10
        assert kwargs_2.get("completion_tokens") == 10


@pytest.mark.skip("Require the key to run test.")
def test_gemini():
    # model = "vertex_ai/gemini-3-pro-preview"
    model = "openrouter/google/gemini-3-pro-preview"
    response = litellm.completion(model=model, messages=[{"role": "user", "content": "hi"}])
    print(response)
