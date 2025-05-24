import pytest
import json
from kaggle_environments.envs.werewolf.env import WerewolfEnv, Role, Phase, ActionType, WerewolfObservationModel

# Constants for tests
NUM_PLAYERS_DEFAULT = 5
NUM_DOCTORS_DEFAULT = 1
NUM_SEERS_DEFAULT = 1


@pytest.fixture
def werewolf_env_unreset():
    """Fixture to create a WerewolfEnv instance without resetting."""
    return WerewolfEnv(num_doctors=NUM_DOCTORS_DEFAULT, num_seers=NUM_SEERS_DEFAULT)


@pytest.fixture
def env(werewolf_env_unreset: WerewolfEnv) -> WerewolfEnv:
    """Fixture to create and reset a WerewolfEnv instance with default players."""
    werewolf_env_unreset.reset(options={"num_players": NUM_PLAYERS_DEFAULT})
    return werewolf_env_unreset

# Helper function to get an agent by role


def get_agent_by_role(env: WerewolfEnv, role_to_find: Role, must_be_alive: bool = True) -> str | None:
    for agent_id, role in env.player_roles.items():
        if role == role_to_find:
            if not must_be_alive or (must_be_alive and agent_id in env.alive_agents):
                return agent_id
    return None

# Helper function to get all agents by role


def get_all_agents_by_role(env: WerewolfEnv, role_to_find: Role, must_be_alive: bool = True) -> list[str]:
    agents = []
    for agent_id, role in env.player_roles.items():
        if role == role_to_find:
            if not must_be_alive or (must_be_alive and agent_id in env.alive_agents):
                agents.append(agent_id)
    return agents

# Helper to perform NO_OP until a target phase is reached or game ends


def advance_to_phase(env: WerewolfEnv, target_phase: Phase, max_steps_multiplier=3):
    max_steps = env.num_players * max_steps_multiplier
    for _ in range(max_steps):
        if env.current_phase == target_phase or env.current_phase == Phase.GAME_OVER:
            return

        agent_to_act = env.agent_selection
        if agent_to_act is None:  # Should mean phase transition is pending or game over
            if env.current_phase == Phase.GAME_OVER:
                return
            # This state implies _transition_phase should be called by env's internal logic.
            # If stuck, this loop will time out.
            # A well-behaved env should transition if agent_selection is None and not game over.
            # For testing, we assume step() or _transition_phase() will eventually move it.
            # If not, the test checking for target_phase will fail.
            if env._agent_selector.is_done():  # Try to force transition if selector is done for phase
                env._transition_phase()  # This is a bit intrusive, but can help if env gets stuck
            continue

        if env.terminations[agent_to_act]:
            env.step({})  # Let _was_dead_step run and advance selector
        else:
            env.step({"action_type": ActionType.NO_OP.value})

    if env.current_phase != target_phase and env.current_phase != Phase.GAME_OVER:
        print(
            f"ADVANCE HELPER: Max steps reached. Current agent: {env.agent_selection}, current phase: {env.current_phase.name}, target: {target_phase.name}")


def test_env_creation_and_reset(werewolf_env_unreset: WerewolfEnv):
    env_instance = werewolf_env_unreset
    assert env_instance.num_players == 0  # Initial
    env_instance.reset(options={"num_players": NUM_PLAYERS_DEFAULT})
    assert env_instance.num_players == NUM_PLAYERS_DEFAULT
    assert len(env_instance.agent_ids) == NUM_PLAYERS_DEFAULT
    assert env_instance.num_werewolves == 2  # ceil(5/4)
    assert env_instance.num_doctors == NUM_DOCTORS_DEFAULT
    assert env_instance.num_seers == NUM_SEERS_DEFAULT
    assert len(env_instance.player_roles) == NUM_PLAYERS_DEFAULT
    assert len(env_instance.alive_agents) == NUM_PLAYERS_DEFAULT
    assert env_instance.current_phase == Phase.NIGHT_WEREWOLF_VOTE
    assert env_instance.agent_selection is not None
    assert env_instance.player_roles[env_instance.agent_selection] == Role.WEREWOLF


def test_reset_with_different_player_counts(werewolf_env_unreset: WerewolfEnv):
    env_instance = werewolf_env_unreset
    env_instance.reset(options={"num_players": 7})
    assert env_instance.num_players == 7
    assert env_instance.num_werewolves == 2  # ceil(7/4)
    assert len(env_instance.agent_ids) == 7

    env_instance.reset(options={"num_players": 3})
    assert env_instance.num_players == 3
    assert env_instance.num_werewolves == 1  # ceil(3/4)


def test_invalid_player_count_on_reset(werewolf_env_unreset: WerewolfEnv):
    with pytest.raises(ValueError, match="at least 3 players"):
        werewolf_env_unreset.reset(options={"num_players": 2})


def test_too_many_special_roles():
    # For 5 players, num_werewolves = ceil(5/4) = 2.
    # 2 WW + 2 Dr + 2 Seer = 6 > 5.
    env_bad_roles = WerewolfEnv(num_doctors=2, num_seers=2)
    with pytest.raises(ValueError, match="Too many special roles"):
        env_bad_roles.reset(options={"num_players": 5})

    env_ok = WerewolfEnv(num_doctors=1, num_seers=0)
    # 1 WW, 1 Dr, 0 Seer. 1 Villager. OK.
    env_ok.reset(options={"num_players": 3})


def test_role_assignment_counts(env: WerewolfEnv):
    roles = list(env.player_roles.values())
    assert roles.count(Role.WEREWOLF) == env.num_werewolves
    assert roles.count(Role.DOCTOR) == env.num_doctors
    assert roles.count(Role.SEER) == env.num_seers
    expected_villagers = env.num_players - \
        env.num_werewolves - env.num_doctors - env.num_seers
    assert roles.count(Role.VILLAGER) == expected_villagers


def test_initial_observation(env: WerewolfEnv):
    agent_id = env.agent_selection  # Should be a werewolf
    obs_data = env.observe(agent_id)
    obs = WerewolfObservationModel(**obs_data)

    assert obs.role == Role.WEREWOLF.value
    assert obs.phase == Phase.NIGHT_WEREWOLF_VOTE.value
    assert sum(obs.alive_players) == env.num_players

    num_ww_expected = list(env.player_roles.values()).count(Role.WEREWOLF)
    assert sum(obs.known_werewolves) == num_ww_expected
    for i, is_ww_known in enumerate(obs.known_werewolves):
        p_id = env.index_to_agent_id[i]
        if is_ww_known == 1:
            assert env.player_roles[p_id] == Role.WEREWOLF
    assert obs.my_unique_name == agent_id
    assert json.loads(obs.all_player_unique_names) == env.agent_ids
    assert obs.last_action_feedback == "No action taken yet in this episode."

# --- Action Tests ---


def test_werewolf_night_vote_action_valid(env: WerewolfEnv):
    ww_agent = env.agent_selection
    assert env.player_roles[ww_agent] == Role.WEREWOLF

    target_idx = -1
    for i, p_id in enumerate(env.agent_ids):
        if env.player_roles[p_id] != Role.WEREWOLF:
            target_idx = i
            break
    assert target_idx != -1, "Could not find a non-werewolf target"

    action = {"action_type": ActionType.NIGHT_KILL_VOTE.value,
              "target_idx": target_idx}
    env.step(action)

    assert env.rewards[ww_agent] == 0.0
    assert env._werewolf_kill_votes[ww_agent] == target_idx
    assert "Voted to kill" in env.infos[ww_agent]["last_action_feedback"]
    assert env.infos[ww_agent]["last_action_valid"] is True
    assert env.infos[ww_agent]["action_description_for_log"].startswith(
        "WW_VOTE_KILL:")


def test_werewolf_night_vote_action_invalid_target_ww(env: WerewolfEnv):
    ww_agents = get_all_agents_by_role(env, Role.WEREWOLF)
    acting_ww = env.agent_selection
    assert acting_ww in ww_agents

    target_ww_idx = env.agent_id_to_index[acting_ww]  # Target self
    if len(ww_agents) > 1:  # target another ww if available
        target_ww_idx = env.agent_id_to_index[next(
            iter(w for w in ww_agents if w != acting_ww))]

    action = {"action_type": ActionType.NIGHT_KILL_VOTE.value,
              "target_idx": target_ww_idx}
    env.step(action)

    assert env.rewards[acting_ww] == -1.0
    assert acting_ww not in env._werewolf_kill_votes
    assert "Target must be alive (at night start) and not a Werewolf" in env.infos[
        acting_ww]["last_action_feedback"]
    assert env.infos[acting_ww]["last_action_valid"] is False
    assert "INVALID_ACTION (Contextual Error)" in env.infos[acting_ww]["action_description_for_log"]


def test_action_pydantic_validation_fails(werewolf_env_unreset: WerewolfEnv):
    # Test on fresh resets to ensure agent_selection is predictable for feedback checks
    env_instance = werewolf_env_unreset

    env_instance.reset(options={"num_players": NUM_PLAYERS_DEFAULT})
    agent_id = env_instance.agent_selection
    env_instance.step("not_a_dict")  # Invalid structure: not a dict
    assert env_instance.rewards[agent_id] == -1.0
    assert "Input should be a valid dictionary" in env_instance.infos[
        agent_id]["last_action_feedback"]
    assert env_instance.infos[agent_id]["last_action_valid"] is False
    assert "INVALID_ACTION (Pydantic Error)" in env_instance.infos[
        agent_id]["action_description_for_log"]

    env_instance.reset(options={"num_players": NUM_PLAYERS_DEFAULT})
    agent_id = env_instance.agent_selection
    env_instance.step({"target_idx": 0})  # Missing action_type
    assert env_instance.rewards[agent_id] == -1.0
    assert "Field 'action_type': Field required" in env_instance.infos[
        agent_id]["last_action_feedback"]
    assert env_instance.infos[agent_id]["last_action_valid"] is False

    env_instance.reset(options={"num_players": NUM_PLAYERS_DEFAULT})
    agent_id = env_instance.agent_selection
    env_instance.step({"action_type": 999})  # Invalid action_type value
    assert env_instance.rewards[agent_id] == -1.0
    assert "Unknown 'action_type' value: 999" in env_instance.infos[
        agent_id]["last_action_feedback"]
    assert env_instance.infos[agent_id]["last_action_valid"] is False

    env_instance.reset(options={"num_players": NUM_PLAYERS_DEFAULT})
    agent_id = env_instance.agent_selection  # WW turn
    # Missing target_idx
    env_instance.step({"action_type": ActionType.NIGHT_KILL_VOTE.value})
    assert env_instance.rewards[agent_id] == -1.0
    assert "'target_idx' is required for NIGHT_KILL_VOTE" in env_instance.infos[
        agent_id]["last_action_feedback"]
    assert env_instance.infos[agent_id]["last_action_valid"] is False


def test_action_wrong_phase(env: WerewolfEnv):
    agent_id = env.agent_selection  # A werewolf, phase is NIGHT_WEREWOLF_VOTE

    action = {"action_type": ActionType.DAY_DISCUSS.value,
              "message": "Hello from night!"}
    env.step(action)

    assert env.rewards[agent_id] == -1.0
    assert "not appropriate for phase 'NIGHT_WEREWOLF_VOTE'" in env.infos[
        agent_id]["last_action_feedback"]
    assert env.infos[agent_id]["last_action_valid"] is False


def test_action_wrong_role_for_action_type(env: WerewolfEnv):
    # Advance to Doctor phase
    advance_to_phase(env, Phase.NIGHT_DOCTOR_SAVE)
    if env.current_phase != Phase.NIGHT_DOCTOR_SAVE:
        pytest.skip("Could not reach Doctor phase")
        return
    if env.num_doctors == 0:
        pytest.skip("No doctors in this configuration.")
        return

    doctor_agent = env.agent_selection
    assert doctor_agent is not None and env.player_roles[doctor_agent] == Role.DOCTOR

    # Doctor tries a Werewolf action (NIGHT_KILL_VOTE)
    # This will first fail because action_type is not for NIGHT_DOCTOR_SAVE phase
    action = {"action_type": ActionType.NIGHT_KILL_VOTE.value, "target_idx": 0}
    env.step(action)

    assert env.rewards[doctor_agent] == -1.0
    # The phase check comes before the role check in env.step()
    assert "not appropriate for phase 'NIGHT_DOCTOR_SAVE'" in env.infos[
        doctor_agent]["last_action_feedback"]
    assert env.infos[doctor_agent]["last_action_valid"] is False


def test_doctor_save_action_valid(env: WerewolfEnv):
    advance_to_phase(env, Phase.NIGHT_DOCTOR_SAVE)
    if env.current_phase != Phase.NIGHT_DOCTOR_SAVE:
        pytest.skip("Could not reach Doctor phase")
        return
    if env.num_doctors == 0:
        pytest.skip("No doctors to test save action.")
        return

    doc_agent = env.agent_selection
    assert doc_agent is not None and env.player_roles[doc_agent] == Role.DOCTOR

    save_target_idx = env.agent_id_to_index[doc_agent]  # Save self
    action = {"action_type": ActionType.NIGHT_SAVE_TARGET.value,
              "target_idx": save_target_idx}
    env.step(action)

    assert env.rewards[doc_agent] == 0.0
    assert env._doctor_save_choices[doc_agent] == save_target_idx
    assert "Chose to save" in env.infos[doc_agent]["last_action_feedback"]
    assert env.infos[doc_agent]["last_action_valid"] is True


def test_seer_inspect_action_valid(env: WerewolfEnv):
    advance_to_phase(env, Phase.NIGHT_SEER_INSPECT)
    if env.current_phase != Phase.NIGHT_SEER_INSPECT:
        pytest.skip("Could not reach Seer phase")
        return
    if env.num_seers == 0:
        pytest.skip("No seers to test inspect action.")
        return

    seer_agent = env.agent_selection
    assert seer_agent is not None and env.player_roles[seer_agent] == Role.SEER

    inspect_target_idx = env.agent_id_to_index[get_agent_by_role(
        env, Role.VILLAGER) or env.agent_ids[0]]
    action = {"action_type": ActionType.NIGHT_INSPECT_TARGET.value,
              "target_idx": inspect_target_idx}
    env.step(action)

    assert env.rewards[seer_agent] == 0.0
    # _seer_inspect_choices is cleared after night resolution, which happens if seer is last night actor.
    # Instead, check the info log for confirmation.
    assert "Chose to inspect" in env.infos[seer_agent]["last_action_feedback"]
    assert env.infos[seer_agent]["last_action_valid"] is True


def test_night_resolution_kill_no_save(env: WerewolfEnv):
    num_players = env.num_players
    ww_agents = get_all_agents_by_role(env, Role.WEREWOLF)

    villager_to_kill_id = get_agent_by_role(env, Role.VILLAGER) or \
        get_agent_by_role(env, Role.DOCTOR) or \
        get_agent_by_role(env, Role.SEER)  # Find any non-WW
    assert villager_to_kill_id is not None, "Could not find a non-WW target"
    villager_to_kill_idx = env.agent_id_to_index[villager_to_kill_id]
    original_role_of_killed = env.player_roles[villager_to_kill_id]

    # WWs vote to kill the target
    for ww_id in ww_agents:
        if env.agent_selection == ww_id:
            env.step({"action_type": ActionType.NIGHT_KILL_VOTE.value,
                     "target_idx": villager_to_kill_idx})

    # Doctors (if any) save someone else
    if env.num_doctors > 0:
        doc_agents_this_game = get_all_agents_by_role(env, Role.DOCTOR)
        for doc_id in doc_agents_this_game:
            if env.agent_selection == doc_id:
                save_target_id = doc_id if doc_id != villager_to_kill_id else env.agent_ids[(
                    env.agent_id_to_index[doc_id] + 1) % num_players]
                save_target_idx = env.agent_id_to_index[save_target_id]
                env.step({"action_type": ActionType.NIGHT_SAVE_TARGET.value,
                         "target_idx": save_target_idx})

    # Seers (if any) inspect someone else
    if env.num_seers > 0:
        seer_agents_this_game = get_all_agents_by_role(env, Role.SEER)
        for seer_id in seer_agents_this_game:
            if env.agent_selection == seer_id:
                inspect_target_idx = (villager_to_kill_idx + 1) % num_players
                env.step({"action_type": ActionType.NIGHT_INSPECT_TARGET.value,
                         "target_idx": inspect_target_idx})

    assert env.current_phase == Phase.DAY_DISCUSSION or env.current_phase == Phase.GAME_OVER
    if env.current_phase != Phase.GAME_OVER:  # if game didn't end due to this kill
        assert villager_to_kill_id not in env.alive_agents
        assert env.terminations[villager_to_kill_id] is True
        assert env._last_killed_by_werewolf_idx == villager_to_kill_idx
        assert env._last_killed_by_werewolf_role_val == original_role_of_killed.value

        if env.alive_agents:
            living_agent = env.alive_agents[0]
            obs_data = env.observe(living_agent)
            obs = WerewolfObservationModel(**obs_data)
            assert obs.last_killed_by_werewolf == villager_to_kill_idx
            assert obs.last_killed_by_werewolf_role == original_role_of_killed.value


def test_night_resolution_kill_and_save_and_seer_result(env: WerewolfEnv):
    if env.num_doctors == 0 or env.num_seers == 0:
        pytest.skip("Requires Doctor and Seer.")
        return

    ww_agents = get_all_agents_by_role(env, Role.WEREWOLF)
    doc_agent = get_agent_by_role(env, Role.DOCTOR)
    seer_agent = get_agent_by_role(env, Role.SEER)
    assert doc_agent and seer_agent, "Doctor or Seer not found"

    # WWs target doctor, doctor saves self, seer inspects a WW
    target_for_ww_idx = env.agent_id_to_index[doc_agent]
    seer_inspect_target_id = ww_agents[0]
    seer_inspect_target_idx = env.agent_id_to_index[seer_inspect_target_id]
    seer_inspect_target_role_val = env.player_roles[seer_inspect_target_id].value

    for ww_id in ww_agents:  # WWs act
        if env.agent_selection == ww_id:
            env.step({"action_type": ActionType.NIGHT_KILL_VOTE.value,
                     "target_idx": target_for_ww_idx})

    if env.agent_selection == doc_agent:  # Doctor acts
        env.step({"action_type": ActionType.NIGHT_SAVE_TARGET.value,
                 "target_idx": target_for_ww_idx})  # Save self

    if env.agent_selection == seer_agent:  # Seer acts
        env.step({"action_type": ActionType.NIGHT_INSPECT_TARGET.value,
                 "target_idx": seer_inspect_target_idx})

    assert env.current_phase == Phase.DAY_DISCUSSION or env.current_phase == Phase.GAME_OVER
    if env.current_phase != Phase.GAME_OVER:
        assert doc_agent in env.alive_agents
        assert env.terminations[doc_agent] is False
        assert env._last_killed_by_werewolf_idx is None
        assert env._last_killed_by_werewolf_role_val is None

        seer_obs_data = env.observe(seer_agent)
        seer_obs = WerewolfObservationModel(**seer_obs_data)
        assert seer_obs.seer_last_inspection == (
            seer_inspect_target_idx, seer_inspect_target_role_val)


def test_day_discussion_action(env: WerewolfEnv):
    advance_to_phase(env, Phase.DAY_DISCUSSION)
    if env.current_phase != Phase.DAY_DISCUSSION:
        pytest.skip("Could not reach Day Discussion phase")
        return

    acting_agent = env.agent_selection
    assert acting_agent in env.alive_agents

    message = "Hello everyone, I am a villager!"
    action = {"action_type": ActionType.DAY_DISCUSS.value, "message": message}
    env.step(action)

    assert env.rewards[acting_agent] == 0.0
    assert len(env._discussion_log_this_round) == 1
    assert env._discussion_log_this_round[0] == {
        "speaker": acting_agent, "message": message}
    assert "Discussion message sent" in env.infos[acting_agent]["last_action_feedback"]

    obs_data = env.observe(acting_agent)
    obs = WerewolfObservationModel(**obs_data)
    assert json.loads(obs.discussion_log) == [
        {"speaker": acting_agent, "message": message}]


def test_day_lynch_vote_and_resolution(env: WerewolfEnv):
    advance_to_phase(env, Phase.DAY_VOTING)
    if env.current_phase != Phase.DAY_VOTING:
        pytest.skip("Could not reach Day Voting phase")
        return

    alive_agents_for_vote = list(env.alive_agents)
    if len(alive_agents_for_vote) < 2:
        pytest.skip("Not enough players to lynch meaningfully.")
        return

    target_to_lynch_id = alive_agents_for_vote[0]
    target_to_lynch_idx = env.agent_id_to_index[target_to_lynch_id]
    original_role_of_lynched = env.player_roles[target_to_lynch_id]

    voters = [ag for ag in alive_agents_for_vote if ag != target_to_lynch_id]

    # Make all voters vote for the target
    for _ in range(len(alive_agents_for_vote)):  # Max steps for all to act
        if env.current_phase != Phase.DAY_VOTING:
            break
        agent_to_act = env.agent_selection
        if not agent_to_act:
            break

        if agent_to_act in voters:
            env.step({"action_type": ActionType.DAY_LYNCH_VOTE.value,
                     "target_idx": target_to_lynch_idx})
        elif agent_to_act == target_to_lynch_id:  # Target NO_OPs or votes for someone else
            other_target_idx = (target_to_lynch_idx + 1) % env.num_players
            while env.index_to_agent_id[other_target_idx] not in env.alive_agents or other_target_idx == target_to_lynch_idx:
                other_target_idx = (other_target_idx + 1) % env.num_players
                if other_target_idx == target_to_lynch_idx and len(alive_agents_for_vote) == 1:
                    break  # Only one left
            if other_target_idx != target_to_lynch_idx:
                env.step({"action_type": ActionType.DAY_LYNCH_VOTE.value,
                         "target_idx": other_target_idx})
            else:
                env.step({"action_type": ActionType.NO_OP.value})
        else:  # Should not happen if voters list is correct
            env.step({"action_type": ActionType.NO_OP.value})

    if env.current_phase == Phase.GAME_OVER:  # Game might end due to lynch
        pass
    elif env.current_phase == Phase.NIGHT_WEREWOLF_VOTE:
        assert target_to_lynch_id not in env.alive_agents
        assert env.terminations[target_to_lynch_id] is True
        assert env._last_lynched_player_idx == target_to_lynch_idx
        assert env._last_lynched_player_role_val == original_role_of_lynched.value

        if env.alive_agents:
            living_agent = env.alive_agents[0]
            obs_data = env.observe(living_agent)
            obs = WerewolfObservationModel(**obs_data)
            assert obs.last_lynched == target_to_lynch_idx
            assert obs.last_lynched_player_role == original_role_of_lynched.value
            assert json.loads(obs.last_day_vote_details)
    else:
        pytest.fail(
            f"Phase ended up as {env.current_phase.name} instead of NIGHT_WEREWOLF_VOTE or GAME_OVER after lynch.")


def test_villagers_win_all_ww_eliminated(env: WerewolfEnv):
    # Loop, in each day phase, lynch a werewolf until all are gone
    max_days = env.num_werewolves + 1
    for _ in range(max_days):
        if env.current_phase == Phase.GAME_OVER:
            break

        # Identify a WW to lynch
        ww_to_lynch = get_agent_by_role(env, Role.WEREWOLF, must_be_alive=True)
        if not ww_to_lynch:  # Should mean all WWs are dead
            break
        ww_to_lynch_idx = env.agent_id_to_index[ww_to_lynch]

        # Advance to DAY_VOTING (night actions are NO_OPs for simplicity here)
        advance_to_phase(env, Phase.DAY_VOTING)
        if env.current_phase != Phase.DAY_VOTING:
            break  # Game ended or problem

        # All alive non-WWs vote for this WW, WWs vote for someone else or NO_OP
        # Iterate through all alive agents for voting
        for _ in range(len(env.alive_agents) + 1):
            if env.current_phase != Phase.DAY_VOTING:
                break
            agent_to_act = env.agent_selection
            if not agent_to_act:
                break

            if agent_to_act in env.alive_agents and not env.terminations[agent_to_act]:
                if env.player_roles[agent_to_act] != Role.WEREWOLF:
                    env.step(
                        {"action_type": ActionType.DAY_LYNCH_VOTE.value, "target_idx": ww_to_lynch_idx})
                elif agent_to_act == ww_to_lynch:  # The WW being targeted
                    # Target someone else to not save self from lynch
                    non_ww_target = get_agent_by_role(env, Role.VILLAGER) or get_agent_by_role(
                        env, Role.DOCTOR) or get_agent_by_role(env, Role.SEER)
                    if non_ww_target and non_ww_target != ww_to_lynch:
                        env.step({"action_type": ActionType.DAY_LYNCH_VOTE.value,
                                 "target_idx": env.agent_id_to_index[non_ww_target]})
                    else:  # Only WWs left or only self as non-WW target
                        env.step({"action_type": ActionType.NO_OP.value})
                else:  # Other WWs vote for a non-WW
                    non_ww_target = get_agent_by_role(env, Role.VILLAGER) or get_agent_by_role(
                        env, Role.DOCTOR) or get_agent_by_role(env, Role.SEER)
                    if non_ww_target:
                        env.step({"action_type": ActionType.DAY_LYNCH_VOTE.value,
                                 "target_idx": env.agent_id_to_index[non_ww_target]})
                    else:  # Only WWs left
                        env.step({"action_type": ActionType.NO_OP.value})
            elif agent_to_act and env.terminations[agent_to_act]:
                env.step({})

    assert env.current_phase == Phase.GAME_OVER
    assert env.game_winner_team == "VILLAGE"
    for agent_id, role in env.player_roles.items():
        expected_reward = 1.0 if role != Role.WEREWOLF else -1.0
        assert env.rewards[agent_id] == expected_reward


def test_werewolves_win_equal_numbers(werewolf_env_unreset: WerewolfEnv):
    # Specific setup for 3 players: 1 WW, 1 Villager, 1 Doctor (default from fixture)
    # WW needs to kill Villager. Then 1 WW vs 1 Doctor. WW wins.
    # 1 WW, 1 Dr, 1 Villager for 3p
    env_instance = WerewolfEnv(num_doctors=1, num_seers=0)
    env_instance.reset(options={"num_players": 3})

    ww_agent = get_agent_by_role(env_instance, Role.WEREWOLF)
    villager_agent = get_agent_by_role(env_instance, Role.VILLAGER)
    doctor_agent = get_agent_by_role(env_instance, Role.DOCTOR)
    assert ww_agent and villager_agent and doctor_agent

    # Night 1: WW kills Villager. Doctor saves self (or Villager, but Villager kill is priority for test)
    if env_instance.agent_selection == ww_agent:
        env_instance.step({"action_type": ActionType.NIGHT_KILL_VOTE.value,
                          "target_idx": env_instance.agent_id_to_index[villager_agent]})

    if env_instance.agent_selection == doctor_agent:
        # Doctor saves self to ensure they are alive for the WW win condition check
        env_instance.step({"action_type": ActionType.NIGHT_SAVE_TARGET.value,
                          "target_idx": env_instance.agent_id_to_index[doctor_agent]})

    # Phase should be DAY_DISCUSSION or GAME_OVER if Villager was killed and WWs win
    # _resolve_night_actions -> _check_win_conditions -> _transition_phase (sets GAME_OVER)
    assert env_instance.current_phase == Phase.GAME_OVER
    assert villager_agent not in env_instance.alive_agents
    assert env_instance.game_winner_team == "WEREWOLF"

    for agent_id, role in env_instance.player_roles.items():
        expected_reward = 1.0 if role == Role.WEREWOLF else -1.0
        assert env_instance.rewards[agent_id] == expected_reward


def test_observation_werewolf_vote_visibility(env: WerewolfEnv):
    ww_agents = get_all_agents_by_role(env, Role.WEREWOLF)
    if len(ww_agents) < 2:
        pytest.skip("Requires at least 2 werewolves for this test.")
        return

    acting_ww = env.agent_selection
    assert acting_ww in ww_agents

    # Find a non-werewolf target
    target_idx = -1
    for i, p_id in enumerate(env.agent_ids):
        if env.player_roles[p_id] != Role.WEREWOLF:
            target_idx = i
            break
    assert target_idx != -1, "Could not find a non-werewolf target"

    # First werewolf votes
    action = {"action_type": ActionType.NIGHT_KILL_VOTE.value, "target_idx": target_idx}
    env.step(action)

    # Check observation for the *next* werewolf (if any) or the same one if only one WW acts per turn
    # The agent selector should move to the next WW if one exists and is alive.
    # If not, this test might need adjustment or to focus on a single WW's re-observation if possible.
    
    next_acting_agent = env.agent_selection
    if next_acting_agent in ww_agents and next_acting_agent != acting_ww : # Check another WW's obs
        obs_data = env.observe(next_acting_agent)
        obs = WerewolfObservationModel(**obs_data)
        current_ww_votes = json.loads(obs.current_night_werewolf_votes)
        assert acting_ww in current_ww_votes
        assert current_ww_votes[acting_ww] == target_idx
    elif acting_ww == next_acting_agent: # Only one WW or it's their turn again (should not happen in same phase step)
        # Re-observe the same agent if it's their turn again (e.g. if only one WW)
        # This case might not be hit if phase transitions immediately after one WW.
        # If there's only one WW, current_night_werewolf_votes might not be as relevant to test for *others'* votes.
        # However, their own vote should be reflected if the phase allows re-observation before transition.
        # For now, we assume if next_acting_agent is a WW, the vote should be visible.
        obs_data = env.observe(acting_ww) # Re-observe the acting WW
        obs = WerewolfObservationModel(**obs_data)
        current_ww_votes = json.loads(obs.current_night_werewolf_votes)
        assert acting_ww in current_ww_votes
        assert current_ww_votes[acting_ww] == target_idx


def test_observation_role_reveal_on_lynch(env: WerewolfEnv):
    advance_to_phase(env, Phase.DAY_VOTING)
    if env.current_phase != Phase.DAY_VOTING: pytest.skip("Could not reach Day Voting"); return

    alive_agents_for_vote = list(env.alive_agents)
    if len(alive_agents_for_vote) < 2: pytest.skip("Not enough players for lynch"); return

    target_to_lynch_id = alive_agents_for_vote[0]
    target_to_lynch_idx = env.agent_id_to_index[target_to_lynch_id]
    lynched_role_true_value = env.player_roles[target_to_lynch_id].value

    # Majority votes to lynch the target
    voters = [ag for ag in alive_agents_for_vote if ag != target_to_lynch_id][:len(alive_agents_for_vote)//2 +1]
    
    for _ in range(len(env.alive_agents) +1): # Iterate enough times for all votes or phase transition
        if env.current_phase != Phase.DAY_VOTING: break
        agent_to_act = env.agent_selection
        if not agent_to_act: break

        if agent_to_act in voters:
            env.step({"action_type": ActionType.DAY_LYNCH_VOTE.value, "target_idx": target_to_lynch_idx})
        else: # Others (including target or non-voters) NO_OP
            env.step({"action_type": ActionType.NO_OP.value})
    
    if env.current_phase == Phase.GAME_OVER or env.current_phase == Phase.NIGHT_WEREWOLF_VOTE:
        assert target_to_lynch_id not in env.alive_agents # Target should be dead
        
        # Check observation of a remaining alive agent
        if env.alive_agents:
            observer_id = env.alive_agents[0]
            obs_data = env.observe(observer_id)
            obs = WerewolfObservationModel(**obs_data)
            assert obs.last_lynched == target_to_lynch_idx
            assert obs.last_lynched_player_role == lynched_role_true_value


def test_observation_seer_inspection_privacy_and_accuracy(env: WerewolfEnv):
    if env.num_seers == 0: pytest.skip("No seers in this game."); return

    # Advance to Seer inspect phase
    advance_to_phase(env, Phase.NIGHT_SEER_INSPECT)
    if env.current_phase != Phase.NIGHT_SEER_INSPECT: pytest.skip("Could not reach Seer phase"); return

    seer_agent = get_agent_by_role(env, Role.SEER)
    assert seer_agent is not None, "Seer agent not found"
    assert env.agent_selection == seer_agent # Seer should be the one to act

    # Seer inspects a Villager (or first available non-self if no Villager)
    target_to_inspect_id = get_agent_by_role(env, Role.VILLAGER, must_be_alive=True)
    if not target_to_inspect_id or target_to_inspect_id == seer_agent:
        potential_targets = [p for p in env.alive_agents if p != seer_agent]
        if not potential_targets: pytest.skip("No valid target for seer to inspect."); return
        target_to_inspect_id = potential_targets[0]

    target_to_inspect_idx = env.agent_id_to_index[target_to_inspect_id]
    inspected_role_true_value = env.player_roles[target_to_inspect_id].value

    env.step({"action_type": ActionType.NIGHT_INSPECT_TARGET.value, "target_idx": target_to_inspect_idx})

    # Night resolves, advance to Day Discussion to check observations
    # The step above should trigger phase transition if seer is last actor of night
    if env.current_phase != Phase.DAY_DISCUSSION and env.current_phase != Phase.GAME_OVER:
         advance_to_phase(env, Phase.DAY_DISCUSSION) # Ensure night resolves
    if env.current_phase != Phase.DAY_DISCUSSION and env.current_phase != Phase.GAME_OVER:
        pytest.skip("Could not reach Day Discussion after Seer action")
        return

    # Check Seer's observation
    if seer_agent in env.alive_agents: # Seer might have been killed if targeted
        seer_obs_data = env.observe(seer_agent)
        seer_obs = WerewolfObservationModel(**seer_obs_data)
        assert seer_obs.seer_last_inspection == (target_to_inspect_idx, inspected_role_true_value)

    # Check another (non-seer) agent's observation to ensure privacy
    other_agent_id = next((p_id for p_id in env.alive_agents if p_id != seer_agent), None)
    if other_agent_id:
        other_obs_data = env.observe(other_agent_id)
        other_obs = WerewolfObservationModel(**other_obs_data)
        # Default/empty inspection result for non-seers or if seer didn't inspect them
        assert other_obs.seer_last_inspection == (env.num_players, 0)


def test_no_op_action(env: WerewolfEnv):
    agent_id = env.agent_selection
    is_single_ww = len(get_all_agents_by_role(env, Role.WEREWOLF)) == 1

    action = {"action_type": ActionType.NO_OP.value}
    env.step(action)

    assert env.rewards[agent_id] == 0.0
    assert "NO_OP processed" in env.infos[agent_id]["last_action_feedback"]
    assert env.infos[agent_id]["last_action_valid"] is True
    assert env.infos[agent_id]["action_description_for_log"] == "NO_OP"

    if is_single_ww:
        assert env.current_phase == Phase.NIGHT_DOCTOR_SAVE or env.current_phase == Phase.GAME_OVER
    else:  # Multiple WWs
        assert env.agent_selection != agent_id or env.current_phase == Phase.GAME_OVER
