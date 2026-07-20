from kaggle_environments import make


def stay_agent(obs, config):
    return "STAY"


def bad_agent(obs, config):
    return "FLY"


def make_scripted_agent(moves):
    """Return an agent that plays `moves` in order, then STAY."""
    state = {"i": 0}

    def agent(obs, config):
        i = state["i"]
        state["i"] += 1
        if i < len(moves):
            return moves[i]
        return "STAY"

    return agent


def test_initialization():
    env = make("capture_the_flag", debug=True)
    obs = env.state[0].observation
    rows = env.configuration.rows
    cols = env.configuration.cols
    mid = cols // 2
    assert list(obs.p1_base) == [0, mid]
    assert list(obs.p2_base) == [rows - 1, mid]
    assert list(obs.p1_pos) == [0, mid]
    assert list(obs.p2_pos) == [rows - 1, mid]
    assert list(obs.p1_flag_pos) == [0, mid]
    assert list(obs.p2_flag_pos) == [rows - 1, mid]
    assert obs.p1_has_flag is False
    assert obs.p2_has_flag is False


def test_stalemate_ends_in_draw():
    env = make("capture_the_flag", configuration={"episodeSteps": 6}, debug=True)
    env.run([stay_agent, stay_agent])
    result = env.toJSON()
    assert result["statuses"] == ["DONE", "DONE"]
    assert result["rewards"] == [0, 0]


def test_p1_wins_by_returning_flag():
    # 7x7 default board. P1 base [0,3], P2 base [6,3].
    # P1 walks 6 south to reach P2 base and grab flag (turns 1..6 for P1),
    # then 6 north back home (turns 7..12).
    # P2 stays out of the way by moving east.
    p1_moves = ["SOUTH"] * 6 + ["NORTH"] * 6
    p2_moves = ["EAST"] * 12
    env = make("capture_the_flag", configuration={"episodeSteps": 60}, debug=True)
    env.run([make_scripted_agent(p1_moves), make_scripted_agent(p2_moves)])
    result = env.toJSON()
    assert result["statuses"] == ["DONE", "DONE"]
    assert result["rewards"] == [1, -1]


def test_p2_wins_by_returning_flag():
    p1_moves = ["EAST"] * 12
    p2_moves = ["NORTH"] * 6 + ["SOUTH"] * 6
    env = make("capture_the_flag", configuration={"episodeSteps": 60}, debug=True)
    env.run([make_scripted_agent(p1_moves), make_scripted_agent(p2_moves)])
    result = env.toJSON()
    assert result["statuses"] == ["DONE", "DONE"]
    assert result["rewards"] == [-1, 1]


def test_invalid_action_loses():
    env = make("capture_the_flag", debug=True)
    env.run([bad_agent, stay_agent])
    result = env.toJSON()
    assert result["statuses"][0] != "DONE"
    assert result["rewards"] == [None, 1]


def test_offboard_move_is_noop():
    # P1 at [0, mid] tries NORTH (off-board). Should stay in place.
    env = make("capture_the_flag", configuration={"episodeSteps": 4}, debug=True)
    env.run([make_scripted_agent(["NORTH"]), stay_agent])
    obs = env.state[0].observation
    mid = env.configuration.cols // 2
    assert list(obs.p1_pos) == [0, mid]


def test_tag_returns_flag():
    # Sequence:
    #   Turn 1 (P1):  SOUTH -> [1, mid]  (in P1 territory)
    #   Turn 2 (P2):  NORTH -> [rows-2, mid]
    #   ...
    # Rather than script tag exactly, use the greedy vs defender agents and just
    # confirm the game finishes; then also directly build a scripted scenario
    # where P2 grabs flag, wanders into P1 territory, and P1 tags.
    from kaggle_environments.envs.capture_the_flag.capture_the_flag import interpreter

    env = make("capture_the_flag", debug=True)
    obs = env.state[0].observation
    # Force a hand-crafted state: P2 has grabbed P1's flag and is standing in
    # P1 territory adjacent to P1. P1 moves onto P2's square to tag.
    obs.p1_pos = [1, 3]
    obs.p2_pos = [2, 3]
    obs.p2_has_flag = True
    obs.p1_flag_pos = [2, 3]
    env.state[0].status = "ACTIVE"
    env.state[1].status = "INACTIVE"
    env.state[0].action = "SOUTH"
    env.state[1].action = "STAY"

    interpreter(env.state, env)

    rows = env.configuration.rows
    cols = env.configuration.cols
    mid = cols // 2
    assert list(env.state[0].observation.p2_pos) == [rows - 1, mid]
    assert env.state[0].observation.p2_has_flag is False
    assert list(env.state[0].observation.p1_flag_pos) == [0, mid]


def test_renderer_returns_string():
    env = make("capture_the_flag", configuration={"episodeSteps": 3}, debug=True)
    env.run([stay_agent, stay_agent])
    out = env.render(mode="ansi")
    assert isinstance(out, str)
    assert len(out) > 0
    assert "Step" in out


def test_built_in_agents_finish():
    env = make("capture_the_flag", configuration={"episodeSteps": 100}, debug=True)
    env.run(["greedy", "defender"])
    result = env.toJSON()
    assert result["statuses"] == ["DONE", "DONE"]
    for r in result["rewards"]:
        assert r in (-1, 0, 1)
