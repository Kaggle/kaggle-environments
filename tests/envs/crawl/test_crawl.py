from kaggle_environments import make
from kaggle_environments.envs.crawl.crawl import _resolve_tiebreak, is_fixed_wall

WALL_N, WALL_E, WALL_S, WALL_W = 1, 2, 4, 8


def test_game_completes():
    """Game runs to completion with two random agents."""
    env = make("crawl", configuration={"episodeSteps": 50, "randomSeed": 42}, debug=True)
    env.run(["random", "random"])
    result = env.toJSON()
    assert result["name"] == "crawl"
    assert result["statuses"] == ["DONE", "DONE"]


def test_initialization():
    """Initial state has correct structure."""
    env = make("crawl", configuration={"randomSeed": 42, "width": 20, "height": 20}, debug=True)
    state = env.reset(2)
    obs0 = state[0].observation
    obs1 = state[1].observation

    assert obs0.southBound == 0
    assert obs0.northBound == 19
    assert len(obs0.walls) == 20 * 20

    # Each player should see exactly one factory (their own) at minimum
    factories_0 = [uid for uid, d in obs0.robots.items() if d[0] == 0 and d[4] == 0]
    factories_1 = [uid for uid, d in obs1.robots.items() if d[0] == 0 and d[4] == 1]
    assert len(factories_0) == 1
    assert len(factories_1) == 1

    # Factories are at symmetric positions
    f0_data = obs0.robots[factories_0[0]]
    f1_data = obs1.robots[factories_1[0]]
    assert f0_data[1] == 20 // 4  # col = width // 4
    assert f1_data[1] == 20 - 1 - 20 // 4  # symmetric mirror
    assert f0_data[2] == f1_data[2] == 2  # row = 2


def test_symmetric_maze():
    """Maze walls are symmetric around the center."""
    env = make("crawl", configuration={"randomSeed": 42, "width": 20}, debug=True)
    state = env.reset(2)
    obs = state[0].observation
    g_walls = obs.globalWalls
    width = 20
    half = width // 2

    for row_key, row_walls in g_walls.items():
        for c in range(half):
            left_w = row_walls[c]
            right_w = row_walls[width - 1 - c]
            # N and S should match
            assert bool(left_w & 1) == bool(right_w & 1), f"N mismatch at row {row_key} col {c}"
            assert bool(left_w & 4) == bool(right_w & 4), f"S mismatch at row {row_key} col {c}"
            # E on left should equal W on right and vice versa
            assert bool(left_w & 2) == bool(right_w & 8), f"E/W mismatch at row {row_key} col {c}"
            assert bool(left_w & 8) == bool(right_w & 2), f"W/E mismatch at row {row_key} col {c}"


def test_factory_energy():
    """Factory starts with configured energy."""
    env = make("crawl", configuration={"randomSeed": 42, "factoryEnergy": 500}, debug=True)
    state = env.reset(2)
    obs = state[0].observation
    for uid, data in obs.robots.items():
        if data[0] == 0 and data[4] == 0:  # Factory, player 0
            assert data[3] == 500


def test_scroll_advances():
    """Southern boundary advances over time."""
    env = make(
        "crawl",
        configuration={"randomSeed": 42, "episodeSteps": 20, "scrollStartInterval": 2, "scrollEndInterval": 1},
        debug=True,
    )
    env.reset(2)
    initial_south = env.state[0].observation.southBound

    # Run a few steps
    def idle_agent(obs, config):
        return {}

    env.run([idle_agent, idle_agent])
    final_south = env.state[0].observation.southBound
    assert final_south > initial_south


def test_renderer_output():
    """Text renderer produces non-empty output."""
    env = make("crawl", configuration={"randomSeed": 42, "episodeSteps": 5}, debug=True)
    env.run(["random", "random"])
    output = env.render(mode="ansi")
    assert isinstance(output, str)
    assert len(output) > 0
    assert "Step:" in output


def test_seed_determinism():
    """Same seed produces same results."""

    def run_game(seed):
        env = make("crawl", configuration={"randomSeed": seed, "episodeSteps": 20}, debug=True)
        env.run(["random", "random"])
        return env.toJSON()

    r1 = run_game(42)
    r2 = run_game(42)
    assert r1["rewards"] == r2["rewards"]
    assert r1["statuses"] == r2["statuses"]


def test_error_agent():
    """Agent that throws exception gets handled."""

    def error_agent(obs, config):
        raise RuntimeError("fail")

    env = make("crawl", configuration={"episodeSteps": 10, "randomSeed": 42})
    env.run([error_agent, "random"])
    result = env.toJSON()
    # Error agent should not be ACTIVE
    assert result["statuses"][0] != "ACTIVE"


def test_fog_of_war():
    """Player cannot see enemy robots outside vision range."""
    env = make("crawl", configuration={"randomSeed": 42, "width": 20}, debug=True)
    state = env.reset(2)
    obs0 = state[0].observation
    obs1 = state[1].observation

    # Player 0's factory is at col 5, player 1's at col 15
    # With vision 4, they shouldn't see each other initially (distance = 10)
    p0_sees_p1 = any(d[4] == 1 for d in obs0.robots.values())
    p1_sees_p0 = any(d[4] == 0 for d in obs1.robots.values())
    assert not p0_sees_p1, "Player 0 should not see player 1's factory"
    assert not p1_sees_p0, "Player 1 should not see player 0's factory"


def test_build_scout():
    """Factory can build a scout."""
    env = make("crawl", configuration={"randomSeed": 42, "episodeSteps": 10}, debug=True)
    state = env.reset(2)

    # Find player 0's factory
    obs = state[0].observation
    factory_uid = None
    for uid, data in obs.robots.items():
        if data[0] == 0 and data[4] == 0:
            factory_uid = uid
            break

    assert factory_uid is not None

    # Issue BUILD_SCOUT action
    def build_agent(obs, config):
        for uid, data in obs.robots.items():
            if data[0] == 0 and data[4] == obs.player:
                return {uid: "BUILD_SCOUT"}
        return {}

    def idle(obs, config):
        return {}

    env.step([{factory_uid: "BUILD_SCOUT"}, {}])
    obs = env.state[0].observation

    # Should now have a scout
    scouts = [uid for uid, d in obs.robots.items() if d[0] == 1 and d[4] == 0]
    assert len(scouts) >= 1, "Should have built at least one scout"


def test_crystal_collection():
    """Robot on a crystal cell collects its energy."""
    env = make(
        "crawl",
        configuration={
            "randomSeed": 42,
            "episodeSteps": 10,
            "crystalDensity": 0.5,
            "crystalMinEnergy": 20,
            "crystalMaxEnergy": 20,
        },
        debug=True,
    )
    state = env.reset(2)
    obs = state[0].observation

    # Check that crystals exist
    g_crystals = obs.globalCrystals
    assert len(g_crystals) > 0, "Should have crystals with high density"


def test_boundary_walls():
    """Maze has proper boundary walls."""
    env = make("crawl", configuration={"randomSeed": 42, "width": 20}, debug=True)
    state = env.reset(2)
    obs = state[0].observation
    g_walls = obs.globalWalls

    width = 20
    # Left edge should have WEST wall
    for row_key, row_walls in g_walls.items():
        assert row_walls[0] & 8, f"Row {row_key} col 0 missing WEST wall"
        assert row_walls[width - 1] & 2, f"Row {row_key} col {width - 1} missing EAST wall"

    # Bottom row should have SOUTH wall
    row_0 = g_walls.get("0", [])
    for c in range(width):
        assert row_0[c] & 4, f"Row 0 col {c} missing SOUTH wall"


def test_is_fixed_wall():
    """Perimeter and middle-axis walls are fixed; everything else is mutable."""
    width = 20
    half = width // 2
    # Perimeter
    assert is_fixed_wall(0, "WEST", width)
    assert is_fixed_wall(width - 1, "EAST", width)
    # Middle axis
    assert is_fixed_wall(half - 1, "EAST", width)
    assert is_fixed_wall(half, "WEST", width)
    # NOT fixed
    assert not is_fixed_wall(0, "NORTH", width)
    assert not is_fixed_wall(0, "SOUTH", width)
    assert not is_fixed_wall(0, "EAST", width)  # only perimeter west of col 0
    assert not is_fixed_wall(5, "EAST", width)
    assert not is_fixed_wall(5, "WEST", width)


def _make_worker(env, col, row, energy=500, owner=0, uid="test-worker"):
    """Inject a worker robot for tests. type=2 (WORKER)."""
    env.state[0].observation.globalRobots[uid] = [2, col, row, energy, owner, 0, 0, 0]
    return uid


def test_worker_remove_wall():
    """Worker REMOVE_DIR clears a wall on its cell and the reciprocal wall on the neighbor."""
    env = make("crawl", configuration={"randomSeed": 42, "episodeSteps": 50}, debug=True)
    env.reset(2)
    obs = env.state[0].observation

    # Find a row + col where a north wall actually exists, away from edges/middle.
    width = env.configuration.width
    target = None
    for row in range(obs.southBound + 1, obs.northBound):
        rw = obs.globalWalls[str(row)]
        for col in range(2, width // 2 - 1):
            if rw[col] & WALL_N:
                target = (col, row)
                break
        if target:
            break
    assert target, "couldn't find a removable north wall in initial maze"
    col, row = target

    uid = _make_worker(env, col, row, energy=500)
    env.step([{uid: "REMOVE_NORTH"}, {}])

    obs = env.state[0].observation
    assert not (obs.globalWalls[str(row)][col] & WALL_N), "north wall not removed on worker cell"
    assert not (obs.globalWalls[str(row + 1)][col] & WALL_S), "south wall not removed on neighbor"
    # Energy charged: 500 - wallActionCost(100) - energyPerTurn(1) = 399
    assert obs.globalRobots[uid][3] == 399


def test_worker_build_wall():
    """Worker BUILD_DIR adds a wall on its cell and the reciprocal wall on the neighbor."""
    env = make("crawl", configuration={"randomSeed": 42, "episodeSteps": 50}, debug=True)
    env.reset(2)
    obs = env.state[0].observation

    # Find a row + col where there's NO north wall.
    width = env.configuration.width
    target = None
    for row in range(obs.southBound + 1, obs.northBound):
        rw = obs.globalWalls[str(row)]
        for col in range(2, width // 2 - 1):
            if not (rw[col] & WALL_N):
                target = (col, row)
                break
        if target:
            break
    assert target, "couldn't find a place with no north wall"
    col, row = target

    uid = _make_worker(env, col, row, energy=500)
    env.step([{uid: "BUILD_NORTH"}, {}])

    obs = env.state[0].observation
    assert obs.globalWalls[str(row)][col] & WALL_N, "north wall not built on worker cell"
    assert obs.globalWalls[str(row + 1)][col] & WALL_S, "south wall not built on neighbor"
    assert obs.globalRobots[uid][3] == 399


def test_worker_cannot_remove_perimeter_wall():
    """REMOVE_WEST at col 0 charges energy but leaves the wall intact (fixed perimeter)."""
    env = make("crawl", configuration={"randomSeed": 42, "episodeSteps": 50}, debug=True)
    env.reset(2)
    obs = env.state[0].observation

    row = obs.southBound + 5
    uid = _make_worker(env, 0, row, energy=500)

    # Pre-condition: perimeter wall exists.
    assert obs.globalWalls[str(row)][0] & WALL_W

    env.step([{uid: "REMOVE_WEST"}, {}])

    obs = env.state[0].observation
    assert obs.globalWalls[str(row)][0] & WALL_W, "perimeter wall was removed"
    # Still charged.
    assert obs.globalRobots[uid][3] == 399


def test_worker_cannot_modify_middle_wall():
    """REMOVE_EAST at col half-1 (middle axis) charges energy but leaves the wall."""
    env = make("crawl", configuration={"randomSeed": 42, "episodeSteps": 50}, debug=True)
    env.reset(2)
    obs = env.state[0].observation

    width = env.configuration.width
    half = width // 2
    row = obs.southBound + 5

    # Force a middle wall to exist for this test (initial maze may have a door here).
    obs.globalWalls[str(row)][half - 1] |= WALL_E
    obs.globalWalls[str(row)][half] |= WALL_W

    uid = _make_worker(env, half - 1, row, energy=500)
    env.step([{uid: "REMOVE_EAST"}, {}])

    obs = env.state[0].observation
    assert obs.globalWalls[str(row)][half - 1] & WALL_E, "middle wall was removed"
    assert obs.globalWalls[str(row)][half] & WALL_W, "middle wall was removed (neighbor side)"


def test_worker_action_charges_even_when_noop():
    """BUILD_DIR where the wall already exists charges energy but doesn't double-build."""
    env = make("crawl", configuration={"randomSeed": 42, "episodeSteps": 50}, debug=True)
    env.reset(2)
    obs = env.state[0].observation

    width = env.configuration.width
    row = obs.southBound + 5
    col = 3
    obs.globalWalls[str(row)][col] |= WALL_N  # ensure north wall exists
    uid = _make_worker(env, col, row, energy=500)
    env.step([{uid: "BUILD_NORTH"}, {}])

    obs = env.state[0].observation
    assert obs.globalRobots[uid][3] == 399, "BUILD no-op should still charge wallActionCost"
    assert obs.globalWalls[str(row)][col] & WALL_N


def test_worker_survives_wall_action():
    """Worker is not destroyed by BUILD/REMOVE; can act repeatedly across turns."""
    env = make("crawl", configuration={"randomSeed": 42, "episodeSteps": 50}, debug=True)
    env.reset(2)
    obs = env.state[0].observation
    row = obs.southBound + 5
    uid = _make_worker(env, 3, row, energy=500)
    env.step([{uid: "BUILD_NORTH"}, {}])
    env.step([{uid: "REMOVE_NORTH"}, {}])
    obs = env.state[0].observation
    assert uid in obs.globalRobots, "worker was destroyed by wall action"
    # 500 - 100 - 1 - 100 - 1 = 298
    assert obs.globalRobots[uid][3] == 298


def test_seed_hidden_from_agents_but_in_replay():
    """Seed drives maze layout (hidden info). Agents must not see it; replay must record it."""
    seen_seeds = []

    def spy_agent(obs, config):
        seen_seeds.append(config.get("randomSeed"))
        return {}

    chosen_seed = 1234567
    env = make("crawl", configuration={"randomSeed": chosen_seed, "episodeSteps": 5}, debug=True)
    env.run([spy_agent, spy_agent])

    assert seen_seeds, "spy agent never received configuration"
    for s in seen_seeds:
        assert s is None, f"agent saw seed={s} in configuration"

    assert env.info.get("seed") == chosen_seed
    replay = env.toJSON()
    assert replay["info"].get("seed") == chosen_seed
    assert replay["configuration"].get("randomSeed") is None


def test_seed_hidden_when_unset_by_user():
    """When user doesn't supply a seed, the generated one is also hidden but recorded."""
    seen_seeds = []

    def spy_agent(obs, config):
        seen_seeds.append(config.get("randomSeed"))
        return {}

    env = make("crawl", configuration={"episodeSteps": 5}, debug=True)
    env.run([spy_agent, spy_agent])

    for s in seen_seeds:
        assert s is None, f"agent saw generated seed={s}"
    assert env.info.get("seed") is not None


def test_resolve_tiebreak_energy():
    """Energy is the primary tiebreaker."""
    robots = {
        "a": {"owner": 0, "energy": 100, "type": 0},
        "b": {"owner": 1, "energy": 50, "type": 0},
    }
    assert _resolve_tiebreak(robots) == (1, 0)


def test_resolve_tiebreak_unit_count():
    """Unit count breaks energy ties."""
    robots = {
        "a": {"owner": 0, "energy": 100, "type": 0},
        "b": {"owner": 0, "energy": 0, "type": 1},
        "c": {"owner": 1, "energy": 100, "type": 0},
    }
    assert _resolve_tiebreak(robots) == (1, 0)


def test_resolve_tiebreak_draw():
    """Equal energy AND units → 0.5 / 0.5 draw."""
    robots = {
        "a": {"owner": 0, "energy": 100, "type": 0},
        "b": {"owner": 1, "energy": 100, "type": 0},
    }
    assert _resolve_tiebreak(robots) == (0.5, 0.5)


def test_factory_factory_mutual_destruction():
    """Two enemy factories on the same cell mutually destroy → tiebreak resolves game."""
    env = make("crawl", configuration={"randomSeed": 42, "episodeSteps": 50}, debug=True)
    env.reset(2)
    obs = env.state[0].observation

    # Find both factories and force them onto the same cell
    f0_uid = f1_uid = None
    for uid, data in obs.globalRobots.items():
        if data[0] == 0 and data[4] == 0:
            f0_uid = uid
        elif data[0] == 0 and data[4] == 1:
            f1_uid = uid
    assert f0_uid and f1_uid

    # Place both factories at the same cell so they mutually destroy.
    target_col, target_row = 10, 5
    obs.globalRobots[f0_uid][1] = target_col
    obs.globalRobots[f0_uid][2] = target_row
    obs.globalRobots[f1_uid][1] = target_col
    obs.globalRobots[f1_uid][2] = target_row

    # Add a surviving scout for p0 so they win the tiebreak (energy/units > 0).
    # Robot list format: [type, col, row, energy, owner, move_cd, jump_cd, build_cd]
    obs.globalRobots["test-scout"] = [1, 0, target_row, 50, 0, 0, 0, 0]

    env.step([{}, {}])
    result = env.toJSON()
    assert result["statuses"] == ["DONE", "DONE"]
    # Player 0 has surviving scout (energy + unit count) → wins tiebreak
    assert result["rewards"] == [1, 0]


def test_factory_crushes_enemy_unit():
    """An enemy non-factory unit on a factory's cell is destroyed; factory survives."""
    env = make("crawl", configuration={"randomSeed": 42, "episodeSteps": 50}, debug=True)
    env.reset(2)
    obs = env.state[0].observation

    f0_uid = next(uid for uid, d in obs.globalRobots.items() if d[0] == 0 and d[4] == 0)
    fc, fr = obs.globalRobots[f0_uid][1], obs.globalRobots[f0_uid][2]

    # Place an enemy scout (owner=1) directly on player 0's factory cell.
    obs.globalRobots["test-enemy-scout"] = [1, fc, fr, 50, 1, 0, 0, 0]

    env.step([{}, {}])

    # Factory survives, enemy scout destroyed.
    assert f0_uid in env.state[0].observation.globalRobots
    assert "test-enemy-scout" not in env.state[0].observation.globalRobots


def test_step_500_tiebreak():
    """If both factories survive to episodeSteps, game ends via tiebreak cascade."""
    env = make("crawl", configuration={"randomSeed": 42, "episodeSteps": 5}, debug=True)
    env.reset(2)

    def idle(obs, config):
        return {}

    env.run([idle, idle])
    result = env.toJSON()
    assert result["statuses"] == ["DONE", "DONE"]
    # Both factories alive (idle agents), so result should be a 1/0/0.5 tiebreak,
    # not raw energy values.
    assert set(result["rewards"]).issubset({0, 0.5, 1})
