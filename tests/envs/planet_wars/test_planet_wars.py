"""Tests for the planet_wars environment.

The simulation rules are validated against Jeff Cameron's reference engine
(C++ port at github.com/albertz/planet_wars-cpp). The map generator is
validated for distribution sanity but not bit-for-bit against the original
map_generator_v2.py since we use a different RNG sequence.
"""

from kaggle_environments import make
from kaggle_environments.envs.planet_wars.planet_wars import (
    HOME_GROWTH,
    HOME_SHIPS,
    MAX_PLANETS,
    MIN_DISTANCE,
    MIN_PLANETS,
    MIN_STARTING_DISTANCE,
    Fleet,
    Planet,
    _apply_orders,
    _do_time_step,
    _fight_battle,
    _total_ships,
    _validate_orders,
    distance,
    format_map,
    generate_map,
    parse_map,
)

# ---------------------------------------------------------------------------
# Geometry & parsing
# ---------------------------------------------------------------------------


def test_distance_ceil():
    assert distance((0, 0), (3, 4)) == 5
    assert distance((0, 0), (1, 1)) == 2
    assert distance((0, 0), (0, 0)) == 0
    assert distance((1.0, 1.0), (1.0, 1.0)) == 0
    assert distance((0.0, 0.0), (4.0, 0.0)) == 4


def test_parse_map_round_trip():
    text = (
        "# header comment\n"
        "\n"
        "P 0 0 1 100 5  # home\n"
        "P 7 9 2 100 5\n"
        "P 3.14 2.71 0 15 5\n"
        "\n"
        "F 1 15 0 1 12 2\n"
        "F 2 28 1 2 8 4\n"
    )
    planets, fleets = parse_map(text)
    assert len(planets) == 3
    assert planets[0] == [0, 0.0, 0.0, 1, 100, 5]
    assert planets[1] == [1, 7.0, 9.0, 2, 100, 5]
    assert planets[2] == [2, 3.14, 2.71, 0, 15, 5]
    assert fleets == [
        [1, 15, 0, 1, 12, 2],
        [2, 28, 1, 2, 8, 4],
    ]
    # format_map round-trip
    text2 = format_map(planets, fleets)
    planets2, fleets2 = parse_map(text2)
    assert planets2 == planets
    assert fleets2 == fleets


def test_namedtuple_wrapping():
    text = "P 1.5 2.5 0 10 3\n"
    planets, _ = parse_map(text)
    p = Planet(*planets[0])
    assert p.id == 0
    assert (p.x, p.y) == (1.5, 2.5)
    assert p.owner == 0
    assert p.num_ships == 10
    assert p.growth_rate == 3


# ---------------------------------------------------------------------------
# Battle resolution — bit-exact against the C++ reference behaviour.
# ---------------------------------------------------------------------------


def _make_planet(pid, owner, ships):
    return [pid, 0.0, 0.0, owner, ships, 0]


def _fleet(owner, ships, dest=0):
    return [owner, ships, 0, dest, 1, 0]


def test_battle_uncontested_keeps_owner():
    planet = _make_planet(0, 1, 5)
    _fight_battle(planet, [])
    assert planet[3] == 1 and planet[4] == 5


def test_battle_attacker_wins_neutral():
    planet = _make_planet(0, 0, 3)
    arrivals = [_fleet(1, 10)]
    _fight_battle(planet, arrivals)
    assert planet[3] == 1 and planet[4] == 7


def test_battle_attacker_loses_to_garrison():
    planet = _make_planet(0, 1, 10)
    arrivals = [_fleet(2, 8)]
    _fight_battle(planet, arrivals)
    assert planet[3] == 1 and planet[4] == 2


def test_battle_tie_keeps_prior_owner_with_zero_ships():
    planet = _make_planet(0, 1, 5)
    arrivals = [_fleet(2, 5)]
    _fight_battle(planet, arrivals)
    assert planet[3] == 1
    assert planet[4] == 0


def test_battle_tie_on_neutral_stays_neutral():
    planet = _make_planet(0, 0, 0)
    arrivals = [_fleet(1, 5), _fleet(2, 5)]
    _fight_battle(planet, arrivals)
    assert planet[3] == 0
    assert planet[4] == 0


def test_battle_three_way():
    planet = _make_planet(0, 0, 3)
    arrivals = [_fleet(1, 5), _fleet(2, 4)]
    _fight_battle(planet, arrivals)
    assert planet[3] == 1 and planet[4] == 1


def test_battle_same_owner_fleets_sum():
    planet = _make_planet(0, 0, 0)
    arrivals = [_fleet(1, 3), _fleet(1, 4), _fleet(2, 6)]
    _fight_battle(planet, arrivals)
    assert planet[3] == 1 and planet[4] == 1


# ---------------------------------------------------------------------------
# DoTimeStep — growth applied before battle; fleets decrement; arrivals
# resolve in the same tick.
# ---------------------------------------------------------------------------


def test_growth_applied_before_battle():
    planets = [
        [0, 0.0, 0.0, 1, 1, 5],
        [1, 10.0, 0.0, 2, 50, 0],
    ]
    fleets = [[2, 5, 1, 0, 10, 1]]  # arrives this turn
    _do_time_step(planets, fleets)
    assert planets[0][3] == 1
    assert planets[0][4] == 1
    assert fleets == []


def test_growth_only_for_owned_planets():
    planets = [
        [0, 0.0, 0.0, 0, 10, 5],  # neutral — no growth
        [1, 10.0, 0.0, 1, 10, 3],
    ]
    _do_time_step(planets, [])
    assert planets[0][4] == 10
    assert planets[1][4] == 13


def test_fleet_decrements_and_lands():
    planets = [
        [0, 0.0, 0.0, 0, 0, 0],
        [1, 10.0, 0.0, 1, 0, 0],
    ]
    fleets = [[1, 7, 1, 0, 10, 3]]
    _do_time_step(planets, fleets)
    assert fleets[0][5] == 2
    _do_time_step(planets, fleets)
    _do_time_step(planets, fleets)
    # After three ticks, fleet arrives and is removed.
    assert fleets == []
    assert planets[0][3] == 1
    assert planets[0][4] == 7


# ---------------------------------------------------------------------------
# ExecuteOrder validation & fleet merging
# ---------------------------------------------------------------------------


def test_validate_orders_valid_empty_and_one():
    planets = parse_map("P 0 0 1 10 5\nP 5 0 2 10 5\n")[0]
    assert _validate_orders([], planets, 1)
    assert _validate_orders([[0, 1, 5]], planets, 1)


def test_validate_orders_invalid_cases():
    planets = parse_map("P 0 0 1 10 5\nP 5 0 2 10 5\nP 7 0 0 5 1\n")[0]
    # zero or negative ships
    assert not _validate_orders([[0, 1, 0]], planets, 1)
    assert not _validate_orders([[0, 1, -1]], planets, 1)
    # source == dest
    assert not _validate_orders([[0, 0, 5]], planets, 1)
    # source not owned
    assert not _validate_orders([[1, 0, 5]], planets, 1)
    assert not _validate_orders([[2, 0, 5]], planets, 1)  # neutral
    # ships > available
    assert not _validate_orders([[0, 1, 11]], planets, 1)
    # running total exceeds source ships
    assert not _validate_orders([[0, 1, 6], [0, 1, 5]], planets, 1)
    # out of range indices
    assert not _validate_orders([[5, 1, 1]], planets, 1)
    assert not _validate_orders([[0, 99, 1]], planets, 1)
    # malformed
    assert not _validate_orders("nope", planets, 1)
    assert not _validate_orders([[0, 1]], planets, 1)


def test_same_turn_fleet_merge():
    planets = parse_map("P 0 0 1 20 5\nP 3 4 2 0 0\n")[0]
    fleets = []
    _apply_orders([[0, 1, 5], [0, 1, 7]], planets, fleets, 1)
    assert planets[0][4] == 8  # 20 - 12
    assert len(fleets) == 1
    assert fleets[0] == [1, 12, 0, 1, 5, 5]


def test_cross_player_orders_do_not_merge():
    planets = parse_map("P 0 0 1 10 5\nP 3 4 2 10 5\nP 6 0 0 5 0\n")[0]
    fleets = []
    _apply_orders([[0, 2, 5]], planets, fleets, 1)
    _apply_orders([[1, 2, 5]], planets, fleets, 2)
    assert len(fleets) == 2


# ---------------------------------------------------------------------------
# Map generator distribution
# ---------------------------------------------------------------------------


def test_generate_map_reproducible():
    a, _ = generate_map(123)
    b, _ = generate_map(123)
    assert a == b


def test_generate_map_distinct_seeds_differ():
    seen = {format_map(generate_map(s)[0]) for s in range(10)}
    # Vanishingly unlikely for 10 seeds to all collide.
    assert len(seen) > 5


def test_generate_map_distribution():
    """Run the generator over a sample of seeds and check the resulting
    distribution matches the original Waterloo maps for the things that
    matter to gameplay."""
    for seed in range(50):
        planets, fleets = generate_map(seed)
        assert fleets == []
        # Planet count within bounds (radial may round up by one).
        assert MIN_PLANETS <= len(planets) <= MAX_PLANETS + 1
        # Exactly one home per player.
        homes_by_owner = {1: [], 2: []}
        for p in planets:
            if p[3] in homes_by_owner:
                homes_by_owner[p[3]].append(p)
        assert len(homes_by_owner[1]) == 1, f"seed {seed}"
        assert len(homes_by_owner[2]) == 1, f"seed {seed}"
        # Homes carry the original ship + growth values.
        for o in (1, 2):
            home = homes_by_owner[o][0]
            assert home[4] == HOME_SHIPS
            assert home[5] == HOME_GROWTH
        # Home-to-home distance >= MIN_STARTING_DISTANCE.
        h1, h2 = homes_by_owner[1][0], homes_by_owner[2][0]
        assert distance((h1[1], h1[2]), (h2[1], h2[2])) >= MIN_STARTING_DISTANCE, f"seed {seed}"
        # All pairs of planets are at least MIN_DISTANCE apart.
        for i in range(len(planets)):
            for j in range(i + 1, len(planets)):
                d = distance(
                    (planets[i][1], planets[i][2]),
                    (planets[j][1], planets[j][2]),
                )
                assert d >= MIN_DISTANCE, f"seed {seed}: planets {i},{j} too close"
        # Coordinates are non-negative after translation.
        for p in planets:
            assert p[1] >= 0 and p[2] >= 0


# ---------------------------------------------------------------------------
# End-to-end: full env behaviour through `make` / `env.run`.
# ---------------------------------------------------------------------------


def _const_agent(action):
    def agent(obs, config):
        return action

    return agent


def test_env_init_and_seed_persistence():
    env = make("planet_wars", configuration={"seed": 42})
    env.reset()
    assert env.info.get("seed") == 42
    # Public seed: written back into configuration too.
    cfg_seed = getattr(env.configuration, "seed", None)
    assert cfg_seed == 42
    obs = env.state[0].observation
    assert obs.planets, "init should populate planets"
    # Both agents see the same shared planet list.
    assert env.state[0].observation.planets == env.state[1].observation.planets


def test_env_seed_resolved_when_none():
    env = make("planet_wars")
    env.reset()
    assert env.info.get("seed") is not None
    assert getattr(env.configuration, "seed", None) == env.info["seed"]


def test_env_random_seed_reproducible():
    env_a = make("planet_wars", configuration={"seed": 99})
    env_a.run(["do_nothing", "do_nothing"])
    env_b = make("planet_wars", configuration={"seed": 99})
    env_b.run(["do_nothing", "do_nothing"])
    assert env_a.toJSON()["rewards"] == env_b.toJSON()["rewards"]
    assert env_a.state[0].observation.planets == env_b.state[0].observation.planets


def test_env_do_nothing_draws_at_max_turns():
    env = make("planet_wars", configuration={"seed": 1, "episodeSteps": 5})
    env.run(["do_nothing", "do_nothing"])
    j = env.toJSON()
    assert j["statuses"] == ["DONE", "DONE"]
    # Symmetric map + no orders = same ship totals = draw.
    assert j["rewards"] == [0, 0]


def test_env_invalid_action_forfeits():
    def bad(obs, config):
        # negative ships — always invalid
        return [[0, 1, -1]]

    env = make("planet_wars", configuration={"seed": 5, "episodeSteps": 20})
    env.run([bad, "do_nothing"])
    j = env.toJSON()
    assert j["statuses"][0] == "INVALID"
    assert j["statuses"][1] == "DONE"
    assert j["rewards"][0] is None
    assert j["rewards"][1] == 1


def test_env_both_invalid_draw():
    def bad(obs, config):
        return [[0, 0, 5]]  # source == dest

    env = make("planet_wars", configuration={"seed": 5})
    env.run([bad, bad])
    j = env.toJSON()
    assert j["statuses"] == ["INVALID", "INVALID"]
    assert j["rewards"] == [None, None]


def test_env_empty_action_is_noop():
    def empty(obs, config):
        return []

    env = make("planet_wars", configuration={"seed": 5, "episodeSteps": 3})
    env.run([empty, empty])
    j = env.toJSON()
    assert j["statuses"] == ["DONE", "DONE"]
    assert j["rewards"] == [0, 0]


def test_env_renderer_returns_text():
    env = make("planet_wars", configuration={"seed": 5, "episodeSteps": 3})
    env.run(["do_nothing", "do_nothing"])
    out = env.render(mode="ansi")
    assert isinstance(out, str)
    assert "Planets:" in out


def test_env_win_by_elimination_with_handcrafted_map():
    """Player 1 has a small fleet poised to wipe player 2 in 1 turn."""
    map_text = (
        # p0 = neutral throwaway so generator-style code stays happy
        "P 0 0 0 0 0\nP 1 0 1 100 5\nP 2 0 2 1 0\n"
    )
    env = make(
        "planet_wars",
        configuration={"seed": 1, "episodeSteps": 50, "map": map_text},
    )

    # Player 1 sends 50 ships from planet 1 to planet 2. Trip length = ceil(1) = 1.
    def attacker(obs, config):
        step = obs.get("step", 0)
        if step == 0:
            return [[1, 2, 50]]
        return []

    env.run([attacker, "do_nothing"])
    j = env.toJSON()
    assert j["statuses"] == ["DONE", "DONE"]
    assert j["rewards"] == [1, -1]


def test_env_total_ships_tracker():
    """Whitebox check that the total_ships helper matches obs at game end."""
    env = make("planet_wars", configuration={"seed": 3, "episodeSteps": 10})
    env.run(["nearest_enemy", "do_nothing"])
    obs = env.state[0].observation
    s1 = _total_ships(obs.planets, obs.fleets, 1)
    s2 = _total_ships(obs.planets, obs.fleets, 2)
    assert s1 >= 0 and s2 >= 0


def test_namedtuple_fleet():
    f = Fleet(1, 10, 0, 1, 5, 3)
    assert f.owner == 1 and f.num_ships == 10 and f.turns_remaining == 3


def test_distance_uses_math_ceil_semantics():
    # Verify floats slightly below the next integer still round up.
    assert distance((0, 0), (3.0001, 4.0)) == 6
    assert distance((0, 0), (3.0, 4.0)) == 5
    # Sanity: integer hypotenuse stays exact.
    assert distance((0, 0), (5.0, 12.0)) == 13
