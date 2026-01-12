from kaggle_environments.envs.werewolf.game.protocols.chat import RoundRobinDiscussion
from kaggle_environments.envs.werewolf.game.protocols.ordering import FirstPlayerStrategy


class MockPlayer:
    def __init__(self, pid):
        self.id = pid


class MockState:
    def __init__(self, all_ids, alive_ids):
        self.all_player_ids = all_ids
        self.alive_ids = set(alive_ids)
        self.events = []

    def alive_players(self):
        return [MockPlayer(pid) for pid in self.all_player_ids if pid in self.alive_ids]

    def is_alive(self, pid):
        return pid in self.alive_ids

    def push_event(self, **kwargs):
        self.events.append(kwargs)


def test_fixed():
    protocol = RoundRobinDiscussion(max_rounds=2, first_to_speak="fixed")
    state = MockState(["p0", "p1", "p2", "p3"], ["p0", "p1", "p2", "p3"])

    protocol.begin(state)

    # Fixed: starts at p0 (index 0)
    # R1: p0, p1, p2, p3
    # R2: p0, p1, p2, p3
    expected = ["p0", "p1", "p2", "p3", "p0", "p1", "p2", "p3"]
    assert list(protocol._queue) == expected


def test_rotate():
    protocol = RoundRobinDiscussion(max_rounds=2, first_to_speak="rotate")
    state = MockState(["p0", "p1", "p2", "p3"], ["p0", "p1", "p2", "p3"])

    protocol.begin(state)

    # Rotate:
    # R1: pivot 0 -> p0, p1, p2, p3.
    # R2: pivot 0 -> p0, p1, p2, p3.
    # Cursor updates once for the day: 0 -> 1
    expected = ["p0", "p1", "p2", "p3", "p0", "p1", "p2", "p3"]
    assert list(protocol._queue) == expected
    assert protocol.pivot_selector._current_cursor == 1

    # Next day (call begin again)
    # R3: pivot 1 -> p1, p2, p3, p0.
    # R4: pivot 1 -> p1, p2, p3, p0.
    # Cursor updates once for the day: 1 -> 2
    protocol.reset()
    protocol.begin(state)
    expected_day2 = ["p1", "p2", "p3", "p0", "p1", "p2", "p3", "p0"]
    assert list(protocol._queue) == expected_day2
    assert protocol.pivot_selector._current_cursor == 2


def test_random():
    protocol = RoundRobinDiscussion(max_rounds=5, first_to_speak="random")
    state = MockState(["p0", "p1", "p2", "p3"], ["p0", "p1", "p2", "p3"])

    daily_starts = []

    for _ in range(10):  # Run 10 days
        protocol.reset()
        protocol.begin(state)
        queue = list(protocol._queue)

        # Check that within a day, all rounds start with the same player
        starts_within_day = queue[0::4]  # 4 players
        assert len(set(starts_within_day)) == 1, f"Random pivot changed within a day: {starts_within_day}"

        daily_starts.append(starts_within_day[0])

    # Check that across days, the start varies
    # With 4 players and 10 days, highly unlikely to be all same
    assert len(set(daily_starts)) > 1, f"Random pivot did not vary across days: {daily_starts}"


def test_rule_property():
    p_fixed = RoundRobinDiscussion(max_rounds=2, first_to_speak="fixed")
    assert "speaking order always starts from the beginning" in p_fixed.rule

    p_rotate = RoundRobinDiscussion(max_rounds=2, first_to_speak="rotate")
    assert "starting speaker rotates to the next player in the list for each subsequent day" in p_rotate.rule

    p_random = RoundRobinDiscussion(max_rounds=2, first_to_speak="random")
    assert "starting speaker is chosen randomly" in p_random.rule


def test_default_is_rotate():
    # Initialize without specifying first_to_speak
    protocol = RoundRobinDiscussion(max_rounds=2)

    # Check that the default is indeed ROTATE
    assert protocol.pivot_selector.strategy == FirstPlayerStrategy.ROTATE

    # Verify behavior
    state = MockState(["p0", "p1", "p2", "p3"], ["p0", "p1", "p2", "p3"])
    protocol.begin(state)

    # Expected for Rotate:
    # R1 (pivot 0): p0, p1, p2, p3
    # R2 (pivot 0): p0, p1, p2, p3
    expected = ["p0", "p1", "p2", "p3", "p0", "p1", "p2", "p3"]
    assert list(protocol._queue) == expected


def test_round_robin_rotate_skip_dead():
    # Setup
    protocol = RoundRobinDiscussion(max_rounds=1, first_to_speak="rotate")
    # p0, p1, p2, p3 are players.
    state = MockState(["p0", "p1", "p2", "p3"], ["p0", "p1", "p2", "p3"])

    # --- Day 1 ---
    # Cursor starts at 0.
    protocol.begin(state)

    # Expected: Pivot at p0. Queue: p0, p1, p2, p3.
    # Cursor should update to 1 (next after p0).
    assert list(protocol._queue) == ["p0", "p1", "p2", "p3"]
    assert protocol.pivot_selector._current_cursor == 1

    # --- Day 2 ---
    # Now assume p1 dies.
    state.alive_ids = set(["p0", "p2", "p3"])

    # Reset queue (usually handled by caller or begin, but protocol.reset() clears it too)
    protocol.reset()
    protocol.begin(state)

    # Logic:
    # Cursor is 1. p1 is dead.
    # Should search 1 -> dead.
    # Search 2 -> alive. Pivot becomes 2 (p2).
    # Cursor becomes 3 (next after p2).
    # Queue should be p2, p3, p0.

    assert list(protocol._queue) == ["p2", "p3", "p0"]
    assert protocol.pivot_selector._current_cursor == 3

    # --- Day 3 ---
    # No more deaths.
    protocol.reset()
    protocol.begin(state)

    # Logic:
    # Cursor is 3. p3 is alive.
    # Pivot becomes 3 (p3).
    # Cursor becomes 0 (next after p3).
    # Queue: p3, p0, p2.

    assert list(protocol._queue) == ["p3", "p0", "p2"]
    assert protocol.pivot_selector._current_cursor == 0


def test_round_robin_more_edge_cases():
    # Setup
    protocol = RoundRobinDiscussion(max_rounds=1, first_to_speak="rotate")
    # p0, p1, p2, p3 are players.
    state = MockState(["p0", "p1", "p2", "p3"], ["p0", "p1", "p2", "p3"])

    # Case 1: User scenario
    # Start a day with p2.
    # We can force this by setting cursor to 2.
    protocol.pivot_selector._current_cursor = 2
    protocol.begin(state)

    # Check Pivot is p2
    assert list(protocol._queue) == ["p2", "p3", "p0", "p1"]
    # Check cursor updated to 3
    assert protocol.pivot_selector._current_cursor == 3

    # p1 dies.
    state.alive_ids = set(["p0", "p2", "p3"])
    protocol.reset()
    protocol.begin(state)

    # Next day should rotate to p3.
    # p1 being dead shouldn't affect p3 being next.
    assert list(protocol._queue) == ["p3", "p0", "p2"]
    # Cursor should update to (3+1)%4 = 0
    assert protocol.pivot_selector._current_cursor == 0

    # Case 2: Wrap around with death
    # Current cursor is 0. p0 is alive.
    # Let's kill p0.
    state.alive_ids = set(["p2", "p3"])  # p1 already dead, now p0 dead.
    protocol.reset()
    protocol.begin(state)

    # Cursor 0 -> p0 dead.
    # Next 1 -> p1 dead.
    # Next 2 -> p2 alive.
    # Pivot should be p2.
    assert list(protocol._queue) == ["p2", "p3"]
    # Cursor should be 3
    assert protocol.pivot_selector._current_cursor == 3
