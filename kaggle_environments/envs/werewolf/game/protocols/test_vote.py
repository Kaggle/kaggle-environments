from kaggle_environments.envs.werewolf.game.protocols.vote import SequentialVoting


class MockPlayer:
    def __init__(self, pid, alive=True):
        self.id = pid
        self.alive = alive


class MockState:
    def __init__(self, all_ids, alive_ids):
        self.all_player_ids = all_ids
        self.alive_ids = set(alive_ids)
        self.events = []
        self.players = [MockPlayer(pid, pid in self.alive_ids) for pid in all_ids]

    def alive_players(self):
        return [p for p in self.players if p.id in self.alive_ids]

    def get_player_by_id(self, pid):
        for p in self.players:
            if p.id == pid:
                return p
        return None

    def push_event(self, **kwargs):
        self.events.append(kwargs)


def test_sequential_voting_fixed():
    protocol = SequentialVoting(first_to_vote="fixed")
    state = MockState(["p0", "p1", "p2", "p3"], ["p0", "p1", "p2", "p3"])

    # Round 1
    protocol.begin_voting(state, state.alive_players(), state.alive_players())
    # Fixed: starts at p0. Order: p0, p1, p2, p3
    assert protocol._expected_voters == ["p0", "p1", "p2", "p3"]

    # Round 2
    protocol.reset()
    protocol.begin_voting(state, state.alive_players(), state.alive_players())
    assert protocol._expected_voters == ["p0", "p1", "p2", "p3"]


def test_sequential_voting_rotate():
    protocol = SequentialVoting(first_to_vote="rotate")
    state = MockState(["p0", "p1", "p2", "p3"], ["p0", "p1", "p2", "p3"])

    # Day 1: Starts at 0.
    protocol.begin_voting(state, state.alive_players(), state.alive_players())
    # Pivot 0. Queue: p0, p1, p2, p3. Cursor -> 1
    assert protocol._expected_voters == ["p0", "p1", "p2", "p3"]
    assert protocol.pivot_selector._current_cursor == 1

    # Day 2: Starts at 1.
    protocol.reset()
    protocol.begin_voting(state, state.alive_players(), state.alive_players())
    # Pivot 1. Queue: p1, p2, p3, p0. Cursor -> 2
    assert protocol._expected_voters == ["p1", "p2", "p3", "p0"]
    assert protocol.pivot_selector._current_cursor == 2


def test_sequential_voting_rotate_skip_dead():
    protocol = SequentialVoting(first_to_vote="rotate")
    state = MockState(["p0", "p1", "p2", "p3"], ["p0", "p1", "p2", "p3"])

    # Day 1: Starts at 0.
    protocol.begin_voting(state, state.alive_players(), state.alive_players())
    assert protocol._expected_voters == ["p0", "p1", "p2", "p3"]
    assert protocol.pivot_selector._current_cursor == 1

    # Day 2: p1 dies.
    state.alive_ids = set(["p0", "p2", "p3"])
    state.players[1].alive = False

    protocol.reset()
    protocol.begin_voting(state, state.alive_players(), state.alive_players())

    # Cursor is 1. p1 is dead.
    # Should search 1 -> dead.
    # Search 2 -> alive. Pivot 2.
    # Cursor -> 3.
    # Order: p2, p3, p0.
    assert protocol._expected_voters == ["p2", "p3", "p0"]
    assert protocol.pivot_selector._current_cursor == 3

    # Day 3: Starts at 3.
    protocol.reset()
    protocol.begin_voting(state, state.alive_players(), state.alive_players())
    # Pivot 3. Queue: p3, p0, p2. Cursor -> 0
    assert protocol._expected_voters == ["p3", "p0", "p2"]
    assert protocol.pivot_selector._current_cursor == 0


def test_sequential_voting_random():
    protocol = SequentialVoting(first_to_vote="random")
    state = MockState(["p0", "p1", "p2", "p3"], ["p0", "p1", "p2", "p3"])

    starts = []
    for _ in range(100):
        protocol.reset()
        protocol.begin_voting(state, state.alive_players(), state.alive_players())
        if protocol._expected_voters:
            starts.append(protocol._expected_voters[0])

    # Should vary
    assert len(set(starts)) > 1


def test_rule_property():
    p_fixed = SequentialVoting(first_to_vote="fixed")
    assert "voting order always starts from the beginning" in p_fixed.rule

    p_rotate = SequentialVoting(first_to_vote="rotate")
    assert "starting voter rotates" in p_rotate.rule

    p_random = SequentialVoting(first_to_vote="random")
    assert "starting voter is chosen randomly" in p_random.rule
