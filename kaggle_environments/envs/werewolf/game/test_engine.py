import pytest

from .engine import Moderator, DetailedPhase
from .states import GameState
from .roles import Player, Werewolf, Villager, Seer, Doctor, Team, Phase, RoleConst
from .actions import EliminateProposalAction, VoteAction, HealAction, InspectAction, ChatAction
from .protocols import RoundRobinDiscussion, SimultaneousMajority, MajorityEliminateResolver, WerewolfEliminationProtocol


@pytest.fixture
def initial_players_simple():
    return [
        Player(id="0", role=Werewolf(descriptions="A werewolf.")),
        Player(id="1", role=Seer(descriptions="A seer.")),
        Player(id="2", role=Doctor(descriptions="A doctor.")),
        Player(id="3", role=Villager(descriptions="An ordinary villager.", name=RoleConst.VILLAGER, team=Team.VILLAGERS)),
        Player(id="4", role=Villager(descriptions="Another ordinary villager.", name=RoleConst.VILLAGER, team=Team.VILLAGERS)),
    ]


@pytest.fixture
def game_state_simple(initial_players_simple):
    return GameState(players=initial_players_simple, history={})


@pytest.fixture
def moderator_simple(game_state_simple):
    discussion_protocol = RoundRobinDiscussion(max_rounds=1)
    day_voting_protocol = SimultaneousMajority()
    night_voting_protocol = SimultaneousMajority()

    return Moderator(state=game_state_simple, discussion=discussion_protocol, 
                     day_voting=day_voting_protocol, night_voting=night_voting_protocol)


def test_moderator_creation(moderator_simple):
    assert moderator_simple is not None
    assert moderator_simple.state.phase == Phase.NIGHT
    assert moderator_simple.detailed_phase == DetailedPhase.NIGHT_START


def test_full_night_cycle_attack_save_inspect(moderator_simple: Moderator):
    """
    Tests a full night cycle:
    - Moderator starts in NIGHT_START. advance({}) transitions to NIGHT_AWAIT_ACTIONS.
    - Doctor (P2) tries to heal P3.
    - Werewolf (P0) votes to eliminate P3.
    - Seer (P1) inspects P0.
    Expected: P3 is attacked but saved. P0 (Werewolf) is inspected. Transition to Day.
    """
    state = moderator_simple.state
    assert moderator_simple.detailed_phase == DetailedPhase.NIGHT_START

    # NIGHT_START -> NIGHT_AWAIT_ACTIONS, action queue populated
    moderator_simple.advance({}) 
    assert moderator_simple.detailed_phase == DetailedPhase.NIGHT_AWAIT_ACTIONS
    active_ids_night_actions = moderator_simple.get_active_player_ids()
    assert "0" in active_ids_night_actions # Werewolf P0
    assert "1" in active_ids_night_actions # Seer P1
    assert "2" in active_ids_night_actions # Doctor P2

    night_actions_dict = {
        "2": HealAction(actor_id="2", target_id="3"),      # Doctor P2 heals P3
        "0": VoteAction(actor_id="0", target_id="3"),      # Werewolf P0 targets P3
        "1": InspectAction(actor_id="1", target_id="0")    # Seer P1 inspects P0
    }
    # NIGHT_AWAIT_ACTIONS -> DAY_START (night actions resolved)
    moderator_simple.advance(night_actions_dict) 

    assert moderator_simple.detailed_phase == DetailedPhase.DAY_START
    assert state.phase == Phase.NIGHT # State phase changes at start of day handler
    assert state.day_count == 0 # Day count increments at start of day handler

    assert state.players[3].alive, "Player '3' should have been saved by the Doctor"
    
    night_0_history = state.history.get(0, [])
    assert any(f"Last night, P3 ({state.players[3].role.name}) was attacked by werewolves but saved by the Doctor!" in entry.description for entry in night_0_history), "Save message not found"
    
    seer_inspection_found = False
    for entry in night_0_history:
        if entry.entry_type == "action_result" and "1" in entry.visible_to and not entry.public:
            if f"You inspected P0. They are a {RoleConst.WEREWOLF} ({Team.WEREWOLVES.value})" in entry.description:
                seer_inspection_found = True
                break
    assert seer_inspection_found, "Seer inspection result not found or not private to Seer"

    # DAY_START -> DAY_DISCUSSION_AWAIT_CHAT
    moderator_simple.advance({}) 

    # Moderator should have processed night actions and set up for day
    assert moderator_simple.detailed_phase == DetailedPhase.DAY_DISCUSSION_AWAIT_CHAT
    assert state.phase == Phase.DAY
    assert state.day_count == 1
    
    assert state.players[0].alive # Player "0" (Werewolf) is alive
    assert state.players[1].alive # Player "1" (Seer) is alive
    assert state.players[2].alive # Player "2" (Doctor) is alive
    day_1_history = state.history.get(1, []) # Day 1 announcements
    assert any("Day 1 begins." in entry.description for entry in day_1_history), "Day 1 announcement missing"


def test_full_day_cycle_discussion_vote_exile(moderator_simple: Moderator):
    """
    Tests a full day cycle after a night where no one died (e.g., P3 was saved).
    - Discussion happens.
    - Voting: P0 (WW), P1 (Seer), P2 (Doctor), P3 (Villager), P4 (Villager) are alive.
    - P1, P2, P3, P4 vote to exile P0 (WW). P0 votes P1.
    Expected: P0 is exiled. Transition to Night. Game over (Villagers win).
    """
    state = moderator_simple.state

    # --- Setup: Advance to Day 1 Discussion ---
    # NIGHT_START -> NIGHT_AWAIT_ACTIONS
    moderator_simple.advance({}) 
    # Actions: WW P0 votes P4, Seer P1 inspects P2, Doctor P2 heals P4 (P4 saved)
    night_actions_setup = {
        "0": VoteAction(actor_id="0", target_id="4"),
        "1": InspectAction(actor_id="1", target_id="2"),
        "2": HealAction(actor_id="2", target_id="4")
    }
    # NIGHT_AWAIT_ACTIONS -> DAY_START
    moderator_simple.advance(night_actions_setup) 
    assert state.players[4].alive # P4 saved
    # DAY_START -> DAY_DISCUSSION_AWAIT_CHAT
    moderator_simple.advance({}) 
    assert moderator_simple.detailed_phase == DetailedPhase.DAY_DISCUSSION_AWAIT_CHAT
    assert state.phase == Phase.DAY
    assert state.day_count == 1
    # --- End Setup ---

    # --- Simulate Discussion Phase ---
    # RoundRobinDiscussion with max_rounds=1, so each of 5 alive players speaks once.
    num_alive_players = len(state.alive_players())
    for i in range(num_alive_players):
        assert moderator_simple.detailed_phase == DetailedPhase.DAY_DISCUSSION_AWAIT_CHAT
        active_speakers = moderator_simple.get_active_player_ids()
        assert len(active_speakers) == 1, f"Expected 1 speaker, got {len(active_speakers)} on turn {i}"
        speaker_id = active_speakers[0]
        chat_action = ChatAction(actor_id=speaker_id, message=f"Player {speaker_id} says something.")
        moderator_simple.advance({speaker_id: chat_action})
        day_1_history = state.history.get(1, [])
        assert any(chat_action.message in entry.description and f"P{speaker_id}" in entry.description for entry in day_1_history), f"Chat from P{speaker_id} not found"

    # After all discussion, should transition to voting
    assert moderator_simple.detailed_phase == DetailedPhase.DAY_VOTING_AWAIT
    active_voters = moderator_simple.get_active_player_ids()
    assert len(active_voters) == num_alive_players # SimultaneousMajority, all vote

    # --- Simulate Voting Phase ---
    # P0(WW), P1(Seer), P2(Doctor), P3(Villager), P4(Villager) are alive.
    # Villagers (P1,P2,P3,P4) vote P0. P0 votes P1.
    vote_actions_dict = {
        "1": VoteAction(actor_id="1", target_id="0"), # P1 (Seer) votes P0
        "2": VoteAction(actor_id="2", target_id="0"), # P2 (Doctor) votes P0
        "3": VoteAction(actor_id="3", target_id="0"), # P3 (Villager) votes P0
        "4": VoteAction(actor_id="4", target_id="0"), # P4 (Villager) votes P0
        "0": VoteAction(actor_id="0", target_id="1")  # P0 (WW) votes P1
    }
    moderator_simple.advance(vote_actions_dict)

    # After votes are processed (P0 is player "0")
    assert not state.players[0].alive, "P0 (Werewolf) should be exiled"
    day_1_history_after_vote = state.history.get(1, []) # Vote resolution happens on Day 1
    assert any(f"P0 ({RoleConst.WEREWOLF.value}) was exiled by vote. They were a {RoleConst.WEREWOLF.value}" in entry.description for entry in day_1_history_after_vote), "Exile message not found"

    # Game should be over, Villagers win
    assert moderator_simple.is_game_over()
    assert moderator_simple.detailed_phase == DetailedPhase.GAME_OVER
    # Game over message is also part of day 1 history as it's resolved after day's events
    assert any("Game Over: Villagers Win!" in entry.description for entry in day_1_history_after_vote), "Villagers win message not found"


def test_game_over_werewolves_win_by_elimination(moderator_simple: Moderator):
    """
    Tests Werewolves winning by eliminating Villagers until parity or majority.
    Scenario: P0 (WW), P3 (Villager), P4 (Villager) are alive. (P1 Seer, P2 Doctor are dead).
    Night: P0 (WW) eliminates P3.
    Expected: P0 (WW) and P4 (Villager) remain. WWs win.
    """
    state = moderator_simple.state
    # Setup: Kill P1 (Seer) and P2 (Doctor) to simplify
    state.players[1].alive = False # Player "1"
    state.players[2].alive = False # Player "2"
    # Ensure moderator is in night phase
    state.day_count = 0 # Reset day for this test scenario
    state.phase = Phase.NIGHT
    moderator_simple.detailed_phase = DetailedPhase.NIGHT_START
    
    # NIGHT_START -> NIGHT_AWAIT_ACTIONS
    moderator_simple.advance({}) 

    assert moderator_simple.detailed_phase == DetailedPhase.NIGHT_AWAIT_ACTIONS
    assert state.phase == Phase.NIGHT
    active_ids_night_elim = moderator_simple.get_active_player_ids()
    assert "0" in active_ids_night_elim # Only WW P0 should be active (Doc and Seer dead)

    # Night actions: P0 (WW) targets P3 (Villager)
    # Since P2 (Doctor) is dead, no save.
    night_elim_actions_dict = {
        "0": VoteAction(actor_id="0", target_id="3")
    }
    # NIGHT_AWAIT_ACTIONS -> GAME_OVER (night actions resolved, game ends)
    moderator_simple.advance(night_elim_actions_dict)

    # P3 (player "3") should be eliminated
    assert not state.players[3].alive, "P3 (Player '3') should be eliminated by Werewolf P0 (Player '0')"
    night_0_history = state.history.get(0, []) # Elimination happens during night 0
    assert any(f"Last night, P3 was eliminated by werewolves. They were a {RoleConst.VILLAGER.value}." in entry.description for entry in night_0_history), "Elimination message not found"

    # Now P0 (WW) and P4 (Villager) are alive. WWs should win.
    assert moderator_simple.detailed_phase == DetailedPhase.GAME_OVER 
    assert moderator_simple.is_game_over(), "Game should be over, Werewolves win by numbers"
    
    # Game over message is logged on day_count 0 as game ends after night 0 actions
    assert any("Game Over: Werewolves Win!" in entry.description for entry in night_0_history), "Werewolves win message not found"
    assert state.players[0].alive # WW P0 is alive
    assert state.players[4].alive # Villager P4 is alive
    assert len(state.alive_players()) == 2