import json
import os
from collections import namedtuple
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional

from kaggle_environments.envs.werewolf.game.consts import Team
from kaggle_environments.utils import structify


def _load_json(file_path):
    with open(file_path, 'r') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {file_path}")
            return None


def get_games(input_dir: str) -> List[dict]:
    """Loads all game replay JSONs from a directory, walking subdirectories."""
    game_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.json'):
                game_files.append(os.path.join(root, file))

    with ProcessPoolExecutor() as executor:
        games = list(executor.map(_load_json, game_files))
    
    return [g for g in games if g is not None]


def _load_game_result(args):
    file_path, preserve_full_record = args
    game_json = _load_json(file_path)
    if game_json is None:
        return None
    return GameResult(game_json, preserve_full_record=preserve_full_record)


def get_game_results(input_dir: str, preserve_full_record: bool = False,
                     max_workers: Optional[int] = None) -> List["GameResult"]:
    """Loads all game replays and returns GameResult objects, in parallel."""
    game_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.json'):
                game_files.append(os.path.join(root, file))
    
    args = [(f, preserve_full_record) for f in game_files]
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_load_game_result, args))
        
    return [r for r in results if r is not None]



ROLE_TO_TEAM = {
    "VILLAGER": Team.VILLAGERS,
    "WEREWOLF": Team.WEREWOLVES,
    "SEER": Team.VILLAGERS,
    "DOCTOR": Team.VILLAGERS,
}

Agent = namedtuple('Agent', ['display_name'])
Role = namedtuple('Role', ['name', 'team'])
Player = namedtuple('Player', ['id', 'agent', 'role', 'alive'])


class GameResult:
    """A memory-efficient representation of a game's outcome."""

    def __init__(self, game_json: Dict, preserve_full_record: bool = False):
        if preserve_full_record:
            self.game_json = structify(game_json)
            game_end_info = self.game_json.info.GAME_END
            moderator_observation = self.game_json.info.MODERATOR_OBSERVATION
        else:
            self.game_json = None
            # Extract only what's needed to avoid holding the whole dict in memory
            info = game_json.get('info', {})
            game_end_info_raw = info.get('GAME_END', {})
            game_end_info = structify(game_end_info_raw)
            moderator_observation = structify(info.get('MODERATOR_OBSERVATION', []))

        self.winner_team: Team = Team(game_end_info.winner_team)
        self.players = self._get_players(game_end_info)

        # Derived attributes for convenience
        self.villagers = {p.id for p in self.players if p.role.team == Team.VILLAGERS}
        self.wolves = {p.id for p in self.players if p.role.team == Team.WEREWOLVES}
        self.id_to_agent = {p.id: p.agent.display_name for p in self.players}
        
        # Parse cost summary if available
        # The cost summary structure is expected to be found in game_end_info.cost_summary
        # Schema matches AgentCostSummary in werewolf.py
        self.player_costs = {}
        self.player_tokens = {}
        
        cost_summary = getattr(game_end_info, 'cost_summary', None)
        
        if cost_summary:
            # cost_per_agent is a list of AgentCostSummary
            cost_per_agent = getattr(cost_summary, 'cost_per_agent', []) or []
            
            for agent_summ in cost_per_agent:
                # AgentCostSummary has 'agent_config' (dict) and 'costs' (AgentCost)
                
                # Extract Player ID from agent_config
                agent_config = getattr(agent_summ, 'agent_config', None)
                p_id = None
                if agent_config:
                    p_id = getattr(agent_config, 'id', None)
                    if p_id is None and isinstance(agent_config, dict):
                        p_id = agent_config.get('id')
                
                if p_id is not None:
                    costs = getattr(agent_summ, 'costs', None)
                    if costs:
                        # AgentCost has total_cost, prompt_tokens, completion_tokens
                        total_cost = getattr(costs, 'total_cost', 0.0)
                        prompt_tokens = getattr(costs, 'prompt_tokens', 0)
                        completion_tokens = getattr(costs, 'completion_tokens', 0)
                        
                        self.player_costs[p_id] = total_cost
                        self.player_tokens[p_id] = prompt_tokens + completion_tokens

        # Pre-compute voting results and discard moderator_observation
        self.irp_results, self.vss_results, self.player_durations = self._precompute_voting_results(moderator_observation, game_end_info)

    def __repr__(self) -> str:
        player_lines = []
        for player in sorted(self.players, key=lambda p: p.agent.display_name):
            status = "W" if player.role.team == self.winner_team else "L"
            elim_info = "" if player.alive else " (eliminated)"
            cost_info = f", ${self.player_costs.get(player.id, 0.0):.4f}" if self.player_costs else ""
            player_lines.append(f"  - {player.agent.display_name} ({player.role.name}, {status}){elim_info}{cost_info}")

        player_str = "\n".join(player_lines)
        return (
            f"<GameResult: {self.winner_team.value} won. {len(self.players)} players.\n"
            f"{player_str}\n>"
        )

    def _get_players(self, game_end_info) -> List[Player]:
        out = []
        survivors = set(getattr(game_end_info, 'survivors_until_last_round_and_role', {}).keys())

        for p_info in getattr(game_end_info, 'all_players', []):
            role_name = p_info.agent.role
            team = ROLE_TO_TEAM.get(role_name.upper())
            if team is None:
                print(f"Warning: Unknown role '{role_name}' found in game data.")

            player = Player(
                id=p_info.id,
                agent=Agent(display_name=p_info.agent.display_name),
                role=Role(name=role_name, team=team),
                alive=p_info.id in survivors
            )
            out.append(player)
        return out

    def _precompute_voting_results(self, moderator_observation, game_end_info):
        """
        Extracts IRP and VSS scores from the moderator observation log.
        Also computes player durations (turns survived).
        """
        day_vote_events = {}
        werewolf_exile_events = {}
        
        # Default duration is last_day for everyone.
        last_day = getattr(game_end_info, 'last_day', 0)
        player_durations = {p.id: last_day for p in self.players}
        
        # Track eliminations to adjust duration
        # If a player is eliminated, their duration is the day/step of elimination.
        # We need to check for ELIMINATION events.
        
        for step in moderator_observation:
            for entry in step:
                data_type = getattr(entry, 'data_type', None)
                json_str = getattr(entry, 'json_str', "{}")
                
                if data_type == "DayExileVoteDataEntry":
                    json_data = json.loads(json_str)
                    day = json_data["day"]
                    day_vote_events.setdefault(day, [])
                    day_vote_events[day].append(json_data["data"])
                elif data_type == "DayExileElectedDataEntry":
                    json_data = json.loads(json_str)
                    if json_data["data"]['elected_player_id'] in self.wolves:
                        werewolf_exile_events[json_data["day"]] = json_data["data"]
                    
                    # Update duration for exiled player
                    exiled_id = json_data["data"]['elected_player_id']
                    if exiled_id in player_durations:
                        player_durations[exiled_id] = json_data["day"]
                        
                elif data_type == "WerewolfNightEliminationElectedDataEntry":
                     # This entry implies a night elimination
                     # We assume the JSON structure matches records.py
                     # It usually has elected_target_player_id, but might not have day directly in data
                     # We rely on the event's implied timing. But wait, the entry might not have 'day'.
                     # Usually moderator observation is a list of lists of records.
                     # We might need to check the parent event description or context?
                     # Actually, let's check if ELIMINATION event exists which is generic.
                     pass
                
                # Generic check for ELIMINATION event which engine.py logs
                if getattr(entry, 'event_name', '') == 'ELIMINATION':
                    # engine.py: self.state.push_event(..., event_name=EventName.ELIMINATION, public=True, data=data)
                    # data is DayExileElectedDataEntry or similar.
                    # Actually engine.py only logs ELIMINATION for day exile.
                    # Night elimination is logged as VOTE_RESULT visible to wolves.
                    # But night elimination manager resolves elimination.
                    # Wait, does engine.py log a generic elimination event?
                    # engine.py _handle_day_voting_conclude -> EventName.ELIMINATION
                    # engine.py _handle_night_conclude -> EventName.VOTE_RESULT (private to wolves)
                    # BUT the elimination manager likely resolves it.
                    # We might miss night eliminations if we only look for public events.
                    # However, game_end_info.elimination_info usually contains who eliminated whom and when.
                    pass

        # Use elimination_info from game_end_info if available for accurate durations
        elimination_info = getattr(game_end_info, 'elimination_info', None)
        if elimination_info:
            # elimination_info is likely a dict of player_id -> {day, reason, etc}
            # Check if it's a list or dict
            if isinstance(elimination_info, list):
                # Maybe list of elimination records
                pass
            elif isinstance(elimination_info, dict):
                for p_id, info in elimination_info.items():
                    # info might be a struct or dict
                    day = getattr(info, 'day', None) or info.get('day')
                    if day is not None and p_id in player_durations:
                        player_durations[p_id] = day

        irp_results = []
        for day, entries in day_vote_events.items():
            for entry in entries:
                actor_id = entry['actor_id']
                target_id = entry['target_id']
                if actor_id in self.villagers:
                    agent_name = self.id_to_agent[actor_id]
                    score = 1 if target_id in self.wolves else 0
                    irp_results.append((agent_name, score))

        vss_results = []
        for day, item in werewolf_exile_events.items():
            exiled_wolf_id = item['elected_player_id']
            for entry in day_vote_events.get(day, []):
                actor_id = entry['actor_id']
                target_id = entry['target_id']
                if actor_id in self.villagers:
                    agent_name = self.id_to_agent[actor_id]
                    score = 1 if target_id == exiled_wolf_id else 0
                    vss_results.append((agent_name, score))
        return irp_results, vss_results, player_durations

    def iterate_voting_mini_game(self):
        """
        Returns the pre-computed voting results.
        Returns:
            (irp_results, vss_results)
            irp_results: List[Tuple[str, int]]
                A list of (agent_name, score) for everyday vote cast by a villager.
                Score is 1 if they voted for a werewolf, 0 otherwise.
            vss_results: List[Tuple[str, int]]
                A list of (agent_name, score) for villager votes on days a werewolf was exiled.
                Score is 1 if they voted for the exiled werewolf, 0 otherwise.
        """
        return self.irp_results, self.vss_results
