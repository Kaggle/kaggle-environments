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
        
        # Pre-compute voting results and discard moderator_observation
        self.irp_results, self.vss_results = self._precompute_voting_results(moderator_observation)

    def __repr__(self) -> str:
        player_lines = []
        for player in sorted(self.players, key=lambda p: p.agent.display_name):
            status = "W" if player.role.team == self.winner_team else "L"
            elim_info = "" if player.alive else " (eliminated)"
            player_lines.append(f"  - {player.agent.display_name} ({player.role.name}, {status}){elim_info}")

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

    def _precompute_voting_results(self, moderator_observation):
        """
        Extracts IRP and VSS scores from the moderator observation log.
        This method processes the log once and stores the results, allowing the
        large observation object to be garbage collected.
        """
        day_vote_events = {}
        werewolf_exile_events = {}
        for step in moderator_observation:
            for entry in step:
                if getattr(entry, 'data_type', None) == "DayExileVoteDataEntry":
                    json_data = json.loads(entry.json_str)
                    day = json_data["day"]
                    day_vote_events.setdefault(day, [])
                    day_vote_events[day].append(json_data["data"])
                if getattr(entry, 'data_type', None) == "DayExileElectedDataEntry":
                    json_data = json.loads(entry.json_str)
                    # Check if the exiled player was a werewolf
                    if json_data["data"]['elected_player_id'] in self.wolves:
                        werewolf_exile_events[json_data["day"]] = json_data["data"]

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
        return irp_results, vss_results

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
