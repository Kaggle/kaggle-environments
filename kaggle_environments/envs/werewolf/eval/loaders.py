import json
import os
from collections import namedtuple
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

import json
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field, ConfigDict
import numpy as np

from kaggle_environments.envs.werewolf.game.consts import RoleConst, Team
from kaggle_environments.envs.werewolf.game.records import GameEndResultsDataEntry
from kaggle_environments.envs.werewolf.game.roles import Player, ROLE_CLASS_MAP
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


GameWinScore = namedtuple("GameWinScore", ['models', 'scores', 'roles'])

def get_score_role_df_from_games(games):
    data_points = []
    for game in games:
        agents = game['configuration']['agents']
        winner_ids = set(game['info']['GAME_END']['winner_ids'])
        model_ids = [agent['display_name'] for agent in agents]
        roles = [agent['role'] for agent in agents]
        scores = [1 if agent['id'] in winner_ids else 0 for agent in agents]
        data_points.append(GameWinScore(models=model_ids, scores=scores, roles=roles))
    return data_points


def extract_win_df_save(input_dir, output_path=None):
    games = get_games(input_dir)
    df = get_score_role_df_from_games(games)
    if output_path:
        df.to_csv(output_path)
    return games, df


class PlayerResult(BaseModel):
    id: str
    """An arbitrary identifier. Typically a human name."""

    agent_name: str
    """This is the underlying agent/model name."""

    survived: bool
    """survived till game end."""

    role: str
    team: str




class GameResult:
    def __init__(self, game_json: Dict, preserve_full_record: bool = False):
        game_json_struct = structify(game_json)
        self.game_end_info = game_json_struct.info.GAME_END
        self.moderator_observation = game_json_struct.info.MODERATOR_OBSERVATION
        self.winner_team: Team = Team(self.game_end_info.winner_team)
        self.costs = self.game_end_info.get('cost_summary')
        self.players = self._get_players()

        self.villagers = {player.id for player in self.players if player.role.team == Team.VILLAGERS}
        self.wolves = {player.id for player in self.players if player.role.team == Team.WEREWOLVES}

        self.id_to_agent = {player.id: player.agent.display_name for player in self.players}
        self.survive_to_end = {player.agent.display_name: player.alive for player in self.players}

        if preserve_full_record:
            self.game_json = game_json_struct
        else:
            self.game_json = None

    def __repr__(self) -> str:
        player_lines = []
        for player in sorted(self.players, key=lambda p: p.agent.display_name):
            status = "W" if player.role.team == self.winner_team else "L"
            elim_info = ""
            if not player.alive:
                elim_info = f" (eliminated {player.eliminated_during_phase} {player.eliminated_during_day})"
            
            player_lines.append(f"  - {player.agent.display_name} ({player.role.name}, {status}){elim_info}")
        
        player_str = "\n".join(player_lines)
        return (
            f"<GameResult: {self.winner_team.value} won. {len(self.players)} players.\n"
            f"{player_str}\n>"
        )

    def _get_players(self) -> List[Player]:
        out = []
        for item in self.game_end_info.all_players:
            player = Player(**item)
            player.role = ROLE_CLASS_MAP[item.agent.role]()
            out.append(player)
        return out

    def iterate_voting_mini_game(self):
        """
        Returns:
            (irp_results, vss_results)
            irp_results: List[Tuple[str, int]]
                A list of (agent_name, score) for everyday vote cast by a villager.
                Score is 1 if they voted for a werewolf, 0 otherwise.
            vss_results: List[Tuple[str, int]]
                A list of (agent_name, score) for villager votes on days a werewolf was exiled.
                Score is 1 if they voted for the exiled werewolf, 0 otherwise.
        """
        day_vote_events = {}
        werewolf_exile_events = {}
        for step in self.moderator_observation:
            for entry in step:
                if entry.data_type == "DayExileVoteDataEntry":
                    json_data = json.loads(entry.json_str)
                    day = json_data["day"]
                    day_vote_events.setdefault(day, [])
                    day_vote_events[day].append(json_data["data"])
                if entry.data_type == "DayExileElectedDataEntry":
                    json_data = json.loads(entry["json_str"])
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
