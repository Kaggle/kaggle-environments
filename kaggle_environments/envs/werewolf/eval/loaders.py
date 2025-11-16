import json
import os
from collections import namedtuple

import pandas as pd

import json
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field, ConfigDict
import numpy as np

from kaggle_environments.envs.werewolf.game.consts import RoleConst, Team
from kaggle_environments.envs.werewolf.game.records import GameEndResultsDataEntry
from kaggle_environments.envs.werewolf.game.roles import Player, ROLE_CLASS_MAP
from kaggle_environments.utils import structify


def get_games(input_dir):
    games = []
    for root, dirs, files in os.walk(input_dir):
        for fname in files:
            if fname == 'werewolf_game.json':
                games.append(json.load(open(os.path.join(root, fname))))
    return games


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
    def __init__(self, game_json: Dict):
        self.game_json = structify(game_json)
        self.game_end_info = self.game_json.info.GAME_END
        self.winner_team: Team = Team(self.game_end_info.winner_team)
        self.costs = self.game_end_info.get('cost_summary')
        self.players = self._get_players()

        self.villagers = {player.id for player in self.players if player.role.team == Team.VILLAGERS}
        self.wolves = {player.id for player in self.players if player.role.team == Team.WEREWOLVES}

        # self.werewolf_ids = {p.id for p in self.players if p.team == Team.WEREWOLVES}

        self.id_to_agent = {player.id: player.agent.display_name for player in self.players}
        # self.agent_to_id = {player.display_name: player.id for player in self.players}

        # self.winner_agents = {self.id_to_agent[wid] for wid in self.game_end_info.winner_ids}
        # self.loser_agents = {self.id_to_agent[lid] for lid in self.game_end_info.loser_ids}
        self.survive_to_end = {player.agent.display_name: player.alive for player in self.players}

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
        info = self.game_json.info
        day_vote_events = {}
        werewolf_exile_events = {}
        for step in info.MODERATOR_OBSERVATION:
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
