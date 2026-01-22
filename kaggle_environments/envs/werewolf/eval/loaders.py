import argparse
import json
import os
from collections import namedtuple, defaultdict
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from kaggle_environments.envs.werewolf.game.consts import Team
from kaggle_environments.envs.werewolf.game.roles import create_players_from_agents_config
from kaggle_environments.utils import structify


def _load_json(file_path):
    with open(file_path, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {file_path}")
            return None


def _extract_csv_row(file_path):
    """Helper to extract a single row for the CSV from a game file.
    
    Returns:
        tuple: (data_dict, error_message)
            - data_dict: The extracted data or None on failure.
            - error_message: Error string or None on success.
    """
    try:
        with open(file_path, 'r') as f:
            try:
                game = json.load(f)
            except json.JSONDecodeError:
                return None, f"JSONDecodeError in {file_path}"

        # Basic validation of structure
        if 'configuration' not in game or 'agents' not in game['configuration']:
            return None, f"Missing configuration/agents in {file_path}"

        agents = game['configuration']['agents']
        game_end = game.get('info', {}).get('GAME_END')

        # If the game didn't end properly, we might not have winner_ids
        if not game_end:
            return None, f"Missing GAME_END in {file_path}"

        winner_ids = set(game_end.get('winner_ids', []))
        
        # Recreate players to handle ID shuffling (randomize_ids)
        # This ensures we map the correct Agent (from config index) to the correct Player ID
        config = game.get('configuration', {})
        agents_config = config.get('agents', [])
        
        try:
            players = create_players_from_agents_config(
                agents_config,
                randomize_roles=config.get('randomize_roles', False),
                randomize_ids=config.get('randomize_ids', False),
                seed=config.get('seed')
            )
        except Exception as e:
            # Fallback if creation fails (e.g. valid seed missing), though unlikely for valid replays
            return None, f"Error creating players: {e}"

        model_ids = [p.agent.display_name for p in players]
        roles = [p.role.name for p in players]
        scores = [1 if p.id in winner_ids else 0 for p in players]

        # Calculate costs/tokens
        # We need to map player IDs to their costs. 
        # The costs logic relies on finding costs in steps for a given player ID.
        player_costs = defaultdict(float)
        player_prompt = defaultdict(int)
        player_completion = defaultdict(int)

        # Iterate steps (similar to _compute_costs)
        steps = game.get('steps', [])
        
        # We need a mapping from index in step to player ID.
        # But wait, step is list of agent-wise observations?
        # In Kaggle Werewolf, step is list of dicts. step[i] corresponds to... Player ID? Or Agent Index?
        # In raw obs, step[i] is for agent i (Kaggle Agent Index).
        # But the cost is inside 'action' which is inside 'step[i]'.
        # Is step[i] always for the SAME agent index i? Yes.
        # But does Agent Index i correspond to the same Player ID?
        # players[i] corresponds to Agent Index i.
        # So players[i].id is the ID for the agent at index i.
        
        for step in steps:
            for i, agent_idx in enumerate(step):
                if i >= len(players): continue # Should not happen
                p_id = players[i].id
                
                action = agent_idx.get('action', {})
                kwargs = action.get('kwargs', {})
                
                cost = kwargs.get('cost')
                prompt_t = kwargs.get('prompt_tokens')
                completion_t = kwargs.get('completion_tokens')
                
                if cost: player_costs[p_id] += float(cost)
                if prompt_t: player_prompt[p_id] += int(prompt_t)
                if completion_t: player_completion[p_id] += int(completion_t)
                
        # Create lists aligned with players (who are aligned with Agents)
        costs = [player_costs[p.id] for p in players]
        prompt_tokens = [player_prompt[p.id] for p in players]
        completion_tokens = [player_completion[p.id] for p in players]

        return {
            'models': model_ids,
            'scores': scores,
            'roles': roles,
            'costs': costs,
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens
        }, None
    except Exception as e:
        return None, f"Error processing {file_path}: {str(e)}"


def get_games(input_dir: str) -> List[dict]:
    """Loads all game replay JSONs from a directory, walking subdirectories.

    Args:
        input_dir: The root directory to search for .json replay files.

    Returns:
        A list of dictionaries, each representing a loaded game replay.
    """
    game_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".json"):
                game_files.append(os.path.join(root, file))

    with ProcessPoolExecutor() as executor:
        games = list(executor.map(_load_json, game_files))

    return [g for g in games if g is not None]


def _load_game_result(args):
    file_path, preserve_full_record = args
    game_json = _load_json(file_path)
    if game_json is None:
        raise ValueError(f"Failed to load JSON from {file_path}")
    return GameResult(game_json, preserve_full_record=preserve_full_record)


def get_game_results(
    input_dir: str, preserve_full_record: bool = False, max_workers: Optional[int] = None
) -> List["GameResult"]:
    """Loads all game replays and returns GameResult objects, in parallel.

    Args:
        input_dir: The root directory to search for .json replay files.
        preserve_full_record: If True, keeps the entire game JSON in memory
            (useful for debugging but consumes significant RAM).
        max_workers: The maximum number of worker processes to use.

    Returns:
        A list of GameResult objects.
    """
    game_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".json"):
                game_files.append(os.path.join(root, file))

    args = [(f, preserve_full_record) for f in game_files]
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_load_game_result, args))

    return [r for r in results if r is not None]


def extract_csv(input_dir: str, output_path: str, log_file_path: Optional[str] = None,
                max_workers: Optional[int] = None) -> pd.DataFrame:
    """Extracts game data to a CSV file in a memory-efficient way.
    
    Args:
        input_dir: Directory containing .json game files.
        output_path: Path to save the extracted CSV.
        log_file_path: Optional path to log errors.
        max_workers: Number of workers for parallel processing.
        
    Returns:
        pd.DataFrame: The extracted data.
    """
    game_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.json'):
                game_files.append(os.path.join(root, file))

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(_extract_csv_row, game_files), total=len(game_files), desc="Extracting CSV"))

    # Filter results
    data_points = []
    errors = []
    for res, err in results:
        if err:
            errors.append(err)
        elif res:
            data_points.append(res)

    # Log errors if requested
    if log_file_path and errors:
        with open(log_file_path, 'w') as f:
            f.write("Extraction Errors:\n")
            for error in errors:
                f.write(f"{error}\n")

    df = pd.DataFrame(data_points)
    df.to_csv(output_path, index=False)

    return df


ROLE_TO_TEAM = {
    "VILLAGER": Team.VILLAGERS,
    "WEREWOLF": Team.WEREWOLVES,
    "SEER": Team.VILLAGERS,
    "DOCTOR": Team.VILLAGERS,
}

Agent = namedtuple("Agent", ["display_name"])
Role = namedtuple("Role", ["name", "team"])
Player = namedtuple("Player", ["id", "agent", "role", "alive"])


class GameResult:
    """A memory-efficient representation of a game's outcome.

    This class processes a raw game replay dictionary to extract only the
    necessary information for evaluation, such as winners, player roles,
    voting history, and costs.

    Attributes:
        winner_team (Team): The team that won the game.
        players (List[Player]): A list of Player namedtuples.
        villagers (Set[int]): A set of player IDs belonging to the Villager team.
        wolves (Set[int]): A set of player IDs belonging to the Werewolf team.
        id_to_agent (Dict[int, str]): A mapping from player ID to agent display name.
        player_costs (Dict[int, float]): Mapping of player ID to total USD cost.
        player_tokens (Dict[int, int]): Mapping of player ID to total tokens used.
        irp_results (List[Tuple[str, int]]): Voting accuracy data for IRP metric.
        vss_results (List[Tuple[str, int]]): Voting accuracy data for VSS metric.
        player_durations (Dict[int, int]): Mapping of player ID to days survived.
    """

    def __init__(self, game_json: Dict, preserve_full_record: bool = False):
        """Initializes the GameResult.

        Args:
            game_json: The raw dictionary of the game replay.
            preserve_full_record: Whether to store the full `game_json` object.
        """
        if preserve_full_record:
            self.game_json = structify(game_json)
            game_end_info = self.game_json.info.GAME_END
            moderator_observation = self.game_json.info.MODERATOR_OBSERVATION
        else:
            self.game_json = None
            # Extract only what's needed to avoid holding the whole dict in memory
            info = game_json.get("info", {})
            game_end_info_raw = info.get("GAME_END", {})
            game_end_info = structify(game_end_info_raw)
            moderator_observation = structify(info.get("MODERATOR_OBSERVATION", []))

        self.winner_team: Team = Team(game_end_info.winner_team)
        self.players = self._get_players(game_end_info)

        # Derived attributes for convenience
        self.villagers = {p.id for p in self.players if p.role.team == Team.VILLAGERS}
        self.wolves = {p.id for p in self.players if p.role.team == Team.WEREWOLVES}
        self.id_to_agent = {p.id: p.agent.display_name for p in self.players}

        # Parse cost summary if available
        # The cost summary structure is expected to be found in game_end_info.cost_summary
        # Schema matches AgentCostSummary in werewolf.py
        # Compute costs (prefer cost_summary, fallback to steps)
        self._compute_costs(game_json, game_end_info)

        # Pre-compute voting results and discard moderator_observation
        self.irp_results, self.vss_results, self.player_durations = self._precompute_voting_results(
            moderator_observation, game_end_info)

    def _compute_costs(self, game_json, game_end_info):
        """Extracts costs from cost_summary or steps."""
        self.player_costs = {}
        self.player_tokens = {}

        # 1. Try cost_summary from GAME_END
        cost_summary = getattr(game_end_info, "cost_summary", None)
        if cost_summary:
            # cost_per_agent is a list of AgentCostSummary
            cost_per_agent = getattr(cost_summary, "cost_per_agent", []) or []
            for agent_summ in cost_per_agent:
                # AgentCostSummary has 'agent_config' (dict) and 'costs' (AgentCost)
                agent_config = getattr(agent_summ, "agent_config", None)

                p_id = None
                if agent_config:
                    p_id = getattr(agent_config, "id", None)
                    if p_id is None and isinstance(agent_config, dict):
                        p_id = agent_config.get("id")


                if p_id is not None:
                    costs = getattr(agent_summ, "costs", None)
                    if costs:
                        # AgentCost has total_cost, prompt_tokens, completion_tokens
                        total_cost = getattr(costs, "total_cost", 0.0)
                        prompt_tokens = getattr(costs, "prompt_tokens", 0)
                        completion_tokens = getattr(costs, "completion_tokens", 0)


                        self.player_costs[p_id] = total_cost
                        self.player_tokens[p_id] = prompt_tokens + completion_tokens
            return

        # 2. Fallback: Extract from steps
        if not game_json or "steps" not in game_json or "configuration" not in game_json:
            return

        # Map agent index to ID
        agents_config = game_json["configuration"].get("agents", [])
        idx_to_id = {}
        for i, agent_conf in enumerate(agents_config):
            if isinstance(agent_conf, dict):
                idx_to_id[i] = agent_conf.get("id")
            else:
                idx_to_id[i] = getattr(agent_conf, "id", None)

        for step in game_json["steps"]:
            for i, agent_idx in enumerate(step):
                p_id = idx_to_id.get(i)
                if not p_id:
                    continue

                action = agent_idx.get("action", {})
                kwargs = action.get("kwargs", {})

                cost = kwargs.get("cost")
                prompt_t = kwargs.get("prompt_tokens")
                completion_t = kwargs.get("completion_tokens")

                if cost is not None:
                    self.player_costs[p_id] = self.player_costs.get(p_id, 0.0) + float(cost)

                tokens = 0
                if prompt_t is not None:
                    tokens += int(prompt_t)
                if completion_t is not None:
                    tokens += int(completion_t)

                if tokens > 0:
                    self.player_tokens[p_id] = self.player_tokens.get(p_id, 0) + tokens


    def __repr__(self) -> str:
        player_lines = []
        for player in sorted(self.players, key=lambda p: p.agent.display_name):
            status = "W" if player.role.team == self.winner_team else "L"
            elim_info = "" if player.alive else " (eliminated)"
            cost_info = f", ${self.player_costs.get(player.id, 0.0):.4f}" if self.player_costs else ""
            player_lines.append(f"  - {player.agent.display_name} ({player.role.name}, {status}){elim_info}{cost_info}")

        player_str = "\n".join(player_lines)
        return f"<GameResult: {self.winner_team.value} won. {len(self.players)} players.\n{player_str}\n>"

    def _get_players(self, game_end_info) -> List[Player]:
        out = []
        survivors = set(getattr(game_end_info, "survivors_until_last_round_and_role", {}).keys())

        for p_info in getattr(game_end_info, "all_players", []):
            role_name = p_info.agent.role
            team = ROLE_TO_TEAM.get(role_name.upper())
            if team is None:
                print(f"Warning: Unknown role '{role_name}' found in game data.")

            player = Player(
                id=p_info.id,
                agent=Agent(display_name=p_info.agent.display_name),
                role=Role(name=role_name, team=team),
                alive=p_info.id in survivors,
            )
            out.append(player)
        return out

    def _precompute_voting_results(self, moderator_observation, game_end_info):
        """Extracts IRP, VSS scores and player durations from logs.

        This method processes the log once and stores the results, allowing the
        large observation object to be garbage collected.

        Args:
            moderator_observation: The raw event log from the moderator.
            game_end_info: The game end summary object.

        Returns:
            A tuple containing (irp_results, vss_results, player_durations).
        """
        day_vote_events = {}
        werewolf_exile_events = {}

        # Default duration is last_day for everyone.
        last_day = getattr(game_end_info, "last_day", 0)
        player_durations = {p.id: last_day for p in self.players}

        # Track eliminations to adjust duration
        # If a player is eliminated, their duration is the day/step of elimination.
        # We need to check for ELIMINATION events.

        for step in moderator_observation:
            for entry in step:
                data_type = getattr(entry, "data_type", None)
                json_str = getattr(entry, "json_str", "{}")

                # Try to extract costs from action events if not already present
                if not self.player_costs:
                    try:
                        json_data = json.loads(json_str)
                        actor_id = json_data.get("actor_id")
                        if actor_id is not None:
                            cost = json_data.get("cost")
                            prompt_t = json_data.get("prompt_tokens")
                            completion_t = json_data.get("completion_tokens")

                            total_tokens = 0
                            if prompt_t is not None:
                                total_tokens += int(prompt_t)
                            if completion_t is not None:
                                total_tokens += int(completion_t)

                            if total_tokens == 0:
                                total_tokens = int(json_data.get("tokens", 0))

                            if cost is not None:
                                self.player_costs[actor_id] = self.player_costs.get(actor_id, 0.0) + cost
                            if total_tokens > 0:
                                self.player_tokens[actor_id] = self.player_tokens.get(actor_id, 0) + total_tokens
                    except (json.JSONDecodeError, AttributeError):
                        pass


                if data_type == "DayExileVoteDataEntry":
                    json_data = json.loads(json_str)
                    day = json_data["day"]
                    day_vote_events.setdefault(day, [])
                    day_vote_events[day].append(json_data["data"])
                elif data_type == "DayExileElectedDataEntry":
                    json_data = json.loads(json_str)
                    if json_data["data"]["elected_player_id"] in self.wolves:
                        werewolf_exile_events[json_data["day"]] = json_data["data"]

                    # Update duration for exiled player
                    exiled_id = json_data["data"]["elected_player_id"]
                    if exiled_id in player_durations:
                        player_durations[exiled_id] = json_data["day"]

                elif data_type == "WerewolfNightEliminationElectedDataEntry":
                    # This entry implies a night elimination
                    pass

                # Generic check for ELIMINATION event which engine.py logs
                if getattr(entry, "event_name", "") == "ELIMINATION":
                    pass

        # Use elimination_info from game_end_info if available for accurate durations
        elimination_info = getattr(game_end_info, "elimination_info", None)
        if elimination_info:
            # elimination_info is likely a dict of player_id -> {day, reason, etc}
            # Check if it's a list or dict
            if isinstance(elimination_info, list):
                # Maybe list of elimination records
                pass
            elif isinstance(elimination_info, dict):
                for p_id, info in elimination_info.items():
                    # info might be a struct or dict
                    day = getattr(info, "day", None) or info.get("day")
                    if day is not None and p_id in player_durations:
                        player_durations[p_id] = day

        irp_results = []
        for day, entries in day_vote_events.items():
            for entry in entries:
                actor_id = entry["actor_id"]
                target_id = entry["target_id"]
                if actor_id in self.villagers:
                    agent_name = self.id_to_agent[actor_id]
                    score = 1 if target_id in self.wolves else 0
                    irp_results.append((agent_name, score))

        vss_results = []
        for day, item in werewolf_exile_events.items():
            exiled_wolf_id = item["elected_player_id"]
            for entry in day_vote_events.get(day, []):
                actor_id = entry["actor_id"]
                target_id = entry["target_id"]
                if actor_id in self.villagers:
                    agent_name = self.id_to_agent[actor_id]
                    score = 1 if target_id == exiled_wolf_id else 0
                    vss_results.append((agent_name, score))
        return irp_results, vss_results, player_durations

    def iterate_voting_mini_game(self):
        """Returns the pre-computed voting results.

        Returns:
            tuple: A tuple containing:
                - irp_results (List[Tuple[str, int]]): (agent_name, score) for everyday vote cast by a villager.
                  Score is 1 if they voted for a werewolf, 0 otherwise.
                - vss_results (List[Tuple[str, int]]): (agent_name, score) for villager votes on days a werewolf was exiled.
                  Score is 1 if they voted for the exiled werewolf, 0 otherwise.
        """
        return self.irp_results, self.vss_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract game records to CSV.")
    parser.add_argument("input_dir", help="Directory containing .json game files.")
    parser.add_argument("output_path", help="Path to save the extracted CSV.")
    parser.add_argument("--log-file", help="Path to log errors.", default=None)
    parser.add_argument("--max-workers", help="Number of worker processes.", type=int, default=None)

    args = parser.parse_args()

    print(f"Extracting games from {args.input_dir} to {args.output_path}...")
    df = extract_csv(args.input_dir, args.output_path, log_file_path=args.log_file, max_workers=args.max_workers)
    print(f"Done. Extracted {len(df)} records.")
