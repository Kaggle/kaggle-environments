import argparse
import functools
import hashlib
import multiprocessing
import os
import pickle
import subprocess
import sys
from collections import defaultdict, namedtuple
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterator, List, Optional, Tuple, Union

import jax.numpy as jnp
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import polarix as plx
from openskill.models import PlackettLuce
from plotly.subplots import make_subplots
from tqdm import tqdm

from kaggle_environments.envs.werewolf.eval.loaders import Agent, Role, Player, _load_game_result
from kaggle_environments.envs.werewolf.game.consts import Team

# Workaround for broken google.colab import in some environments (incompatibility with IPython)
# Plotly tries to import google.colab to detect the environment. If google.colab is installed
# but broken (e.g. AttributeError: type object 'TermColors' has no attribute 'Green'),
# Plotly crashes. We force it to fail with ImportError so Plotly skips it.
try:
    import google.colab
except AttributeError:
    sys.modules["google.colab"] = None
except ImportError:
    pass

# Set a default sophisticated template
pio.templates.default = "plotly_white"

_cached_git_hash = None
def _get_git_hash() -> str:
    global _cached_git_hash
    if _cached_git_hash is not None:
        return _cached_git_hash
    try:
        _cached_git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('utf-8')
    except Exception:
        _cached_git_hash = "unknown_git"
    return _cached_git_hash


def _get_input_hash(file_list: List[str]) -> str:
    """Computes a hash of the input files (modification times + size).
    
    Using just path+mtime+size is much faster than hashing content.
    """
    hasher = hashlib.md5()
    # Sort to ensure deterministic order
    for file_path in sorted(file_list):
        path = Path(file_path)
        try:
            stat = path.stat()
            # Include path, mtime, size in hash
            # Relative path is better but absolute is safer if different machines (though cache is local)
            # We use absolute path here since it's local cache
            info = f"{path.absolute()}:{stat.st_mtime}:{stat.st_size}"
            hasher.update(info.encode('utf-8'))
        except OSError:
            pass  # Skip missing files

    return hasher.hexdigest()


def _compute_mean_std_sem(data: np.ndarray, axis: Optional[int] = None) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]]:
    """Unified logic for computing mean, StdDev and SEM, handling edge cases consistently.

    Optimized for speed by avoiding slow nanmean/nanstd and using masks directly.
    Supports both scalar inputs (axis=None) and vectorized inputs (axis=0 or 1 for bootstrapping).
    Returns (mean, stddev, sem).
    """
    mask = ~np.isnan(data)
    count = np.sum(mask, axis=axis)
    
    # Fill NaNs with 0 for summation
    clean_data = np.where(mask, data, 0.0)
    
    data_sum = np.sum(clean_data, axis=axis)
    mean = np.divide(data_sum, count, out=np.zeros_like(data_sum, dtype=np.float64), where=count > 0)
    
    # Variance: E[X^2] - (E[X])^2
    data_sq_sum = np.sum(clean_data**2, axis=axis)
    mean_sq = np.divide(data_sq_sum, count, out=np.zeros_like(data_sum, dtype=np.float64), where=count > 0)
    
    # Use max(0, var) to avoid tiny negative numbers due to precision
    var = np.maximum(0.0, mean_sq - mean**2)
    
    # Bessel's correction for sample variance: var * (n / (n-1))
    sample_var = np.divide(var * count, count - 1, out=np.zeros_like(var), where=count > 1)
    std = np.sqrt(sample_var)
    
    # SEM is std / sqrt(n)
    sem = np.divide(std, np.sqrt(count), out=np.zeros_like(std), where=count > 1)

    if axis is None:
        return float(mean), float(std), float(sem)
    return mean, std, sem


def _mean_sem(values: List[float]) -> Tuple[float, float]:
    """Helper to calculate mean and standard error of the mean.

    Args:
        values: A list of numerical values.

    Returns:
        A tuple (mean, sem). Returns (0.0, 0.0) if the list is empty.
    """
    if not values:
        return 0.0, 0.0
    mean, std, sem = _compute_mean_std_sem(np.asarray(values))
    return mean, sem


def _safe_load_game_result(args):
    try:
        return _load_game_result(args), None
    except Exception as e:
        return None, e


# --- Plotting utils ---


def _save_figure(fig: "go.Figure", output_path: Union[str, List[str], None], width=None, height=None):
    """Saves a Plotly figure to one or multiple files.

    Args:
        fig: The Plotly Figure object.
        output_path: A single filename (str) or a list of filenames (List[str]).
        width: Width of the output image.
        height: Height of the output image.
    """
    if not output_path:
        return

    # Handle multiple paths (recursion)
    if isinstance(output_path, (list, tuple)):
        for path in output_path:
            _save_figure(fig, path, width, height)
        return

    # Handle single path
    ext = os.path.splitext(output_path)[1].lower()
    try:
        if ext == ".html":
            fig.write_html(output_path)
        elif ext in [".png", ".jpg", ".jpeg"]:
            # scale=3 ensures high DPI (Retina quality)
            fig.write_image(output_path, width=width, height=height, scale=3)
        elif ext in [".pdf", ".svg"]:
            fig.write_image(output_path, width=width, height=height)
        else:
            print(f"Unknown format {ext}, defaulting to HTML.")
            fig.write_html(output_path + ".html")
        print(f"Saved chart to {output_path}")
    except ValueError as e:
        print(f"Error saving to {output_path} (did you install 'kaleido'?): {e}")


def _get_color_discrete_sequence(n):
    """Returns a sophisticated color palette."""
    # Custom palette: Teal, Indigo, Rose, Amber, Cyan, Emerald
    colors = ["#0d9488", "#4338ca", "#e11d48", "#d97706", "#0891b2", "#059669"]
    if n > len(colors):
        return px.colors.qualitative.Bold
    return colors


LightGame = namedtuple("LightGame", ["players", "winner_team", "irp_results", "vss_results", "player_durations"])


# --- Worker Globals and Functions ---

def _openskill_bootstrap_worker_fast(games_data, num_agents, model):
    if model is None:
        return []
    ratings = GameSetEvaluator._compute_openskill_ratings_fast(games_data, num_agents, model)
    return [r.mu for r in ratings]


def _gte_bootstrap_worker_fast(matrix_tuple, agents, tasks):
    """Fast worker using pre-computed matrices."""
    mean_matrix, stddev_matrix = matrix_tuple

    # Regularization logic matches original
    if mean_matrix.size > 0:
        rnd = np.random.default_rng()
        for j in range(mean_matrix.shape[1]):
            # ptp fails on empty arrays, but we checked size > 0
            if np.ptp(mean_matrix[:, j]) < 1e-9:
                mean_matrix[:, j] += rnd.random(mean_matrix.shape[0]) * 1e-6
                stddev_matrix[:, j] += rnd.random(mean_matrix.shape[0]) * 1e-6

    # Solve Game using Polarix
    game_plx = plx.agent_vs_task_game(
        agents=agents, tasks=tasks, agent_vs_task=mean_matrix, agent_vs_task_stddev=stddev_matrix,
        task_player='metric', normalizer='winrate'
    )
    res = plx.solve(game_plx, plx.ce_maxent, disable_progress_bar=True)
    marginals = plx.marginals_from_joint(res.joint)
    r2m_contributions = plx.joint_payoffs_contribution(
        game_plx.payoffs, res.joint, rating_player=1, contrib_player=0
    )
    m2r_contributions = plx.joint_payoffs_contribution(
        game_plx.payoffs, res.joint, rating_player=0, contrib_player=1
    )

    ratings_np = [np.array(r) for r in res.ratings]
    joint_np = np.array(res.joint)
    marginals_np = [np.array(m) for m in marginals]
    r2m_contributions_np = np.array(r2m_contributions)
    m2r_contributions_np = np.array(m2r_contributions)

    game_meta = SimpleNamespace(actions=game_plx.actions)

    return ratings_np, joint_np, marginals_np, r2m_contributions_np, m2r_contributions_np, game_meta


def _default_elo():
    return 1200.0


def _default_gte_contrib():
    return 0.0, 0.0


class AgentMetrics:
    """Stores and calculates performance metrics for a single agent.

    Attributes:
        agent_name (str): The display name of the agent.
        wins (List[int]): List of win/loss (1/0) outcomes.
        total_costs (List[float]): List of costs incurred per game.
        total_tokens (List[int]): List of total tokens used per game.
        durations (List[int]): List of days survived per game.
        gte_rating (Tuple[float, float]): (mean, std) of GTE rating.
        elo (float): Current Elo rating.
    """

    def __init__(self, agent_name: str, openskill_model):
        self.agent_name = agent_name
        self.wins: List[int] = []
        self.wins_by_role: Dict[str, List[int]] = defaultdict(list)
        self.irp_scores: List[int] = []
        self.vss_scores: List[int] = []
        self.survival_scores: List[int] = []
        self.survival_by_role: Dict[str, List[int]] = defaultdict(list)
        self.wd_survival_scores: List[int] = []
        self.wd_survival_by_role: Dict[str, List[int]] = defaultdict(list)
        self.wd_irp_scores: List[int] = []
        self.wd_vss_scores: List[int] = []

        # Dominance Metrics (winning games only)
        self.margin_of_win_scores: List[float] = []
        self.speed_of_win_scores: List[float] = []

        # Costs
        self.total_costs: List[float] = []
        self.total_tokens: List[int] = []
        self.total_prompt_tokens: List[int] = []
        self.total_completion_tokens: List[int] = []
        self.durations: List[int] = []  # days survived per game

        # For GTE
        self.gte_rating: Tuple[float, float] = (0.0, 0.0)
        self.gte_contributions: Dict[str, Tuple[float, float]] = defaultdict(_default_gte_contrib)

        # Ratings
        self.elo: float = 1200.0
        self.elo_std: float = 0.0
        self.openskill_model = openskill_model
        self.openskill_rating = None
        self.openskill_mu_std: float = 0.0

    def set_agent_name(self, name: str):
        self.agent_name = name
        self.openskill_rating = self.openskill_model.rating(name=name) if self.openskill_model else None

    def get_win_rate(self) -> Tuple[float, float]:
        """Returns the win rate and its standard error."""
        return _mean_sem(self.wins)

    def get_avg_cost(self) -> Tuple[float, float]:
        """Returns the average cost per game and its standard error."""
        return _mean_sem(self.total_costs)

    def get_avg_tokens(self) -> Tuple[float, float]:
        """Returns the average tokens per game and its standard error."""
        return _mean_sem(self.total_tokens)

    def get_avg_cost_per_turn(self) -> Tuple[float, float]:
        """Returns the average cost per turn and its standard error.

        Note: If duration is 0 (e.g. died immediately), cost/turn is assumed 0.0.
        """
        costs_per_turn = []
        for c, d in zip(self.total_costs, self.durations):
            if d > 0:
                costs_per_turn.append(c / d)
            else:
                costs_per_turn.append(0.0)
        return _mean_sem(costs_per_turn)

    def get_avg_output_tokens_per_turn(self) -> Tuple[float, float]:
        """Returns the average output tokens per turn and its standard error."""
        tokens_per_turn = []
        for c, d in zip(self.total_completion_tokens, self.durations):
            if d > 0:
                tokens_per_turn.append(c / d)
            else:
                tokens_per_turn.append(0.0)
        return _mean_sem(tokens_per_turn)

    def get_avg_input_tokens_per_turn(self) -> Tuple[float, float]:
        """Returns the average input tokens per turn and its standard error."""
        tokens_per_turn = []
        for c, d in zip(self.total_prompt_tokens, self.durations):
            if d > 0:
                tokens_per_turn.append(c / d)
            else:
                tokens_per_turn.append(0.0)
        return _mean_sem(tokens_per_turn)

    def get_win_rate_for_role(self, role: str) -> Tuple[float, float]:
        return _mean_sem(self.wins_by_role.get(role, []))

    def get_irp(self) -> Tuple[float, float]:
        """Returns IRP (Identification Precision) mean and sem."""
        return _mean_sem(self.irp_scores)

    def get_vss(self) -> Tuple[float, float]:
        """Returns VSS (Voting Success Score) mean and sem."""
        return _mean_sem(self.vss_scores)

    def get_ksr(self) -> Tuple[float, float]:
        """Returns KSR (Kill Survival Rate) mean and sem."""
        return _mean_sem(self.survival_scores)

    def get_ksr_for_role(self, role: str) -> Tuple[float, float]:
        return _mean_sem(self.survival_by_role.get(role, []))

    def get_wd_ksr(self) -> Tuple[float, float]:
        return _mean_sem(self.wd_survival_scores)

    def get_wd_ksr_for_role(self, role: str) -> Tuple[float, float]:
        return _mean_sem(self.wd_survival_by_role.get(role, []))

    def get_wd_irp(self) -> Tuple[float, float]:
        return _mean_sem(self.wd_irp_scores)

    def get_wd_vss(self) -> Tuple[float, float]:
        return _mean_sem(self.wd_vss_scores)

    def get_margin_of_win(self) -> Tuple[float, float]:
        """Returns Margin of Win (living teammates / total teammates) mean and sem, for winning games."""
        return _mean_sem(self.margin_of_win_scores)

    def get_speed_of_win(self) -> Tuple[float, float]:
        """Returns Speed of Win (1 / turn count) mean and sem, for winning games."""
        return _mean_sem(self.speed_of_win_scores)


class GameSetEvaluator:
    """Evaluates a set of game replays and calculates metrics for each agent.

    This class handles the loading of games, computation of various metrics
    (Win Rate, IRP, VSS, Elo, OpenSkill, GTE), and generation of plots.

    IRP: Identification Precision. Quantifies the precision with which a player deduces the roles of other
        participants in Werewolf.
    VSS: Voting Success Score. Assesses the efficacy of a player's voting decisions during pivotal moments in a
        game of Werewolf. (When a werewolf is exiled, did the player (in villager team) vote for the exiled?
    KSR: Key Role Survival Rate. Evaluates the likelihood of key roles, such as Seer or Werewolf surviving until the
        end of the game.
    """

    def __init__(
        self,
        input_dir: Union[str, List[str]],
        gte_tasks: Union[str, List[str]] = "win_dependent",
        preserve_full_game_records: bool = False,
        error_log_path: str = "game_loading_errors.log",
        cache_dir: str = ".werewolf_metrics_cache",
        seed: int = 42,
    ):
        if isinstance(input_dir, str):
            input_dirs = [input_dir]
        else:
            input_dirs = input_dir

        self.seed = seed

        self.games = []
        game_files = []
        for directory in input_dirs:
            for root, _, files in os.walk(directory):
                for file in files:
                    if file.endswith(".json"):
                        game_files.append(os.path.join(root, file))

        # Setup GTE Tasks (moved up for cache key)
        roles = ["Doctor", "Seer", "Villager", "Werewolf"] # Hardcoded common roles if games aren't loaded yet? 
        # Actually we can't fully know roles if games aren't loaded, but 'gte_tasks' logic usually uses strings.
        # If 'gte_tasks' == 'non_win_dependent', it relies on 'roles'.
        # But 'roles' was derived from games: roles = sorted(list(set(p.role.name for g in self.games...)))
        # This creates a dependency: Load Games -> Know Roles -> Define Tasks
        # But Cache Key needs Tasks -> Load Games (from cache).
        # Catch-22 if we depend on dynamic roles from games.
        
        # However, usually roles are standard.
        # Let's defer strict Role dependency or default it?
        # The default 'win_dependent' uses hardcoded strings.
        
        # If gte_tasks is 'non_win_dependent', it needs roles.
        # The previous code derived roles from self.games AFTER loading.
        # If loading from cache, self.games is populated.
        # If NOT loading from cache, self.games is empty initially.
        
        # We can calculate 'tasks_hash' based on the 'gte_tasks' ARGUMENT string/list initially.
        # If it's a string like "win_dependent", that string is enough for the key?
        # YES. Because if "win_dependent" maps to a specific list, that mapping is static code (unless code changes, which git_hash covers... mostly).
        # But if we change the mapping in code without commit, git_hash misses it.
        # But we can't expand "non_win_dependent" without games if it depends on games.
        
        # Compromise: Add str(gte_tasks) to cache key.
        # This covers the switch between "win_dependent" and "role_win_rates".
        
        # Caching logic
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Calculate cache key
        self.git_hash = _get_git_hash()
        self.input_hash = _get_input_hash(game_files)
        # We also need to consider preserve_full_game_records and gte_tasks in the cache key
        cache_key = f"{self.git_hash}_{self.input_hash}_{preserve_full_game_records}_{str(gte_tasks)}.pkl"
        # Sanitize cache_key (it might be too long or contain bad chars if gte_tasks is a long list)
        if len(cache_key) > 255 or "[" in cache_key:
             # Use hash for the tail if too long/complex
             tasks_hash = hashlib.md5(str(gte_tasks).encode()).hexdigest()
             cache_key = f"{self.git_hash}_{self.input_hash}_{preserve_full_game_records}_{tasks_hash}.pkl"
            
        cache_path = self.cache_dir / cache_key

        loaded_from_cache = False
        errors = {}
        if cache_path.exists():
            try:
                print(f"Loading cached game results from {cache_path}...")
                with open(cache_path, 'rb') as f:
                    self.games = pickle.load(f)
                loaded_from_cache = True
                print(f"Successfully loaded {len(self.games)} games from cache.")
                
                # Reconstruct gte_game structure for plotting if not in cache (or even if it is, for safety)
                # It just needs actions [agents, tasks]
                # Note: self.metrics isn't populated yet, doing it later.
            except Exception as e:
                print(f"Failed to load cache: {e}. Reloading games...")

        errors = defaultdict(int)

        if not loaded_from_cache:
            args_list = [(f, preserve_full_game_records) for f in game_files]
            errors = defaultdict(list)

            with ProcessPoolExecutor(
                max_workers=(os.cpu_count() or 1) * 4, mp_context=multiprocessing.get_context("spawn")
            ) as executor:
                results_iter = tqdm(
                    executor.map(_safe_load_game_result, args_list), total=len(args_list), desc="Loading Games"
                )
                # zip preserves order because executor.map preserves order
                for (args, (result, error)) in zip(args_list, results_iter):
                    if error:
                        error_key = f"{type(error).__name__}: {str(error)}"
                        file_path = args[0]
                        errors[error_key].append(file_path)
                    elif result:
                        self.games.append(result)

            # Save to cache if successful
            if self.games:
                try:
                    print(f"Saving game results to cache {cache_path}...")
                    with open(cache_path, "wb") as f:
                        pickle.dump(self.games, f)
                except Exception as e:
                    print(f"Failed to save cache: {e}")

        if errors:
            print("\nErrors encountered during game loading:")
            Path(error_log_path).parent.mkdir(parents=True, exist_ok=True)
            with open(error_log_path, "w") as f:
                f.write("Game Loading Errors Report\n")
                f.write("==========================\n")
                for error_key, files in sorted(errors.items(), key=lambda x: len(x[1]), reverse=True):
                    count = len(files)
                    print(f"  {error_key}: {count} files")
                    f.write(f"\n{error_key}: {count} files\n")
                    for file_path in files:
                        f.write(f"  - {file_path}\n")
            print(f"Detailed error report saved to {error_log_path}")

        self.openskill_model = PlackettLuce()

        self.metrics: Dict[str, AgentMetrics] = defaultdict(
            lambda: AgentMetrics(agent_name=None, openskill_model=self.openskill_model)
        )

        # For GTE
        self.gte_game = None
        self.gte_joint = None
        self.gte_ratings = None
        self.gte_marginals = None
        self.gte_contributions_raw = None
        self.gte_metric_contributions_raw = None

        # Determine roles for task generation (fallback to defaults if no games loaded yet? No, we have games now)
        roles = sorted(list(set(p.role.name for g in self.games for p in g.players))) if self.games else ["Doctor", "Seer", "Villager", "Werewolf"]
        
        if gte_tasks == "win_dependent":
            self.gte_tasks = [
                "WinRate-Doctor",
                "WinRate-Seer",
                "WinRate-Villager",
                "WinRate-Werewolf",
                "WD-KSR-Werewolf",
                "WD-KSR-Seer",
                "WD-KSR-Doctor",
                "Margin of Win",
                "Speed of Win",
            ]
        elif gte_tasks == "non_win_dependent":
            self.gte_tasks = [f"WinRate-{r}" for r in roles] + [
                "KSR-Werewolf",
                "KSR-Seer",
                "KSR-Doctor",
                "Margin of Win",
                "Speed of Win",
            ]
        elif gte_tasks == "role_win_rates":
            self.gte_tasks = ["WinRate-Doctor", "WinRate-Seer", "WinRate-Villager", "WinRate-Werewolf"]
        elif isinstance(gte_tasks, list):
            self.gte_tasks = gte_tasks
        else:
            self.gte_tasks = [gte_tasks]

    @staticmethod
    def _compute_elo_ratings(games: List) -> Dict[str, float]:
        """Computes Elo ratings for a set of games using a team-based update."""
        elos = defaultdict(_default_elo)
        for game in games:
            villager_agents = []
            werewolf_agents = []
            for player in game.players:
                agent_name = player.agent.display_name
                if player.role.team == Team.VILLAGERS:
                    villager_agents.append(agent_name)
                else:
                    werewolf_agents.append(agent_name)
            v_elos = [elos[a] for a in villager_agents]
            w_elos = [elos[a] for a in werewolf_agents]
            if v_elos and w_elos:
                avg_v_elo = np.mean(v_elos)
                avg_w_elo = np.mean(w_elos)

                # Calculate expected win probability for Villagers
                exponent = (avg_w_elo - avg_v_elo) / 400.0
                if exponent > 40:
                    exponent = 40.0
                elif exponent < -40:
                    exponent = -40.0
                expected_v = 1.0 / (1.0 + 10.0 ** exponent)

                result_v = 1 if game.winner_team == Team.VILLAGERS else 0

                # Total points to exchange (K=32)
                # This ensures zero-sum: what V gains, W loses.
                k = 32
                total_delta = k * (result_v - expected_v)

                # Distribute points
                # Each member gets equal share of the team's total gain/loss
                v_change = total_delta / len(villager_agents)
                w_change = -total_delta / len(werewolf_agents)

                for agent in villager_agents:
                    elos[agent] += v_change
                for agent in werewolf_agents:
                    elos[agent] += w_change
        return elos

    @staticmethod
    def _compute_openskill_ratings(games: List, model) -> Dict[str, object]:
        """Computes OpenSkill (TrueSkill-like) ratings."""
        if not model:
            return {}

        # Initialize ratings for all agents encountered
        current_ratings = {}

        for game in games:
            villager_agents = []
            werewolf_agents = []

            # Ensure all players have a rating object in our local scope
            for player in game.players:
                agent_name = player.agent.display_name
                if agent_name not in current_ratings:
                    current_ratings[agent_name] = model.rating(name=agent_name)

                if player.role.team == Team.VILLAGERS:
                    villager_agents.append(agent_name)
                else:
                    werewolf_agents.append(agent_name)

            team_v = [current_ratings[a] for a in villager_agents]
            team_w = [current_ratings[a] for a in werewolf_agents]

            teams = None
            if game.winner_team == Team.VILLAGERS:
                teams = [team_v, team_w]
            elif game.winner_team == Team.WEREWOLVES:
                teams = [team_w, team_v]

            if teams:
                # rate() returns updated rating objects in the same structure as input
                try:
                    new_ratings = model.rate(teams)
                    flat_ratings = [r for team in new_ratings for r in team]
                    for r in flat_ratings:
                        current_ratings[r.name] = r
                except Exception:
                    # Fallback for degenerate cases (e.g. empty teams)
                    pass

        return current_ratings

    @staticmethod
    def _compute_elo_ratings_fast(games_data: List[Tuple[Tuple[int], Tuple[int], int]], num_agents: int) -> List[float]:
        """Computes Elo ratings using integer indices for speed."""
        elos = [1200.0] * num_agents
        k = 32

        for v_indices, w_indices, result_v in games_data:
            if not v_indices or not w_indices:
                continue

            # Calculate team averages
            avg_v_elo = sum(elos[i] for i in v_indices) / len(v_indices)
            avg_w_elo = sum(elos[i] for i in w_indices) / len(w_indices)

            # Expected score for villagers
            exponent = (avg_w_elo - avg_v_elo) / 400.0
            if exponent > 40:
                exponent = 40.0
            elif exponent < -40:
                exponent = -40.0
            expected_v = 1.0 / (1.0 + 10.0 ** exponent)

            # Total points to exchange (K=32)
            # This ensures zero-sum: what V gains, W loses.
            total_delta = k * (result_v - expected_v)

            # Distribute points
            # Each member gets equal share of the team's total gain/loss
            v_change = total_delta / len(v_indices)
            w_change = -total_delta / len(w_indices)
            for i in v_indices:
                elos[i] += v_change
            for i in w_indices:
                elos[i] += w_change
        return elos

    @staticmethod
    def _compute_openskill_ratings_fast(
        games_data: List[Tuple[Tuple[int], Tuple[int], int]], num_agents: int, model
    ) -> List:
        """Computes OpenSkill ratings using integer indices and pre-allocated ratings."""
        # Create initial ratings for all agents (unnamed)
        ratings = [model.rating() for _ in range(num_agents)]

        for v_indices, w_indices, result_v in games_data:
            if not v_indices or not w_indices:
                continue

            # Construct teams from current rating state
            team_v = [ratings[i] for i in v_indices]
            team_w = [ratings[i] for i in w_indices]

            # Prepare for rate()
            # If win, order is [winner, loser].
            if result_v == 1:
                teams = [team_v, team_w]
            else:
                teams = [team_w, team_v]

            try:
                new_ratings = model.rate(teams)

                # Unpack and update state
                if result_v == 1:
                    new_team_v = new_ratings[0]
                    new_team_w = new_ratings[1]
                else:
                    new_team_w = new_ratings[0]
                    new_team_v = new_ratings[1]

                # Assign back to master list
                for i, r in zip(v_indices, new_team_v):
                    ratings[i] = r
                for i, r in zip(w_indices, new_team_w):
                    ratings[i] = r

            except Exception:
                pass

        return ratings

    def _generate_bootstrap_samples(
        self, light_games_iterator: Iterator[Union[LightGame, Tuple]], num_samples: int
    ) -> Iterator[List[Union[LightGame, Tuple]]]:
        light_games = list(light_games_iterator)
        if not light_games:
            return

        # Create an object array and fill it to prevent numpy from unpacking the tuples
        light_games_np = np.empty(len(light_games), dtype=object)
        light_games_np[:] = light_games

        rnd_master = np.random.default_rng(self.seed)

        for _ in range(num_samples):
            sampled_array = light_games_np[rnd_master.integers(0, len(light_games), size=len(light_games))]
            yield list(sampled_array)

    def _bootstrap_elo(self, num_samples=100):
        if not self.games or num_samples <= 0:
            return

        # Check cache
        cache_key = f"elo_{self.git_hash}_{self.input_hash}_{self.seed}_{num_samples}.pkl"
        cache_path = self.cache_dir / cache_key

        if cache_path.exists():
            try:
                print(f"Loading cached Elo bootstrap results from {cache_path}...")
                with open(cache_path, 'rb') as f:
                    agent_stds = pickle.load(f)
                for agent, std in agent_stds.items():
                    self.metrics[agent].elo_std = std
                return
            except Exception as e:
                print(f"Failed to load Elo cache: {e}. Recomputing...")

        # Pre-process data for speed
        all_agents = sorted(list(self.metrics.keys()))
        agent_to_idx = {name: i for i, name in enumerate(all_agents)}
        num_agents = len(all_agents)

        fast_games = []
        for g in self.games:
            v_indices = []
            w_indices = []
            for p in g.players:
                idx = agent_to_idx.get(p.agent.display_name)
                if idx is not None:
                    if p.role.team == Team.VILLAGERS:
                        v_indices.append(idx)
                    else:
                        w_indices.append(idx)

            result_v = 1 if g.winner_team == Team.VILLAGERS else 0
            if v_indices and w_indices:
                fast_games.append((tuple(v_indices), tuple(w_indices), result_v))

        # We pass the full list of games to _generate_bootstrap_samples
        samples_iterator = self._generate_bootstrap_samples(fast_games, num_samples)

        worker = functools.partial(GameSetEvaluator._compute_elo_ratings_fast, num_agents=num_agents)

        with ProcessPoolExecutor(
            max_workers=os.cpu_count(), mp_context=multiprocessing.get_context("spawn")
        ) as executor:
            results = list(
                tqdm(executor.map(worker, samples_iterator), total=num_samples, desc="Elo Bootstrap")
            )

        bootstrapped_elos = defaultdict(list)

        for sample_elos_list in results:
            for i, elo in enumerate(sample_elos_list):
                bootstrapped_elos[all_agents[i]].append(elo)

        agent_stds = {}
        for agent, values in bootstrapped_elos.items():
            if len(values) > 1:
                std = float(np.std(values, ddof=1))
                self.metrics[agent].elo_std = std
                agent_stds[agent] = std

        # Save to cache
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(agent_stds, f)
            print(f"Saved Elo bootstrap results to {cache_path}")
        except Exception as e:
            print(f"Failed to save Elo cache: {e}")

    def _bootstrap_openskill(self, num_samples=100):
        if not self.games or not self.openskill_model or num_samples <= 0:
            return

        # Check cache
        cache_key = f"openskill_{self.git_hash}_{self.input_hash}_{self.seed}_{num_samples}.pkl"
        cache_path = self.cache_dir / cache_key

        if cache_path.exists():
            try:
                print(f"Loading cached OpenSkill bootstrap results from {cache_path}...")
                with open(cache_path, 'rb') as f:
                    agent_stds = pickle.load(f)
                for agent, std in agent_stds.items():
                    self.metrics[agent].openskill_mu_std = std
                return
            except Exception as e:
                print(f"Failed to load OpenSkill cache: {e}. Recomputing...")

        # Pre-process data for speed
        all_agents = sorted(list(self.metrics.keys()))
        agent_to_idx = {name: i for i, name in enumerate(all_agents)}
        num_agents = len(all_agents)

        fast_games = []
        for g in self.games:
            v_indices = []
            w_indices = []
            for p in g.players:
                idx = agent_to_idx.get(p.agent.display_name)
                if idx is not None:
                    if p.role.team == Team.VILLAGERS:
                        v_indices.append(idx)
                    else:
                        w_indices.append(idx)

            result_v = 1 if g.winner_team == Team.VILLAGERS else 0
            if v_indices and w_indices:
                fast_games.append((tuple(v_indices), tuple(w_indices), result_v))

        # We pass the full list of games to _generate_bootstrap_samples
        samples_iterator = self._generate_bootstrap_samples(fast_games, num_samples)

        worker = functools.partial(_openskill_bootstrap_worker_fast, num_agents=num_agents, model=self.openskill_model)

        with ProcessPoolExecutor(
            max_workers=os.cpu_count(), mp_context=multiprocessing.get_context("spawn")
        ) as executor:
            results = list(tqdm(executor.map(worker, samples_iterator), total=num_samples, desc="OpenSkill Bootstrap"))

        bootstrapped_mus = defaultdict(list)
        default_mu = self.openskill_model.rating().mu

        for sample_ratings_list in results:
            for i, mu in enumerate(sample_ratings_list):
                # _openskill_bootstrap_worker_fast returns list of mus
                bootstrapped_mus[all_agents[i]].append(mu)

        agent_stds = {}
        for agent, values in bootstrapped_mus.items():
            if len(values) > 1:
                std = float(np.std(values, ddof=1))
                self.metrics[agent].openskill_mu_std = std
                agent_stds[agent] = std

        # Save to cache
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(agent_stds, f)
            print(f"Saved OpenSkill bootstrap results to {cache_path}")
        except Exception as e:
            print(f"Failed to save OpenSkill cache: {e}")

    def evaluate(self, gte_samples=3, elo_samples=3, openskill_samples=3):
        self.gte_samples = gte_samples
        self.elo_samples = elo_samples
        self.openskill_samples = openskill_samples
        light_gte_games = []
        for game in tqdm(self.games, desc="Evaluating Games"):
            villagers_won = game.winner_team == Team.VILLAGERS
            # Capture costs if available from GameResult
            player_costs = getattr(game, "player_costs", {})
            player_tokens = getattr(game, "player_tokens", {})
            player_prompt_tokens = getattr(game, "player_prompt_tokens", {})
            player_completion_tokens = getattr(game, "player_completion_tokens", {})
            player_durations = getattr(game, "player_durations", {})

            # --- Calculate dominance metrics for the winning team ---
            margin_of_win = 0.0
            speed_of_win = 0.0
            if game.winner_team is not None:
                # Margin of win
                winning_team_players = [p for p in game.players if p.role.team == game.winner_team]
                if winning_team_players:
                    total_team_size = len(winning_team_players)
                    living_teammates = sum(1 for p in winning_team_players if p.alive)
                    margin_of_win = living_teammates / total_team_size

                # Speed of win
                game_duration = 0
                if player_durations:
                    game_duration = max(player_durations.values())
                if game_duration > 0:
                    speed_of_win = 1.0 / game_duration

            for player in game.players:
                agent_name = player.agent.display_name
                if self.metrics[agent_name].agent_name is None:
                    self.metrics[agent_name].set_agent_name(agent_name)

                won = 1 if player.role.team == game.winner_team else 0
                survived = 1 if player.alive else 0

                self.metrics[agent_name].wins.append(won)
                self.metrics[agent_name].wins_by_role[player.role.name].append(won)
                self.metrics[agent_name].survival_scores.append(survived)
                self.metrics[agent_name].survival_by_role[player.role.name].append(survived)

                if won:
                    self.metrics[agent_name].margin_of_win_scores.append(margin_of_win)
                    self.metrics[agent_name].speed_of_win_scores.append(speed_of_win)

                # WD-KSR is the joint probability of winning AND surviving
                wd_ksr_score = 1 if won and survived else 0
                self.metrics[agent_name].wd_survival_scores.append(wd_ksr_score)
                self.metrics[agent_name].wd_survival_by_role[player.role.name].append(wd_ksr_score)

                # Record costs
                if player.id in player_costs:
                    self.metrics[agent_name].total_costs.append(player_costs[player.id])
                    self.metrics[agent_name].total_tokens.append(player_tokens.get(player.id, 0))
                    self.metrics[agent_name].total_prompt_tokens.append(player_prompt_tokens.get(player.id, 0))
                    self.metrics[agent_name].total_completion_tokens.append(player_completion_tokens.get(player.id, 0))
                    self.metrics[agent_name].durations.append(player_durations.get(player.id, 0))

            irp_results, vss_results = game.iterate_voting_mini_game()
            for agent_name, score in irp_results:
                self.metrics[agent_name].irp_scores.append(score)
                # WD-IRP is the joint probability of a correct vote AND winning
                self.metrics[agent_name].wd_irp_scores.append(score if villagers_won else 0)
            for agent_name, score in vss_results:
                self.metrics[agent_name].vss_scores.append(score)
                # WD-VSS is the joint probability of a correct vote AND winning
                self.metrics[agent_name].wd_vss_scores.append(score if villagers_won else 0)

            # Prepare GTE light game object using the already extracted data (only if needed)
            if gte_samples > 0:
                light_gte_games.append(LightGame(game.players, game.winner_team, irp_results, vss_results, player_durations))

        # Elo
        final_elos = self._compute_elo_ratings(self.games)
        for agent, rating in final_elos.items():
            self.metrics[agent].elo = rating

        # OpenSkill
        if self.openskill_model:
            final_openskill_ratings = self._compute_openskill_ratings(self.games, self.openskill_model)
            for agent, rating in final_openskill_ratings.items():
                self.metrics[agent].openskill_rating = rating

        if elo_samples > 0:
            self._bootstrap_elo(num_samples=elo_samples)
        if openskill_samples > 0:
            self._bootstrap_openskill(num_samples=openskill_samples)
        if gte_samples > 0:
            self._run_gte_evaluation(num_samples=gte_samples, light_gte_games=light_gte_games)

    def _precompute_gte_tensor(self, light_games, agents, tasks):
        """Pre-computes scores for all games into a tensor."""
        n_games = len(light_games)
        n_agents = len(agents)
        n_tasks = len(tasks)

        agent_idx_map = {a: i for i, a in enumerate(agents)}
        task_idx_map = {t: i for i, t in enumerate(tasks)}
        task_set = set(tasks)

        # Shape: (games, agents, tasks)
        tensor = np.full((n_games, n_agents, n_tasks), np.nan, dtype=np.float32)

        for g_i, game in enumerate(light_games):
            # --- Calculate dominance metrics for the winning team ---
            margin_of_win = 0.0
            speed_of_win = 0.0
            if game.winner_team is not None:
                # Margin of win
                winning_team_players = [p for p in game.players if p.role.team == game.winner_team]
                if winning_team_players:
                    total_team_size = len(winning_team_players)
                    living_teammates = sum(1 for p in winning_team_players if p.alive)
                    margin_of_win = living_teammates / total_team_size

                # Speed of win
                game_duration = 0
                if game.player_durations:
                    game_duration = max(game.player_durations.values())
                if game_duration > 0:
                    speed_of_win = 1.0 / game_duration

            for player in game.players:
                agent_name = player.agent.display_name
                a_i = agent_idx_map.get(agent_name)
                if a_i is None:
                    continue

                role_name = player.role.name
                won = player.role.team == game.winner_team
                alive = 1 if player.alive else 0

                # Check metrics that apply to this player

                # WinRate
                win_rate_task = f'WinRate-{role_name}'
                if win_rate_task in task_idx_map:
                    tensor[g_i, a_i, task_idx_map[win_rate_task]] = 1 if won else 0

                # KSR
                if 'KSR' in task_idx_map:
                    tensor[g_i, a_i, task_idx_map['KSR']] = alive
                ksr_task = f'KSR-{role_name}'
                if ksr_task in task_idx_map:
                    tensor[g_i, a_i, task_idx_map[ksr_task]] = alive

                # WD-KSR
                wd_ksr_score = 1 if won and alive else 0
                if 'WD-KSR' in task_idx_map:
                    tensor[g_i, a_i, task_idx_map['WD-KSR']] = wd_ksr_score
                wd_ksr_task = f'WD-KSR-{role_name}'
                if wd_ksr_task in task_idx_map:
                    tensor[g_i, a_i, task_idx_map[wd_ksr_task]] = wd_ksr_score

                if won:
                    if 'Margin of Win' in task_idx_map:
                        tensor[g_i, a_i, task_idx_map['Margin of Win']] = margin_of_win
                    if 'Speed of Win' in task_idx_map:
                        tensor[g_i, a_i, task_idx_map['Speed of Win']] = speed_of_win

            # IRP & VSS
            villagers_won = game.winner_team == Team.VILLAGERS
            irp_results, vss_results = game.irp_results, game.vss_results

            for agent_name, score in irp_results:
                a_i = agent_idx_map.get(agent_name)
                if a_i is None: continue
                if 'IRP' in task_idx_map:
                    tensor[g_i, a_i, task_idx_map['IRP']] = score
                if 'WD-IRP' in task_idx_map:
                    tensor[g_i, a_i, task_idx_map['WD-IRP']] = score if villagers_won else 0

            for agent_name, score in vss_results:
                a_i = agent_idx_map.get(agent_name)
                if a_i is None: continue
                if 'VSS' in task_idx_map:
                    tensor[g_i, a_i, task_idx_map['VSS']] = score
                if 'WD-VSS' in task_idx_map:
                    tensor[g_i, a_i, task_idx_map['WD-VSS']] = score if villagers_won else 0

        return tensor

    def _run_gte_evaluation(self, num_samples: int, light_gte_games: List[LightGame] = None):
        # Filter agents to only those who actually played in the loaded games
        # This prevents "phantom" agents (from defaultdict pollution) from breaking GTE
        active_agents = set()
        for g in self.games:
            for p in g.players:
                active_agents.add(p.agent.display_name)
        
        all_metrics_agents = sorted(list(self.metrics.keys()))
        agents = [a for a in all_metrics_agents if a in active_agents]
        
        if len(agents) < len(all_metrics_agents):
             print(f"Filtered {len(all_metrics_agents) - len(agents)} phantom agents from GTE (No games played).")
        
        if not agents:
            print("No active agents found in games. Skipping GTE.")
            return

        # Check cache
        # Tasks are part of the cache key
        if isinstance(self.gte_tasks, list):
            tasks_str = "_".join(sorted(self.gte_tasks))
        else:
            tasks_str = str(self.gte_tasks)

        # Hash the tasks string to avoid long filenames
        tasks_hash = hashlib.md5(tasks_str.encode()).hexdigest()

        cache_key = f"gte_{self.git_hash}_{self.input_hash}_{self.seed}_{num_samples}_{tasks_hash}.pkl"
        cache_path = self.cache_dir / cache_key

        if cache_path.exists():
            try:
                print(f"Loading cached GTE bootstrap results from {cache_path}...")
                with open(cache_path, 'rb') as f:
                    cache_data = pickle.load(f)

                self.gte_ratings = cache_data['gte_ratings']
                
                # Robust validation against current tasks to prevent stale cache crashes
                # self.gte_ratings should be (ratings_mean, ratings_std)
                # ratings_mean should be [mean_agent_ratings, mean_task_ratings]
                try:
                    # Player 0 = Tasks, Player 1 = Agents
                    agent_ratings_mean = self.gte_ratings[0][1]
                    task_ratings_mean = self.gte_ratings[0][0]
                    print(f"[GTE Cache Load] Agents in config: {len(agents)}, Agents in cache: {len(agent_ratings_mean)}")
                    print(f"[GTE Cache Load] Tasks in config: {len(self.gte_tasks)}, Tasks in cache: {len(task_ratings_mean)}")
                    
                    if len(agent_ratings_mean) != len(agents) or len(task_ratings_mean) != len(self.gte_tasks):
                        print(f"!!! GTE Cache Size Mismatch: Agents({len(agents)} vs {len(agent_ratings_mean)}) or Tasks({len(self.gte_tasks)} vs {len(task_ratings_mean)}). Recomputing...")
                        raise ValueError("Cache size mismatch")
                except (IndexError, TypeError) as e:
                    print(f"!!! GTE Cache structure mismatch: {e}. Recomputing...")
                    raise ValueError("Cache structure mismatch")

                self.gte_joint = cache_data.get('gte_joint')
                self.gte_marginals = cache_data.get('gte_marginals')
                self.gte_contributions_raw = cache_data.get('gte_contributions_raw')
                self.gte_metric_contributions_raw = cache_data.get('gte_metric_contributions_raw')

                agent_metrics_cache = cache_data['agent_metrics']
                for agent_name, m_cache in agent_metrics_cache.items():
                    self.metrics[agent_name].gte_rating = m_cache['gte_rating']
                    self.metrics[agent_name].gte_contributions = m_cache['gte_contributions']

                # Reconstruct gte_game structure for plotting if not in cache (or even if it is, for safety)
                # It just needs actions [tasks, agents] (Player 0=Tasks, Player 1=Agents)
                agents = sorted(list(self.metrics.keys()))
                self.gte_game = SimpleNamespace(actions=[self.gte_tasks, agents])

                return
            except Exception as e:
                print(f"Failed to load GTE cache: {e}. Recomputing...")

        light_gte_games = []
        for g in tqdm(self.games, desc="Preparing GTE data"):
            irp_results, vss_results = g.iterate_voting_mini_game()
            players = [
                Player(p.id, Agent(p.agent.display_name), Role(p.role.name, p.role.team), p.alive)
                for p in g.players
            ]
            player_durations = getattr(g, "player_durations", {})
            light_gte_games.append(LightGame(players, g.winner_team, irp_results, vss_results, player_durations))

        # Build tensor
        tensor = self._precompute_gte_tensor(light_gte_games, agents, self.gte_tasks)

        # Print Input Metrics Summary for transparency
        mean_mtrx, std_mtrx, sem_mtrx = _compute_mean_std_sem(tensor, axis=0)
        print("\nGTE Input Metrics Summary (Mean ± StdDev):")
        header = f"{'Agent':<20}"
        for task in self.gte_tasks:
            header += f" | {task:<25}"
        print(header)
        print("-" * len(header))
        for i, agent in enumerate(agents):
            agent_display = agent if agent else f"Agent_{i}"
            row = f"{agent_display:<20}"
            for j, task in enumerate(self.gte_tasks):
                val = mean_mtrx[i, j]
                err = std_mtrx[i, j]
                if np.isnan(val):
                    row += f" | {'N/A':<25}"
                else:
                    row += f" | {val:.4f} ± {err:.4f}"
            print(row)
        print("-" * len(header))

        # Use indices for bootstrap
        n_games_total = len(light_gte_games)
        n_agents = len(agents)
        n_tasks = len(self.gte_tasks)
        rnd_master = np.random.default_rng(self.seed)

        # Chunked Vectorized Matrix Calculation
        # We calculate means and stds in chunks to avoid memory explosion (4D tensor)
        # while still benefiting from vectorized NumPy speed.
        all_means = []
        all_stds = []

        # Target ~1GB per chunk (4 bytes per float32)
        chunk_size = max(1, int(2.5e8 / (n_games_total * n_agents * n_tasks + 1)))
        chunk_size = min(chunk_size, num_samples)

        num_chunks = (num_samples + chunk_size - 1) // chunk_size
        print(f"Calculating GTE matrices in {num_chunks} vectorized chunks (chunk_size={chunk_size})...")

        for c in tqdm(range(num_chunks), desc="Vectorizing Math"):
            start_s = c * chunk_size
            end_s = min((c + 1) * chunk_size, num_samples)
            current_chunk_size = end_s - start_s

            # Generate indices for this chunk: (chunk_size, n_games_total)
            chunk_indices = rnd_master.integers(0, n_games_total, size=(current_chunk_size, n_games_total))

            # Sample the tensor: (chunk_size, n_games_total, n_agents, n_tasks)
            # This is the memory-intensive part, but limited by chunk_size
            sample_tensor = tensor[chunk_indices]

            # Compute means and stds for the whole chunk at once
            # Axis 1 is 'games_total'
            means, stds, sems = _compute_mean_std_sem(sample_tensor, axis=1) # (chunk_size, n_agents, n_tasks)
            all_means.append(means)
            all_stds.append(stds)

        all_means = np.concatenate(all_means, axis=0)
        all_stds = np.concatenate(all_stds, axis=0)

        # Solver Pass: Using spawn multiprocessing to bypass GIL and avoid JAX deadlocks.
        # Minimal data (small mean/std matrices) is passed to workers.
        print("\nStarting GTE solver with spawn multiprocessing...")
        worker_func = functools.partial(_gte_bootstrap_worker_fast, agents=agents, tasks=self.gte_tasks)
        matrices_iterator = list(zip(all_means, all_stds))
        
        # Use spawn context to avoid inheriting JAX/XLA state/locks from parent process
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=os.cpu_count()) as pool:
            res = list(tqdm(
                pool.imap(worker_func, matrices_iterator), 
                total=num_samples, 
                desc="GTE Bootstrap (Solver)"
            ))

        # Unpack
        ratings, joints, marginals, r2m_contributions, m2r_contributions, games = zip(*res)
        self.gte_game = games[0]  # Save one game instance for structure

        # --- CRITICAL FIX: Update agents list to match what the solver actually used ---
        # If the solver filtered out agents (e.g. all NaNs), we must update our local 'agents' list
        # to match the returned ratings dimensions.
        if self.gte_game and getattr(self.gte_game, 'actions', None) and len(self.gte_game.actions) > 1:
            # Player 0 = Tasks, Player 1 = Agents
            solver_agents = self.gte_game.actions[1]
            if len(solver_agents) != len(agents):
                print(f"!!! Solver dropped {len(agents) - len(solver_agents)} agents (insufficient data). Syncing agent list...")
                print(f"    Original: {len(agents)} -> Solver: {len(solver_agents)}")
                agents = solver_agents

        ratings_mean = [np.mean(r, axis=0) for r in zip(*ratings)]
        ratings_std = [np.std(r, axis=0) for r in zip(*ratings)]

        joints_mean = np.mean(joints, axis=0)

        marginals_by_dim = list(zip(*marginals))
        marginals_mean = [np.mean(m, axis=0) for m in marginals_by_dim]
        marginals_std = [np.std(m, axis=0) for m in marginals_by_dim]

        # Print GTE Results Summary (Nash Equilibrium Win Rates)
        print("\nGTE Bootstrap Results (Nash Equilibrium Win Rate ± StdDev):")
        header = f"{'Agent':<20} | {'GTE Win Rate':<25}"
        print(header)
        print("-" * len(header))
                # ratings_mean[1] is the Nash win rate (index 1 is the player dimension in this case?)
        # Actually it depends on how many players. In Werewolf it's usually 2 perspectives.
        # But let's look at how gte_rating is stored below.
        # rating_val = (ratings_mean[1][i], ratings_std[1][i])
        for i, agent in enumerate(agents):
            agent_display = agent if agent else f"Agent_{i}"
            if len(ratings_mean) > 1:
                # Safety check for index out of bounds
                if i < len(ratings_mean[1]):
                    val = ratings_mean[1][i]
                    err = ratings_std[1][i]
                    print(f"{agent_display:<20} | {val:.8f} ± {err:.8f}")
                else:
                    print(f"{agent_display:<20} | Error: Index {i} out of bounds for ratings size {len(ratings_mean[1])}")
            else:
                print(f"{agent_display:<20} | GTE analysis failed (insufficient dimensions)")
        print("-" * len(header))

        r2m_contributions_mean = np.mean(r2m_contributions, axis=0)
        r2m_contributions_std = np.std(r2m_contributions, axis=0)

        m2r_contributions_mean = np.mean(m2r_contributions, axis=0)
        m2r_contributions_std = np.std(m2r_contributions, axis=0)

        self.gte_ratings = (ratings_mean, ratings_std)
        self.gte_joint = (joints_mean, None)
        self.gte_marginals = (marginals_mean, marginals_std)
        self.gte_contributions_raw = (r2m_contributions_mean, r2m_contributions_std)
        self.gte_metric_contributions_raw = (m2r_contributions_mean, m2r_contributions_std)

        agent_metrics_cache = {}

        for i, agent_name in enumerate(agents):
            if len(ratings_mean) > 1:
                rating_val = (ratings_mean[1][i], ratings_std[1][i])
            else:
                print(f"!!! WARNING: GTE ratings_mean len <= 1 ({len(ratings_mean)}). Defaulting {agent_name} to 0.0")
                rating_val = (0.0, 0.0)
            
            self.metrics[agent_name].gte_rating = rating_val

            contrib_map = {}
            for j, task_name in enumerate(self.gte_tasks):
                if r2m_contributions_mean.shape[0] > i:
                    val = (r2m_contributions_mean[i, j], r2m_contributions_std[i, j])
                else:
                    val = (0.0, 0.0)
                self.metrics[agent_name].gte_contributions[task_name] = val
                contrib_map[task_name] = val

            agent_metrics_cache[agent_name] = {"gte_rating": rating_val, "gte_contributions": contrib_map}

        # Save to cache
        try:
            cache_data = {
                "gte_ratings": self.gte_ratings,
                "gte_joint": self.gte_joint,
                "gte_marginals": self.gte_marginals,
                "gte_contributions_raw": self.gte_contributions_raw,
                "gte_metric_contributions_raw": self.gte_metric_contributions_raw,
                "agent_metrics": agent_metrics_cache,
            }
            with open(cache_path, "wb") as f:
                pickle.dump(cache_data, f)
            print(f"Saved GTE bootstrap results to {cache_path}")
        except Exception as e:
            print(f"Failed to save GTE cache: {e}")

    def print_results(self):
        sorted_items = sorted(self.metrics.items(), key=lambda item: item[0])
        for agent_name, stats in sorted_items:
            print(f"Agent: {agent_name}")
            win_rate, win_std = stats.get_win_rate()
            print(f"  Overall Win Rate: {win_rate:.2f} ± {win_std * 1.96:.2f} (CI95) ({len(stats.wins)} games)")
            ksr, ksr_std = stats.get_ksr()
            print(f"  Overall Survival Rate (KSR): {ksr:.2f} ± {ksr_std * 1.96:.2f} (CI95)")
            wd_ksr, wd_ksr_std = stats.get_wd_ksr()
            print(f"  Win-Dependent Survival Rate (WD-KSR): {wd_ksr:.2f} ± {wd_ksr_std * 1.96:.2f} (CI95)")

            print("  Role-Specific Win Rates:")
            for role in sorted(stats.wins_by_role.keys()):
                role_rate, role_std = stats.get_win_rate_for_role(role)
                game_count = len(stats.wins_by_role[role])
                print(f"    {role:<10}: {role_rate:.2f} ± {role_std * 1.96:.2f} (CI95) ({game_count} games)")

            print("  Voting Accuracy (Villager Team):")
            irp, irp_std = stats.get_irp()
            vss, vss_std = stats.get_vss()
            print(
                f"    IRP (Identification Precision): {irp:.2f} ± {irp_std * 1.96:.2f} (CI95) ({len(stats.irp_scores)} votes)"
            )
            print(
                f"    VSS (Voting Success Score):     {vss:.2f} ± {vss_std * 1.96:.2f} (CI95) ({len(stats.vss_scores)} votes)"
            )

            print("  Win-Dependent Voting Accuracy (Villager Team):")
            wd_irp, wd_irp_std = stats.get_wd_irp()
            wd_vss, wd_vss_std = stats.get_wd_vss()
            print(f"    WD-IRP: {wd_irp:.2f} ± {wd_irp_std * 1.96:.2f} (CI95)")
            print(f"    WD-VSS: {wd_vss:.2f} ± {wd_vss_std * 1.96:.2f} (CI95)")

            print("  Dominance Metrics (in winning games):")
            margin_of_win, margin_of_win_std = stats.get_margin_of_win()
            speed_of_win, speed_of_win_std = stats.get_speed_of_win()
            n_wins = len(stats.margin_of_win_scores)
            print(f"    Margin of Win: {margin_of_win:.2f} ± {margin_of_win_std * 1.96:.2f} (CI95) ({n_wins} wins)")
            print(f"    Speed of Win: {speed_of_win:.2f} ± {speed_of_win_std * 1.96:.2f} (CI95) ({n_wins} wins)")

            # Cost Stats
            avg_cost, cost_sem = stats.get_avg_cost()
            avg_tokens, tokens_sem = stats.get_avg_tokens()
            avg_cost_per_turn, cost_per_turn_sem = stats.get_avg_cost_per_turn()
            avg_input_tokens_per_turn, input_tokens_per_turn_sem = stats.get_avg_input_tokens_per_turn()
            avg_output_tokens_per_turn, output_tokens_per_turn_sem = stats.get_avg_output_tokens_per_turn()
            if avg_cost > 0:
                print("  Cost Metrics:")
                print(f"    Avg Cost/Game: ${avg_cost:.4f} ± {cost_sem * 1.96:.4f}")
                print(f"    Avg Cost/Turn: ${avg_cost_per_turn:.4f} ± {cost_per_turn_sem * 1.96:.4f}")
                print(f"    Avg Tokens/Game: {avg_tokens:.1f} ± {tokens_sem * 1.96:.1f}")
                print(f"    Avg Input Tokens/Turn: {avg_input_tokens_per_turn:.1f} ± {input_tokens_per_turn_sem * 1.96:.1f}")
                print(f"    Avg Output Tokens/Turn: {avg_output_tokens_per_turn:.1f} ± {output_tokens_per_turn_sem * 1.96:.1f}")

            print("  Ratings:")
            print(f"    Elo: {stats.elo:.2f} ± {stats.elo_std * 1.96:.2f} (CI95)")
            if stats.openskill_rating:
                print(
                    f"    TrueSkill: mu={stats.openskill_rating.mu:.2f} ± {stats.openskill_mu_std * 1.96:.2f} (CI95), sigma={stats.openskill_rating.sigma:.2f}"
                )
            print("  Game Theoretic Evaluation (GTE):")
            gte_mean, gte_std = getattr(stats, 'gte_rating', (0.0, 0.0))
            print(f"    Overall GTE Rating: {gte_mean:.2f} ± {gte_std * 1.96:.2f} (CI95)")
            
            # Print individual task contributions with safety
            for task in self.gte_tasks:
                contrib = stats.gte_contributions.get(task, (0.0, 0.0))
                contrib_mean, contrib_std = contrib
                print(f"    - {task:<30} Contribution: {contrib_mean:.2f} ± {contrib_std * 1.96:.2f} (CI95)")
            print("-" * 30)

        # Global GTE Metric Importance (moved OUTSIDE agent loop)
        if getattr(self, 'gte_ratings', None):
            print("\n  GTE Metric Importance (Ratings):")
            try:
                metric_ratings_mean = self.gte_ratings[0][1]
                metric_ratings_std = self.gte_ratings[1][1]

                if len(metric_ratings_mean) != len(self.gte_tasks):
                    print(f"    !!! WARNING: Mismatch in GTE task ratings size! Expected {len(self.gte_tasks)}, got {len(metric_ratings_mean)}")
                    # Truncate or pad to avoid crash, though it indicates a deeper issue
                    limit = min(len(metric_ratings_mean), len(self.gte_tasks))
                else:
                    limit = len(self.gte_tasks)

                metric_ratings = []
                for i in range(limit):
                    try:
                        task_name = self.gte_tasks[i]
                        metric_ratings.append((task_name, metric_ratings_mean[i], metric_ratings_std[i]))
                    except IndexError as e:
                        # This should be caught by the 'limit' check but added for absolute safety
                        print(f"    !!! DIAGNOSTIC ERROR: IndexError at index {i}. Tasks({len(self.gte_tasks)}), Mean({len(metric_ratings_mean)}), Std({len(metric_ratings_std)})")
                        break

                # Sort by rating descending
                sorted_metric_ratings = sorted(metric_ratings, key=lambda x: x[1], reverse=True)

                for task, rating_mean, rating_std in sorted_metric_ratings:
                    print(f"    - {task:<30} Rating: {rating_mean:.2f} ± {rating_std * 1.96:.2f} (CI95)")
            except Exception as e:
                print(f"    !!! ERROR printing GTE Metric Importance: {e}")
        print("=" * 50)

    def _prepare_plot_data(self):
        plot_data = []
        for agent_name, metrics in self.metrics.items():
            # 1. Overall
            win_rate, win_std = metrics.get_win_rate()
            ksr, ksr_std = metrics.get_ksr()
            plot_data.extend(
                [
                    {
                        "agent": agent_name,
                        "metric": "Win Rate",
                        "value": win_rate,
                        "CI95": win_std * 1.96,
                        "category": "Overall",
                    },
                    {"agent": agent_name, "metric": "KSR", "value": ksr, "CI95": ksr_std * 1.96, "category": "Overall"},
                ]
            )

            # 2. Voting
            irp, irp_std = metrics.get_irp()
            vss, vss_std = metrics.get_vss()
            plot_data.extend(
                [
                    {
                        "agent": agent_name,
                        "metric": "IRP",
                        "value": irp,
                        "CI95": irp_std * 1.96,
                        "category": "Voting Accuracy",
                    },
                    {
                        "agent": agent_name,
                        "metric": "VSS",
                        "value": vss,
                        "CI95": vss_std * 1.96,
                        "category": "Voting Accuracy",
                    },
                ]
            )

            # 3. Win-Dependent Metrics
            wd_ksr, wd_ksr_std = metrics.get_wd_ksr()
            wd_irp, wd_irp_std = metrics.get_wd_irp()
            wd_vss, wd_vss_std = metrics.get_wd_vss()
            plot_data.extend(
                [
                    {
                        "agent": agent_name,
                        "metric": "WD-KSR",
                        "value": wd_ksr,
                        "CI95": wd_ksr_std * 1.96,
                        "category": "Win-Dependent Metrics",
                    },
                    {
                        "agent": agent_name,
                        "metric": "WD-IRP",
                        "value": wd_irp,
                        "CI95": wd_irp_std * 1.96,
                        "category": "Win-Dependent Metrics",
                    },
                    {
                        "agent": agent_name,
                        "metric": "WD-VSS",
                        "value": wd_vss,
                        "CI95": wd_vss_std * 1.96,
                        "category": "Win-Dependent Metrics",
                    },
                ]
            )

            # 4. Dominance Metrics
            margin_of_win, margin_of_win_std = metrics.get_margin_of_win()
            speed_of_win, speed_of_win_std = metrics.get_speed_of_win()
            plot_data.extend(
                [
                    {
                        "agent": agent_name,
                        "metric": "Margin of Win",
                        "value": margin_of_win,
                        "CI95": margin_of_win_std * 1.96,
                        "category": "Dominance Metrics",
                    },
                    {
                        "agent": agent_name,
                        "metric": "Speed of Win",
                        "value": speed_of_win,
                        "CI95": speed_of_win_std * 1.96,
                        "category": "Dominance Metrics",
                    },
                ]
            )

            # 5. Role Specific Win Rates
            for role in sorted(metrics.wins_by_role.keys()):
                role_rate, role_std = metrics.get_win_rate_for_role(role)
                plot_data.append(
                    {
                        "agent": agent_name,
                        "metric": f"{role}",
                        "value": role_rate,
                        "CI95": role_std * 1.96,
                        "category": "Role-Specific Win Rate",
                    }
                )

            # 6. Role Specific Survival
            for role in sorted(metrics.survival_by_role.keys()):
                role_ksr, role_ksr_std = metrics.get_ksr_for_role(role)
                plot_data.append(
                    {
                        "agent": agent_name,
                        "metric": f"{role}",
                        "value": role_ksr,
                        "CI95": role_ksr_std * 1.96,
                        "category": "Role-Specific KSR",
                    }
                )

            # 7. Win-Dependent Role Specific KSR
            for role in sorted(metrics.wd_survival_by_role.keys()):
                role_ksr, role_ksr_std = metrics.get_wd_ksr_for_role(role)
                plot_data.append(
                    {
                        "agent": agent_name,
                        "metric": f"{role}",
                        "value": role_ksr,
                        "CI95": role_ksr_std * 1.96,
                        "category": "Win-Dependent KSR",
                    }
                )

            # 8. Cost Metrics (New)
            avg_cost_per_turn, cost_per_turn_sem = metrics.get_avg_cost_per_turn()
            avg_input_tokens_per_turn, input_tokens_per_turn_sem = metrics.get_avg_input_tokens_per_turn()
            avg_output_tokens_per_turn, output_tokens_per_turn_sem = metrics.get_avg_output_tokens_per_turn()
            plot_data.extend([
                {
                    "agent": agent_name,
                    "metric": "Avg Cost/Turn",
                    "value": avg_cost_per_turn,
                    "CI95": cost_per_turn_sem * 1.96,
                    "category": "Cost Metrics",
                },
                {
                    "agent": agent_name,
                    "metric": "Avg Input Tokens/Turn",
                    "value": avg_input_tokens_per_turn,
                    "CI95": input_tokens_per_turn_sem * 1.96,
                    "category": "Cost Metrics",
                },
                {
                    "agent": agent_name,
                    "metric": "Avg Output Tokens/Turn",
                    "value": avg_output_tokens_per_turn,
                    "CI95": output_tokens_per_turn_sem * 1.96,
                    "category": "Cost Metrics",
                }
            ])

            # 9. Ratings
            plot_data.append(
                {
                    "agent": agent_name,
                    "metric": "Elo",
                    "value": metrics.elo,
                    "CI95": metrics.elo_std * 1.96,
                    "category": "Ratings",
                }
            )

            if metrics.openskill_rating:
                plot_data.append(
                    {
                        "agent": agent_name,
                        "metric": "OpenSkill",
                        "value": metrics.openskill_rating.mu,
                        "CI95": metrics.openskill_mu_std * 1.96,
                        "category": "Ratings",
                    }
                )


                gte_mean, gte_std = metrics.gte_rating
                plot_data.append(
                    {
                        "agent": agent_name,
                        "metric": "GTE Rating",
                        "value": gte_mean,
                        "CI95": gte_std * 1.96,
                        "category": "Ratings",
                    }
                )
                
                for task, (contrib_mean, contrib_std) in metrics.gte_contributions.items():
                    plot_data.append(
                        {
                            "agent": agent_name,
                            "metric": f"GTE Contrib: {task}",
                            "value": contrib_mean,
                            "CI95": contrib_std * 1.96,
                            "category": "GTE Contributions",
                        }
                    )

            # 9. Cost
            avg_cost, cost_sem = metrics.get_avg_cost()
            if avg_cost > 0:
                plot_data.append(
                    {
                        "agent": agent_name,
                        "metric": "Avg Cost/Game",
                        "value": avg_cost,
                        "CI95": cost_sem * 1.96,
                        "category": "Cost",
                    }
                )

            avg_cost_turn, cost_turn_sem = metrics.get_avg_cost_per_turn()
            if avg_cost_turn > 0:
                plot_data.append(
                    {
                        "agent": agent_name,
                        "metric": "Avg Cost/Turn",
                        "value": avg_cost_turn,
                        "CI95": cost_turn_sem * 1.96,
                        "category": "Cost",
                    }
                )



        # 11. GTE Metric Importance (System Level)
        if self.gte_ratings and len(self.gte_ratings[0]) > 0:
            metric_ratings_mean = self.gte_ratings[0][0]
            metric_ratings_std = self.gte_ratings[1][0]
            for i, task_name in enumerate(self.gte_tasks):
                if i < len(metric_ratings_mean):
                    plot_data.append({
                        'agent': 'System', 'metric': f'Importance: {task_name}', 'value': metric_ratings_mean[i],
                        'CI95': metric_ratings_std[i] * 1.96, 'category': 'GTE Metric Importance'
                    })

        return pd.DataFrame(plot_data)

    def export_to_csv(self, output_path: str):
        """Exports detailed results to a CSV file in wide format with delta columns."""
        df = self._prepare_plot_data()
        if df.empty:
            print("No data to export to CSV.")
            return

        # Pre-calculate CI bounds
        # Lower delta should be negative (Value - CI_Lower) * -1? 
        # User requested: "lower delta should be a negative number"
        # Since CI95 is the half-width:
        # LowerCI = Value - CI95
        # LowerDelta = LowerCI - Value = -CI95
        # UpperDelta = UpperCI - Value = +CI95
        
        # However, usually Delta implies (Value - Ref).
        # User said: "lower delta should be a negative number"
        # So it is -(CI95).
        
        df['Values'] = df['value']
        df['Upper Delta'] = df['CI95']
        df['Lower Delta'] = -df['CI95']
        
        # Pivot the table to wide format
        # Index: Agent
        # Columns: Metric -> (Values, Lower Delta, Upper Delta)
        pivot_df = df.pivot(index='agent', columns='metric', values=['Values', 'Lower Delta', 'Upper Delta'])
        
        # Swap levels so Metric is top level: Metric -> (Values, Lower, Upper)
        pivot_df = pivot_df.swaplevel(0, 1, axis=1)
        pivot_df = pivot_df.sort_index(axis=1)
        
        # Flatten columns
        # Format: "{Metric}", "{Metric} 95 CI lower delta", "{Metric} 95 CI upper delta"
        flat_columns = []
        # We need to preserve the Metric grouping
        # Get unique metrics from the columns
        metrics = sorted(set(pivot_df.columns.get_level_values(0)))
        
        final_df = pd.DataFrame(index=pivot_df.index)
        
        # Use simple column assignment to ensure correct order
        for metric in metrics:
            # Main Value
            final_df[metric] = pivot_df[(metric, 'Values')]
            # Deltas
            final_df[f"{metric} 95 CI lower delta"] = pivot_df[(metric, 'Lower Delta')]
            final_df[f"{metric} 95 CI upper delta"] = pivot_df[(metric, 'Upper Delta')]
            
        # Reset index to make Agent a column
        final_df = final_df.reset_index()

        final_df.to_csv(output_path, index=False)
        print(f"Exported detailed results to {output_path}")

    def plot_metrics(self, output_path: Union[str, List[str]] = "metrics.html"):
        df = self._prepare_plot_data()

        category_order = [
            "Overall",
            "Voting Accuracy",
            "Win-Dependent Metrics",
            "Dominance Metrics",
            "Role-Specific Win Rate",
            "Role-Specific KSR",
            "Win-Dependent KSR",
            "Ratings",
            "Cost",
        ]
        present_categories = [cat for cat in category_order if cat in df["category"].unique()]

        max_cols = 0
        category_metrics_map = {}
        for cat in present_categories:
            metrics_in_cat = df[df["category"] == cat]["metric"].unique()
            if cat == "Ratings":
                metrics_in_cat = sorted(metrics_in_cat, key=lambda x: 0 if x == "Elo" else 1)
            else:
                metrics_in_cat = sorted(metrics_in_cat)

            category_metrics_map[cat] = metrics_in_cat
            max_cols = max(max_cols, len(metrics_in_cat))

        plot_titles = []
        for cat in present_categories:
            metrics = category_metrics_map[cat]
            for m in metrics:
                plot_titles.append(m)
            for _ in range(max_cols - len(metrics)):
                plot_titles.append("")

        fig = make_subplots(
            rows=len(present_categories),
            cols=max_cols,
            row_titles=present_categories,
            subplot_titles=plot_titles,
            vertical_spacing=0.1,
            horizontal_spacing=0.04,
        )

        # Create a consistent color map for all agents and a default sorting order
        agent_gte_ratings = {name: metrics.gte_rating[0] for name, metrics in self.metrics.items()}
        all_agents_sorted = sorted(agent_gte_ratings, key=agent_gte_ratings.get)

        colors = _get_color_discrete_sequence(len(all_agents_sorted))
        agent_color_map = {agent: colors[i % len(colors)] for i, agent in enumerate(all_agents_sorted)}

        for row_idx, cat in enumerate(present_categories):
            metrics = category_metrics_map[cat]
            for col_idx, metric in enumerate(metrics):
                metric_data = df[(df["category"] == cat) & (df["metric"] == metric)]

                # Default to GTE rating order, but for Ratings category, sort by the metric's value
                if cat == "Ratings":
                    sorted_agents_by_value = metric_data.sort_values("value", ascending=True)["agent"].tolist()
                    fig.update_xaxes(
                        categoryorder="array", categoryarray=sorted_agents_by_value, row=row_idx + 1, col=col_idx + 1
                    )
                else:
                    fig.update_xaxes(
                        categoryorder="array", categoryarray=all_agents_sorted, row=row_idx + 1, col=col_idx + 1
                    )

                fig.add_trace(
                    go.Bar(
                        name=metric,
                        x=metric_data["agent"],
                        y=metric_data["value"],
                        error_y=dict(type="data", array=metric_data["CI95"]),
                        marker_color=metric_data["agent"].map(agent_color_map),
                        showlegend=False,
                        hovertemplate="<b>%{x}</b><br>%{y:.2f} ± %{error_y.array:.2f}<extra></extra>",
                    ),
                    row=row_idx + 1,
                    col=col_idx + 1,
                )

                if cat == "Ratings":
                    fig.update_yaxes(matches=None, row=row_idx + 1, col=col_idx + 1)
                else:
                    # Auto-range for cost
                    fig.update_yaxes(range=[0, 1.05] if cat != "Cost" else None, row=row_idx + 1, col=col_idx + 1)
                fig.update_xaxes(tickangle=45, row=row_idx + 1, col=col_idx + 1)

        # Move Row Titles to Left
        fig.for_each_annotation(
            lambda a: a.update(
                x=-0.06, xanchor="right", font=dict(size=14, color="#111827", weight="bold"), yanchor="middle"
            )
            if a.text in present_categories
            else None
        )

        title_text = f"Agent Performance Metrics<br><sup>Total Games: {len(self.games)} | Bootstrap Samples: Elo={getattr(self, 'elo_samples', 'N/A')}, OpenSkill={getattr(self, 'openskill_samples', 'N/A')}</sup>"
        fig.update_layout(
            title_text=title_text,
            title_font_size=24,
            title_x=0.01,
            height=350 * len(present_categories),
            width=250 * max_cols if max_cols > 2 else 1000,
            font=dict(family="Inter, sans-serif"),
            showlegend=False,
            margin=dict(l=120, r=50),
        )

        _save_figure(fig, output_path, width=fig.layout.width, height=fig.layout.height)
        return fig

    def plot_pareto_frontier(self, output_path: Union[str, List[str]] = "pareto_frontier.html"):
        # Gather data
        data = []
        for agent_name, metrics in self.metrics.items():
            # Use GTE Rating (Mean) for performance
            gte_mean, _ = getattr(metrics, 'gte_rating', (0.0, 0.0))
            avg_cost, _ = metrics.get_avg_cost()

            # We need valid cost data
            if avg_cost <= 0:
                continue

            data.append({"agent": agent_name, "gte_rating": gte_mean, "cost": avg_cost})

        if not data:
            print("No cost data available for Pareto plot.")
            return

        df = pd.DataFrame(data)

        # Find Pareto Frontier
        # A point is on the frontier if there is no other point that has HIGHER gte_rating AND LOWER cost.
        sorted_df = df.sort_values(by=["cost", "gte_rating"], ascending=[True, False])
        frontier_points = []
        current_max_rating = -float("inf")

        for index, row in sorted_df.iterrows():
            if row["gte_rating"] > current_max_rating:
                frontier_points.append(row)
                current_max_rating = row["gte_rating"]

        frontier_df = pd.DataFrame(frontier_points)

        # Plot
        fig = go.Figure()

        # All Agents
        fig.add_trace(
            go.Scatter(
                x=df["cost"],
                y=df["gte_rating"],
                mode="markers+text",
                text=df["agent"],
                textposition="top center",
                marker=dict(size=10, color="#6366F1"),
                name="Agents",
                hovertemplate="<b>%{text}</b><br>Cost: $%{x:.2f}<br>GTE Rating: %{y:.2f}<extra></extra>",
            )
        )

        # Frontier Line
        fig.add_trace(
            go.Scatter(
                x=frontier_df["cost"],
                y=frontier_df["gte_rating"],
                mode="lines",
                line=dict(color="#10B981", width=2, dash="dash"),
                name="Pareto Frontier",
                hoverinfo="skip",
            )
        )

        fig.update_layout(
            title=f"Cost-Performance Pareto Frontier (GTE)<br><sup>Total Games: {len(self.games)} | GTE Bootstrap Samples: {getattr(self, 'gte_samples', 'N/A')}</sup>",
            xaxis_title="Average Cost per Game ($)",
            yaxis_title="GTE Overall Rating",
            template="plotly_white",
            font=dict(family="Inter, sans-serif"),
            width=900,
            height=600,
        )

        _save_figure(fig, output_path, width=900, height=600)
        return fig

    def plot_gte_evaluation(self, output_path: Union[str, List[str]] = "gte_evaluation.html"):
        if self.gte_game is None:
            print("GTE evaluation has not been run. Please run .evaluate() first.")
            return None

        # --- 1. Data Preparation ---
        # Use agents from the GTE game structure if available
        # If rating_player=1 (Agents), we should look at actions[1]
        game = self.gte_game
        if game and getattr(game, 'actions', None) and len(game.actions) > 1:
             agents = game.actions[1]
        else:
             print("Warning: GTE Game structure missing agent list. Falling back to metrics keys (risky).")
             agents = sorted(list(self.metrics.keys()))

        tasks = self.gte_tasks

        ratings_mean = self.gte_ratings[0][1]
        ratings_std = self.gte_ratings[1][1]
        
        joint_mean = self.gte_joint[0]
        r2m_contributions_mean = self.gte_contributions_raw[0]

        agent_rating_map = {agent: ratings_mean[i] for i, agent in enumerate(agents)}
        sorted_agents = sorted(agents, key=lambda x: agent_rating_map[x])

        game = self.gte_game
        rating_player = 1  # Agents
        contrib_player = 0 # Metrics

        if not game or not getattr(game, 'actions', None) or len(game.actions) <= rating_player:
            print(f"    !!! WARNING: Cannot plot GTE evaluation: Metadata actions missing or incomplete.")
            return None
        
        rating_actions = game.actions[rating_player]
        contrib_actions = game.actions[contrib_player]

        joint_support = jnp.sum(
            jnp.moveaxis(joint_mean, (rating_player, contrib_player), (0, 1)),
            axis=tuple(range(2, len(joint_mean.shape))),
        )

        rating_actions_grid, contrib_actions_grid = np.meshgrid(
            jnp.arange(len(rating_actions)),
            jnp.arange(len(contrib_actions)),
            indexing="ij",
        )

        data = pd.DataFrame.from_dict(
            {
                "agent": [rating_actions[i] for i in rating_actions_grid.ravel()],
                "metric": [contrib_actions[i] for i in contrib_actions_grid.ravel()],
                "contrib": r2m_contributions_mean.ravel(),
                "support": joint_support.ravel(),
            }
        )

        # Validate parallel array lengths before DataFrame creation
        if len(agents) != len(ratings_mean):
            print(f"    !!! ERROR: Mismatched GTE rating lengths: Agents={len(agents)}, Ratings={len(ratings_mean)}. Skipping plot.")
            return None

        agent_ratings_df = pd.DataFrame(
            {
                "agent": agents,
                "rating": ratings_mean,
                "CI95": ratings_std * 1.96,  # 95% CI
            }
        )

        n_items = len(agents) + 2
        
        # --- 3. Build Plot ---
        # Single plot for Agents
        fig = go.Figure()

        # Contributions
        colors = _get_color_discrete_sequence(len(tasks))
        for i, metric in enumerate(tasks):
            subset = data[data["metric"] == metric]
            fig.add_trace(
                go.Bar(
                    name=metric,
                    y=subset["agent"],
                    x=subset["contrib"],
                    orientation="h",
                    legendgroup="metrics",
                    legendgrouptitle_text="Metrics",
                    marker_color=colors[i % len(colors)],
                    hovertemplate=f"<b>Metric: {metric}</b><br>Contrib: %{{x:.2%}}<extra></extra>",
                )
            )

        # Net Rating
        fig.add_trace(
            go.Scatter(
                name="Net Rating",
                y=agent_ratings_df["agent"],
                x=agent_ratings_df["rating"],
                mode="markers",
                marker=dict(symbol="diamond", size=12, color="black", line=dict(width=1.5, color="white")),
                error_x=dict(type="data", array=agent_ratings_df["CI95"], color="black", thickness=2),
                hovertemplate="<b>%{y}</b><br>Net Rating: %{x:.2%}<br>Std: %{error_x.array:.4f}<extra></extra>",
            )
        )

        # --- 4. Layout Formatting ---
        fig.update_layout(
            barmode="relative",
            height=max(600, n_items * 40),
            width=1300,
            title_text=f"Game Theoretic Evaluation: Agent Ratings<br><sup>Total Games: {len(self.games)} | GTE Bootstrap Samples: {getattr(self, 'gte_samples', 'N/A')}</sup>",
            title_font_size=24,
            bargap=0.15,
            legend=dict(yanchor="top", y=1, xanchor="left", x=1.02, orientation="v", tracegroupgap=20),
            font=dict(family="Inter, sans-serif"),
        )

        # Font Sizing
        fig.update_yaxes(categoryorder="array", categoryarray=sorted_agents, tickfont=dict(size=14, color="#1f2937"))
        
        fig.update_xaxes(
            tickformat=".0%",
            title_text="Win Rate Contribution (and Net Rating)",
            tickfont=dict(size=14, color="#1f2937"),
            title_font=dict(size=16, color="#111827"),
            gridcolor="#F3F4F6",
        )
        fig.update_yaxes(gridcolor="#F3F4F6")

        _save_figure(fig, output_path, width=1300, height=fig.layout.height)
        return fig

    def plot_gte_metrics_analysis(self, output_path: Union[str, List[str]] = "gte_metrics.html"):
        """Plots GTE Metric analysis: Weights (Marginals) and Ratings (Payoffs)."""
        if not hasattr(self, "gte_game") or self.gte_game is None:
            print("GTE evaluation not run or failed. Skipping metric analysis plot.")
            return None

        if not self.gte_game or not getattr(self.gte_game, 'actions', None) or len(self.gte_game.actions) < 2:
            print("    !!! WARNING: Cannot plot GTE metrics analysis: Metadata actions missing or incomplete.")
            return None
            
        # Tasks are Player 0
        agents = self.gte_game.actions[1]
        tasks = self.gte_game.actions[0]
        
        # Unpack Data
        # Marginals (Weights) - Player 0
        try:
            metric_weights_mean = self.gte_marginals[0][0]
            metric_weights_std = self.gte_marginals[1][0]
            
            # Ratings (Values) - Player 0
            metric_ratings_mean = self.gte_ratings[0][0]
            metric_ratings_std = self.gte_ratings[1][0]
        except (IndexError, TypeError) as e:
            print(f"    !!! WARNING: GTE results structure mismatch in plot: {e}")
            return None

        tasks = self.gte_tasks
        
        # Create DataFrames
        weights_df = pd.DataFrame({
            "metric": tasks,
            "weight": metric_weights_mean,
            "CI95": metric_weights_std * 1.96
        })
        
        ratings_df = pd.DataFrame({
            "metric": tasks,
            "rating": metric_ratings_mean,
            "CI95": metric_ratings_std * 1.96
        })

        # Sort by Weight for consistency? Or Rating?
        # User might want to see most important metrics first.
        weights_df = weights_df.sort_values("weight", ascending=True)
        sorted_tasks = weights_df["metric"].tolist()
        
        # --- Build Plot ---
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.5, 0.5],
            vertical_spacing=0.15,
            subplot_titles=("Metric Importance (Marginal Probability / Weight)", "Metric Ratings (Nash Value)")
        )

        # 1. Weights (Marginals)
        fig.add_trace(
            go.Bar(
                name="Metric Weight",
                y=weights_df["metric"],
                x=weights_df["weight"],
                orientation="h",
                marker_color="#3b82f6", # Blue
                error_x=dict(type="data", array=weights_df["CI95"], color="black", thickness=1.5),
                hovertemplate="<b>%{y}</b><br>Weight: %{x:.4f} ± %{error_x.array:.4f}<extra></extra>"
            ),
            row=1, col=1
        )

        # 2. Ratings (Values)
        # Align with sorted tasks from weights
        ratings_df = ratings_df.set_index("metric").reindex(sorted_tasks).reset_index()
        
        fig.add_trace(
            go.Scatter(
                name="Metric Rating",
                y=ratings_df["metric"],
                x=ratings_df["rating"],
                mode="markers",
                marker=dict(symbol="diamond", size=10, color="#ef4444", line=dict(width=1, color="white")),
                error_x=dict(type="data", array=ratings_df["CI95"], color="black", thickness=1.5),
                hovertemplate="<b>%{y}</b><br>Rating: %{x:.2f} ± %{error_x.array:.2f}<extra></extra>"
            ),
            row=2, col=1
        )

        # Layout
        fig.update_layout(
            height=max(800, len(tasks) * 50),
            width=1000,
            title_text="GTE Metric Analysis",
            showlegend=False,
            font=dict(family="Inter, sans-serif"),
        )
        
        # Sort Top Plot by Weight
        fig.update_yaxes(categoryorder="array", categoryarray=sorted_tasks, tickfont=dict(size=12), row=1, col=1)
        
        # Sort Bottom Plot by Rating
        ratings_df_sorted = ratings_df.sort_values("rating", ascending=True)
        sorted_tasks_rating = ratings_df_sorted["metric"].tolist()
        fig.update_yaxes(categoryorder="array", categoryarray=sorted_tasks_rating, tickfont=dict(size=12), row=2, col=1)

        fig.update_xaxes(title_text="Marginal Probability", row=1, col=1, gridcolor="#F3F4F6")
        fig.update_xaxes(title_text="Nash Value (Rating)", row=2, col=1, gridcolor="#F3F4F6")

        _save_figure(fig, output_path, width=1000, height=fig.layout.height)
        return fig

    def save_metrics_csv(self, output_path: str = "metrics.csv"):
        """Saves the metrics to a CSV file in a wide format with deltas.

        Format: Index=Agent, Columns=Metric, {Metric} 95 CI lower delta, {Metric} 95 CI upper delta
        """
        df = self._prepare_plot_data()
        
        # Calculate deltas (Mean - Lower, Upper - Mean). 
        # In _prepare_plot_data, CI95 is symmetric (1.96 * std).
        # User wants lower_delta to be negative (e.g. -0.05) and upper_delta to be positive (e.g. +0.05)
        df["lower_delta"] = -df["CI95"]
        df["upper_delta"] = df["CI95"]

        # Disambiguate Role-Specific metrics
        def disambiguate_metric(row):
            cat = row["category"]
            met = row["metric"]
            if cat == "Role-Specific Win Rate":
                return f"WinRate {met}"
            elif cat == "Role-Specific KSR":
                return f"KSR {met}"
            elif cat == "Win-Dependent KSR":
                return f"WD-KSR {met}"
            return met
        
        if "category" in df.columns:
            df["metric"] = df.apply(disambiguate_metric, axis=1)

        # Pivot
        # Use pivot_table to handle potential duplicates (though unlikely with disambiguation)
        pivot_df = df.pivot_table(index="agent", columns="metric", values=["value", "lower_delta", "upper_delta"], aggfunc="mean")
        
        final_df = pd.DataFrame(index=pivot_df.index)
        
        # Determine column order
        present_metrics = sorted(df["metric"].unique())
        priority_order = ["GTE Rating", "Elo", "OpenSkill"]
        
        sorted_metrics = []
        for pm in priority_order:
            if pm in present_metrics:
                sorted_metrics.append(pm)
                present_metrics.remove(pm)
        sorted_metrics.extend(present_metrics)
        
        for metric in sorted_metrics:
            if ("value", metric) in pivot_df.columns:
                final_df[metric] = pivot_df[("value", metric)]
            if ("lower_delta", metric) in pivot_df.columns:
                final_df[f"{metric} 95 CI lower delta"] = pivot_df[("lower_delta", metric)]
            if ("upper_delta", metric) in pivot_df.columns:
                final_df[f"{metric} 95 CI upper delta"] = pivot_df[("upper_delta", metric)]
                
        print(f"Saving wide-format metrics to {output_path}")
        final_df.to_csv(output_path)

    def save_metrics_csv(self, output_path: str = "metrics.csv"):
        """Saves the metrics to a CSV file in a wide format with deltas.

        Format: Index=Agent, Columns=Metric, {Metric} 95 CI lower delta, {Metric} 95 CI upper delta
        """
        df = self._prepare_plot_data()
        
        # Calculate deltas (Mean - Lower, Upper - Mean). 
        # In _prepare_plot_data, CI95 is symmetric (1.96 * std).
        # User wants lower_delta to be negative (e.g. -0.05) and upper_delta to be positive (e.g. +0.05)
        df["lower_delta"] = -df["CI95"]
        df["upper_delta"] = df["CI95"]

        # Rename metrics if needed (already handled "OpenSkill" in _prepare_plot_data)
        
        # Disambiguate Role-Specific metrics
        def disambiguate_metric(row):
            cat = row["category"]
            met = row["metric"]
            if cat == "Role-Specific Win Rate":
                return f"WinRate {met}"
            elif cat == "Role-Specific KSR":
                return f"KSR {met}"
            elif cat == "Win-Dependent KSR":
                return f"WD-KSR {met}"
            return met
        
        if "category" in df.columns:
            df["metric"] = df.apply(disambiguate_metric, axis=1)

        # Pivot
        # Use pivot_table to handle potential duplicates (though unlikely with disambiguation)
        pivot_df = df.pivot_table(index="agent", columns="metric", values=["value", "lower_delta", "upper_delta"], aggfunc="mean")
        
        final_df = pd.DataFrame(index=pivot_df.index)
        
        # Determine column order
        present_metrics = sorted(df["metric"].unique())
        priority_order = ["GTE Rating", "Elo", "OpenSkill"]
        
        sorted_metrics = []
        for pm in priority_order:
            if pm in present_metrics:
                sorted_metrics.append(pm)
                present_metrics.remove(pm)
        sorted_metrics.extend(present_metrics)
        
        for metric in sorted_metrics:
            if ("value", metric) in pivot_df.columns:
                final_df[metric] = pivot_df[("value", metric)]
            if ("lower_delta", metric) in pivot_df.columns:
                final_df[f"{metric} 95 CI lower delta"] = pivot_df[("lower_delta", metric)]
            if ("upper_delta", metric) in pivot_df.columns:
                final_df[f"{metric} 95 CI upper delta"] = pivot_df[("upper_delta", metric)]
                
        print(f"Saving wide-format metrics to {output_path}")
        final_df.to_csv(output_path)


    def save_metrics_json(self, output_path: str = "metrics.json"):
        """Saves global stats and agent metrics to a JSON file."""
        import json
        
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NumpyEncoder, self).default(obj)
        
        # 1. Global Stats
        total_games = len(self.games)
        villager_wins = sum(1 for g in self.games if g.winner_team == Team.VILLAGERS)
        werewolf_wins = sum(1 for g in self.games if g.winner_team == Team.WEREWOLVES)
        
        avg_duration = 0
        durations = []
        for g in self.games:
            if getattr(g, "player_durations", None):
                durations.append(max(g.player_durations.values()))
        if durations:
            avg_duration = sum(durations) / len(durations)

        global_stats = {
            "total_games": total_games,
            "villager_wins": villager_wins,
            "villager_win_rate": villager_wins / total_games if total_games > 0 else 0,
            "werewolf_wins": werewolf_wins,
            "werewolf_win_rate": werewolf_wins / total_games if total_games > 0 else 0,
            "avg_game_duration": avg_duration
        }

        # 2. Agent Metrics serialization
        agents_data = {}
        for agent_name, stats in self.metrics.items():
            win_rate, win_std = stats.get_win_rate()
            ksr, ksr_std = stats.get_ksr()
            wd_ksr, wd_ksr_std = stats.get_wd_ksr()
            
            # Role data
            roles_data = {}
            for role in stats.wins_by_role.keys():
                r_wr, r_wr_std = stats.get_win_rate_for_role(role)
                r_ksr, r_ksr_std = stats.get_ksr_for_role(role)
                roles_data[role] = {
                    "win_rate": r_wr, "win_rate_ci95": r_wr_std * 1.96,
                    "ksr": r_ksr, "ksr_ci95": r_ksr_std * 1.96,
                    "games": len(stats.wins_by_role[role])
                }

            gte_mean, gte_std = getattr(stats, 'gte_rating', (0.0, 0.0))
            gte_attribs = {}
            for t, val in stats.gte_contributions.items():
                gte_attribs[t] = {"mean": val[0], "ci95": val[1] * 1.96}

            agent_dict = {
                "win_rate": win_rate, "win_rate_ci95": win_std * 1.96,
                "ksr": ksr, "ksr_ci95": ksr_std * 1.96,
                "wd_ksr": wd_ksr, "wd_ksr_ci95": wd_ksr_std * 1.96,
                "elo": stats.elo,
                "openskill_mu": stats.openskill_rating.mu if stats.openskill_rating else None,
                "gte_rating": gte_mean, "gte_rating_ci95": gte_std * 1.96,
                "gte_contributions": gte_attribs,
                "roles": roles_data,
                "total_games": len(stats.wins)
            }
            agents_data[agent_name] = agent_dict

        output_data = {
            "global_stats": global_stats,
            "agents": agents_data
        }

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2, cls=NumpyEncoder)
        print(f"Saved JSON metrics to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Werewolf game records.")
    parser.add_argument("input_dir", help="Directory containing .json game files.")
    parser.add_argument("--error-log", default="game_loading_errors.log", help="Path to log game loading errors.")
    parser.add_argument("--gte-tasks", default="win_dependent", help="GTE tasks configuration.")
    parser.add_argument("--output-prefix", default="", help="Prefix for output files (plots).")
    parser.add_argument("--cache-dir", default=".werewolf_metrics_cache",
                        help="Directory to store cached game results (default: .werewolf_metrics_cache).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for bootstrapping (default: 42).")
    parser.add_argument("--gte-samples", type=int, default=100, help="Number of bootstrap samples for GTE (default: 100).")
    parser.add_argument("--elo-samples", type=int, default=100, help="Number of bootstrap samples for Elo (default: 100).")
    parser.add_argument("--openskill-samples", type=int, default=100, help="Number of bootstrap samples for OpenSkill (default: 100).")
    
    parser.add_argument("--csv-output", help="Path to save metrics as CSV (default: <output_prefix>/<gte_tasks>/metrics.csv).")
    parser.add_argument("--json-output", help="Path to save metrics as JSON (default: <output_prefix>/<gte_tasks>/metrics.json).")

    args = parser.parse_args()

    evaluator = GameSetEvaluator(
        input_dir=args.input_dir,
        gte_tasks=args.gte_tasks,
        error_log_path=args.error_log,
        cache_dir=args.cache_dir,
        seed=args.seed
    )

    # Run evaluation
    evaluator.evaluate(gte_samples=args.gte_samples, elo_samples=args.elo_samples, openskill_samples=args.openskill_samples)

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    evaluator.print_results()

    # Determine output directory
    task_subdir = args.gte_tasks if isinstance(args.gte_tasks, str) else "custom_tasks"
    output_base_dir = Path(args.output_prefix) if args.output_prefix else Path(".")
    output_dir = output_base_dir / task_subdir
    os.makedirs(output_dir, exist_ok=True)

    # Save CSV (default to metrics.csv in output_dir)
    csv_path = args.csv_output if args.csv_output else str(output_dir / "metrics.csv")
    evaluator.save_metrics_csv(csv_path)

    # Save JSON (default to metrics.json in output_dir)
    json_path = args.json_output if args.json_output else str(output_dir / "metrics.json")
    evaluator.save_metrics_json(json_path)

    # Save plots
    print(f"\nSaving plots to {output_dir}...")
    evaluator.plot_metrics([str(output_dir / "metrics.html"), str(output_dir / "metrics.png")])
    evaluator.plot_gte_evaluation([str(output_dir / "gte.html"), str(output_dir / "gte.png")])
    evaluator.plot_gte_metrics_analysis([str(output_dir / "gte_metrics.html"), str(output_dir / "gte_metrics.png")])
    evaluator.plot_pareto_frontier([str(output_dir / "pareto.html"), str(output_dir / "pareto.png")])
