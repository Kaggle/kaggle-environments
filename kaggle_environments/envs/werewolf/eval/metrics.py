import argparse
import functools
import os
import sys
from collections import defaultdict, namedtuple
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterator, List, Tuple, Union

try:
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import polarix as plx
except ImportError:
    raise ImportError("The 'polarix' library is required for metrics calculation. Please install it via pip: `pip install polarix`.")

try:
    from openskill.models import PlackettLuce
except ImportError:
    raise ImportError("The 'openskill' library is required for rating calculation. Please install it via pip: `pip install openskill`.")

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

try:
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.io as pio
    from plotly.subplots import make_subplots

    # Set a default sophisticated template
    pio.templates.default = "plotly_white"
except ImportError:
    raise ImportError("The 'plotly' library is required for plotting. Please install it via pip: `pip install plotly`.")

from kaggle_environments.envs.werewolf.eval.loaders import Agent, Player, Role, _load_game_result
from kaggle_environments.envs.werewolf.game.consts import Team


def _mean_sem(values: List[float]) -> Tuple[float, float]:
    """Helper to calculate mean and standard error of the mean.

    Args:
        values: A list of numerical values.

    Returns:
        A tuple (mean, sem). Returns (0.0, 0.0) if the list is empty.
    """
    if not values:
        return 0.0, 0.0
    if len(values) < 2:
        return float(np.mean(values)), 0.0
    return float(np.mean(values)), float(np.std(values, ddof=1) / np.sqrt(len(values)))


def calculate_elo_change(p1_elo, p2_elo, result, k=32):
    """Calculates the change in Elo rating for Player 1.

    Args:
        p1_elo: Rating of Player 1.
        p2_elo: Rating of Player 2.
        result: 1 if Player 1 wins, 0 if Player 1 loses, 0.5 for draw.
        k: K-factor.

    Returns:
        The change in rating for Player 1 (can be negative).
    """
    expected = 1 / (1 + 10 ** ((p2_elo - p1_elo) / 400))
    return k * (result - expected)


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


def _openskill_bootstrap_worker(games_sample, model):
    if model is None:
        return {}
    ratings = GameSetEvaluator._compute_openskill_ratings(games_sample, model)
    return {name: r.mu for name, r in ratings.items()}


def _gte_bootstrap_worker(sampled_games, agents, tasks):
    """Worker function for parallel GTE bootstrapping.

    Args:
        sampled_games: A list of LightGame objects.
        agents: A list of agent names.
        tasks: A list of task names (metrics).

    Returns:
        A tuple containing:
        - ratings_np: List of numpy arrays of ratings.
        - joint_np: Numpy array of the joint distribution.
        - marginals_np: List of numpy arrays of marginal distributions.
        - contributions_np: Numpy array of contributions.
        - game_meta: SimpleNamespace containing game metadata (actions).
    """
    rnd = np.random.default_rng()

    agent_set = set(agents)
    task_set = set(tasks)
    agent_scores = {agent: {task: [] for task in tasks} for agent in agents}

    for game in sampled_games:
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
                game_duration = max(game.player_durations.values()) if game.player_durations else 0
            if game_duration > 0:
                speed_of_win = 1.0 / game_duration

        for player in game.players:
            agent_name = player.agent.display_name
            if agent_name not in agent_set:
                continue

            role_name = player.role.name
            won = player.role.team == game.winner_team
            alive = 1 if player.alive else 0

            # WinRate
            win_rate_task = f"WinRate-{role_name}"
            if win_rate_task in task_set:
                agent_scores[agent_name][win_rate_task].append(1 if won else 0)

            # KSR
            if "KSR" in task_set:
                agent_scores[agent_name]["KSR"].append(alive)
            ksr_task = f"KSR-{role_name}"
            if ksr_task in task_set:
                agent_scores[agent_name][ksr_task].append(alive)

            # WD-KSR
            wd_ksr_score = 1 if won and alive else 0
            if "WD-KSR" in task_set:
                agent_scores[agent_name]["WD-KSR"].append(wd_ksr_score)
            wd_ksr_task = f"WD-KSR-{role_name}"
            if wd_ksr_task in task_set:
                agent_scores[agent_name][wd_ksr_task].append(wd_ksr_score)

            if won:
                if "Margin of Win" in task_set:
                    agent_scores[agent_name]["Margin of Win"].append(margin_of_win)
                if "Speed of Win" in task_set:
                    agent_scores[agent_name]["Speed of Win"].append(speed_of_win)

        # IRP & VSS
        villagers_won = game.winner_team == Team.VILLAGERS
        irp_results, vss_results = game.irp_results, game.vss_results

        for agent_name, score in irp_results:
            if agent_name not in agent_set:
                continue
            if "IRP" in task_set:
                agent_scores[agent_name]["IRP"].append(score)
            if "WD-IRP" in task_set:
                agent_scores[agent_name]["WD-IRP"].append(score if villagers_won else 0)

        for agent_name, score in vss_results:
            if agent_name not in agent_set:
                continue
            if "VSS" in task_set:
                agent_scores[agent_name]["VSS"].append(score)
            if "WD-VSS" in task_set:
                agent_scores[agent_name]["WD-VSS"].append(score if villagers_won else 0)

    # 3. Build Matrices
    mean_matrix = np.zeros((len(agents), len(tasks)))
    stddev_matrix = np.zeros((len(agents), len(tasks)))
    for i, agent in enumerate(agents):
        for j, task in enumerate(tasks):
            scores = agent_scores[agent][task]
            if scores:
                mean_matrix[i, j] = np.mean(scores)
                if len(scores) > 1:
                    stddev_matrix[i, j] = np.std(scores, ddof=1) / np.sqrt(len(scores))
                else:
                    stddev_matrix[i, j] = 0.0

    # Regularization for degenerate cases
    for j in range(mean_matrix.shape[1]):
        if np.ptp(mean_matrix[:, j]) < 1e-9:
            mean_matrix[:, j] += rnd.random(mean_matrix.shape[0]) * 1e-6
            stddev_matrix[:, j] += rnd.random(mean_matrix.shape[0]) * 1e-6

    # 4. Solve Game
    game_plx = plx.agent_vs_task_game(
        agents=agents,
        tasks=tasks,
        agent_vs_task=mean_matrix,
        agent_vs_task_stddev=stddev_matrix,
        task_player="metric",
        normalizer="winrate",
    )
    res = plx.solve(game_plx, plx.ce_maxent, disable_progress_bar=True)
    marginals = plx.marginals_from_joint(res.joint)
    r2m_contributions = plx.joint_payoffs_contribution(game_plx.payoffs, res.joint, rating_player=1, contrib_player=0)

    m2r_contributions = plx.joint_payoffs_contribution(game_plx.payoffs, res.joint, rating_player=0, contrib_player=1)

    # Convert JAX arrays to Numpy to avoid initializing JAX in the parent process via pickling
    # which can cause deadlocks when forking subsequent worker processes.
    ratings_np = [np.array(r) for r in res.ratings]
    joint_np = np.array(res.joint)
    marginals_np = [np.array(m) for m in marginals]
    r2m_contributions_np = np.array(r2m_contributions)
    m2r_contributions_np = np.array(m2r_contributions)

    # Use a SimpleNamespace to pass back only the actions, avoiding the full game object
    # which contains JAX arrays (payoffs).
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
    ):
        if isinstance(input_dir, str):
            input_dirs = [input_dir]
        else:
            input_dirs = input_dir

        self.games = []
        game_files = []
        for directory in input_dirs:
            for root, _, files in os.walk(directory):
                for file in files:
                    if file.endswith(".json"):
                        game_files.append(os.path.join(root, file))

        args_list = [(f, preserve_full_game_records) for f in game_files]
        errors = defaultdict(int)

        with ProcessPoolExecutor() as executor:
            results_iter = tqdm(
                executor.map(_safe_load_game_result, args_list), total=len(args_list), desc="Loading Games"
            )
            for result, error in results_iter:
                if error:
                    error_key = f"{type(error).__name__}: {str(error)}"
                    errors[error_key] += 1
                elif result:
                    self.games.append(result)

        if errors:
            print("\nErrors encountered during game loading:")
            with open(error_log_path, "w") as f:
                f.write("Game Loading Errors Report\n")
                f.write("==========================\n")
                for error_key, count in sorted(errors.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {error_key}: {count}")
                    f.write(f"{error_key}: {count}\n")
            print(f"Detailed error report saved to {error_log_path}")

        self.openskill_model = PlackettLuce() if OPENSKILL_AVAILABLE else None

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

        roles = sorted(list(set(p.role.name for g in self.games for p in g.players)))
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
                result_v = 1 if game.winner_team == Team.VILLAGERS else 0
                for agent in villager_agents:
                    elos[agent] += calculate_elo_change(elos[agent], avg_w_elo, result_v)
                for agent in werewolf_agents:
                    elos[agent] += calculate_elo_change(elos[agent], avg_v_elo, 1 - result_v)
        return elos

    @staticmethod
    def _compute_openskill_ratings(games: List, model) -> Dict[str, object]:
        """Computes OpenSkill (TrueSkill-like) ratings."""
        if not OPENSKILL_AVAILABLE or not model:
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

    def _generate_bootstrap_samples(
        self, light_games_iterator: Iterator[LightGame], num_samples: int
    ) -> Iterator[List[LightGame]]:
        light_games = list(light_games_iterator)
        if not light_games:
            return

        # Create an object array and fill it to prevent numpy from unpacking the tuples
        light_games_np = np.empty(len(light_games), dtype=object)
        light_games_np[:] = light_games

        rnd_master = np.random.default_rng(42)

        for _ in range(num_samples):
            sampled_array = light_games_np[rnd_master.integers(0, len(light_games), size=len(light_games))]
            yield list(sampled_array)

    def _bootstrap_elo(self, num_samples=100):
        if not self.games:
            return

        def light_games_iter():
            for g in self.games:
                # Re-create lightweight players for this specific bootstrap
                players = [
                    Player(p.id, Agent(p.agent.display_name), Role(p.role.name, p.role.team), p.alive)
                    for p in g.players
                ]
                yield LightGame(players, g.winner_team, None, None, None)

        samples_iterator = self._generate_bootstrap_samples(light_games_iter(), num_samples)

        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            results = list(
                tqdm(
                    executor.map(GameSetEvaluator._compute_elo_ratings, samples_iterator),
                    total=num_samples,
                    desc="Elo Bootstrap",
                )
            )

        bootstrapped_elos = defaultdict(list)
        all_agents = list(self.metrics.keys())
        for sample_elos in results:
            for agent in all_agents:
                bootstrapped_elos[agent].append(sample_elos.get(agent, 1200.0))

        for agent, values in bootstrapped_elos.items():
            if len(values) > 1:
                self.metrics[agent].elo_std = float(np.std(values, ddof=1))

    def _bootstrap_openskill(self, num_samples=100):
        if not self.games or not OPENSKILL_AVAILABLE or not self.openskill_model:
            return

        def light_games_iter():
            for g in self.games:
                players = [
                    Player(p.id, Agent(p.agent.display_name), Role(p.role.name, p.role.team), p.alive)
                    for p in g.players
                ]
                yield LightGame(players, g.winner_team, None, None, None)

        samples_iterator = self._generate_bootstrap_samples(light_games_iter(), num_samples)

        worker = functools.partial(_openskill_bootstrap_worker, model=self.openskill_model)

        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            results = list(tqdm(executor.map(worker, samples_iterator), total=num_samples, desc="OpenSkill Bootstrap"))

        bootstrapped_mus = defaultdict(list)
        all_agents = list(self.metrics.keys())

        default_mu = self.openskill_model.rating().mu

        for sample_ratings in results:
            for agent in all_agents:
                if agent in sample_ratings:
                    bootstrapped_mus[agent].append(sample_ratings[agent])
                else:
                    bootstrapped_mus[agent].append(default_mu)

        for agent, values in bootstrapped_mus.items():
            if len(values) > 1:
                self.metrics[agent].openskill_mu_std = float(np.std(values, ddof=1))

    def evaluate(self, gte_samples=3, elo_samples=3, openskill_samples=3):
        for game in self.games:
            villagers_won = game.winner_team == Team.VILLAGERS
            # Capture costs if available from GameResult
            player_costs = getattr(game, "player_costs", {})
            player_tokens = getattr(game, "player_tokens", {})
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

        # Elo
        final_elos = self._compute_elo_ratings(self.games)
        for agent, rating in final_elos.items():
            self.metrics[agent].elo = rating

        # OpenSkill
        if OPENSKILL_AVAILABLE and self.openskill_model:
            final_openskill_ratings = self._compute_openskill_ratings(self.games, self.openskill_model)
            for agent, rating in final_openskill_ratings.items():
                self.metrics[agent].openskill_rating = rating

        self._bootstrap_elo(num_samples=elo_samples)
        self._bootstrap_openskill(num_samples=openskill_samples)
        self._run_gte_evaluation(num_samples=gte_samples)

    def _run_gte_evaluation(self, num_samples: int):
        if not POLARIX_AVAILABLE:
            print("Warning: `polarix` library not found. Skipping Game Theoretic Evaluation.")
            return
        agents = sorted(list(self.metrics.keys()))

        def light_gte_games_iter():
            for g in tqdm(self.games, desc="Pre-calculating GTE data"):
                irp_results, vss_results = g.iterate_voting_mini_game()
                players = [
                    Player(p.id, Agent(p.agent.display_name), Role(p.role.name, p.role.team), p.alive)
                    for p in g.players
                ]
                player_durations = getattr(g, "player_durations", {})
                yield LightGame(players, g.winner_team, irp_results, vss_results, player_durations)

        samples_iterator = self._generate_bootstrap_samples(light_gte_games_iter(), num_samples)

        worker_func = functools.partial(_gte_bootstrap_worker, agents=agents, tasks=self.gte_tasks)

        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            res = list(tqdm(executor.map(worker_func, samples_iterator), total=num_samples, desc="GTE Bootstrap"))

        # Unpack
        ratings, joints, marginals, r2m_contributions, m2r_contributions, games = zip(*res)
        self.gte_game = games[0]  # Save one game instance for structure

        ratings_mean = [np.mean(r, axis=0) for r in zip(*ratings)]
        ratings_std = [np.std(r, axis=0) for r in zip(*ratings)]

        joints_mean = np.mean(joints, axis=0)

        marginals_by_dim = list(zip(*marginals))
        marginals_mean = [np.mean(m, axis=0) for m in marginals_by_dim]
        marginals_std = [np.std(m, axis=0) for m in marginals_by_dim]

        r2m_contributions_mean = np.mean(r2m_contributions, axis=0)
        r2m_contributions_std = np.std(r2m_contributions, axis=0)

        m2r_contributions_mean = np.mean(m2r_contributions, axis=0)
        m2r_contributions_std = np.std(m2r_contributions, axis=0)

        self.gte_ratings = (ratings_mean, ratings_std)
        self.gte_joint = (joints_mean, None)
        self.gte_marginals = (marginals_mean, marginals_std)
        self.gte_contributions_raw = (r2m_contributions_mean, r2m_contributions_std)
        self.gte_metric_contributions_raw = (m2r_contributions_mean, m2r_contributions_std)

        for i, agent_name in enumerate(agents):
            self.metrics[agent_name].gte_rating = (ratings_mean[1][i], ratings_std[1][i])
            for j, task_name in enumerate(self.gte_tasks):
                self.metrics[agent_name].gte_contributions[task_name] = (
                    r2m_contributions_mean[i, j],
                    r2m_contributions_std[i, j],
                )

    def print_results(self):
        # [Keep original code]
        sorted_metrics = sorted(self.metrics.values(), key=lambda m: m.agent_name)
        for stats in sorted_metrics:
            print(f"Agent: {stats.agent_name}")
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
            if avg_cost > 0:
                print("  Cost Metrics:")
                print(f"    Avg Cost/Game: ${avg_cost:.4f} ± {cost_sem * 1.96:.4f}")
                print(f"    Avg Cost/Turn: ${avg_cost_per_turn:.4f} ± {cost_per_turn_sem * 1.96:.4f}")
                print(f"    Avg Tokens/Game: {avg_tokens:.1f} ± {tokens_sem * 1.96:.1f}")

            print("  Ratings:")
            print(f"    Elo: {stats.elo:.2f} ± {stats.elo_std * 1.96:.2f} (CI95)")
            if OPENSKILL_AVAILABLE and stats.openskill_rating:
                print(
                    f"    TrueSkill: mu={stats.openskill_rating.mu:.2f} ± {stats.openskill_mu_std * 1.96:.2f} (CI95), sigma={stats.openskill_rating.sigma:.2f}"
                )
            if POLARIX_AVAILABLE:
                print("  Game Theoretic Evaluation (GTE):")
                gte_mean, gte_std = stats.gte_rating
                print(f"    Overall GTE Rating: {gte_mean:.2f} ± {gte_std * 1.96:.2f} (CI95)")
                for task in self.gte_tasks:
                    contrib_mean, contrib_std = stats.gte_contributions[task]
                    print(f"    - {task:<30} Contribution: {contrib_mean:.2f} ± {contrib_std * 1.96:.2f} (CI95)")

                print("  GTE Metric Importance (Ratings):")
                metric_ratings_mean = self.gte_ratings[0][0]
                metric_ratings_std = self.gte_ratings[1][0]

                metric_ratings = []
                for i, task_name in enumerate(self.gte_tasks):
                    metric_ratings.append((task_name, metric_ratings_mean[i], metric_ratings_std[i]))

                # Sort by rating descending
                sorted_metric_ratings = sorted(metric_ratings, key=lambda x: x[1], reverse=True)

                for task, rating_mean, rating_std in sorted_metric_ratings:
                    print(f"    - {task:<30} Rating: {rating_mean:.2f} ± {rating_std * 1.96:.2f} (CI95)")
            print("-" * 30)

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

            # 8. Ratings
            plot_data.append(
                {
                    "agent": agent_name,
                    "metric": "Elo",
                    "value": metrics.elo,
                    "CI95": metrics.elo_std * 1.96,
                    "category": "Ratings",
                }
            )

            if OPENSKILL_AVAILABLE and metrics.openskill_rating:
                plot_data.append(
                    {
                        "agent": agent_name,
                        "metric": "TrueSkill",
                        "value": metrics.openskill_rating.mu,
                        "CI95": metrics.openskill_mu_std * 1.96,
                        "category": "Ratings",
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

        return pd.DataFrame(plot_data)

    def plot_metrics(self, output_path: Union[str, List[str]] = "metrics.html"):
        if not PLOTLY_AVAILABLE:
            print("Warning: `plotly` not found. Cannot plot metrics.")
            return

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

        fig.update_layout(
            title_text="Agent Performance Metrics",
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
            gte_mean, _ = metrics.gte_rating
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
            title="Cost-Performance Pareto Frontier (GTE)",
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
        agents = sorted(list(self.metrics.keys()))
        tasks = self.gte_tasks

        ratings_mean = self.gte_ratings[0][1]
        ratings_std = self.gte_ratings[1][1]
        metric_ratings_mean = self.gte_ratings[0][0]
        metric_ratings_std = self.gte_ratings[1][0]
        joint_mean = self.gte_joint[0]
        r2m_contributions_mean = self.gte_contributions_raw[0]
        m2r_contributions_mean = self.gte_metric_contributions_raw[0]

        agent_rating_map = {agent: ratings_mean[i] for i, agent in enumerate(agents)}
        sorted_agents = sorted(agents, key=lambda x: agent_rating_map[x])

        metric_rating_map = {task: metric_ratings_mean[i] for i, task in enumerate(tasks)}
        sorted_tasks = sorted(tasks, key=lambda x: metric_rating_map[x])

        game = self.gte_game
        rating_player = 1
        contrib_player = 0

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

        agent_ratings_df = pd.DataFrame(
            {
                "agent": agents,
                "rating": ratings_mean,
                "CI95": ratings_std * 1.96,  # 95% CI
            }
        )

        metric_ratings_df = pd.DataFrame(
            {"metric": tasks, "rating": metric_ratings_mean, "CI95": metric_ratings_std * 1.96}
        )

        tasks_actions_grid, agent_actions_grid = np.meshgrid(
            jnp.arange(len(tasks)),
            jnp.arange(len(agents)),
            indexing="ij",
        )

        metric_data = pd.DataFrame.from_dict(
            {
                "metric": [tasks[i] for i in tasks_actions_grid.ravel()],
                "agent": [agents[i] for i in agent_actions_grid.ravel()],
                "contrib": m2r_contributions_mean.ravel(),
            }
        )

        # --- 2. Dynamic Layout Calculation ---
        n_top = len(agents) + 2
        n_bottom = len(tasks) + 2
        total_items = n_top + n_bottom

        h_top = max(0.3, min(0.7, n_top / total_items))
        h_bottom = 1.0 - h_top

        # --- 3. Build Plot ---
        fig = make_subplots(
            rows=2,
            cols=1,
            row_heights=[h_top, h_bottom],
            vertical_spacing=0.1,
            subplot_titles=(
                "Win Rate Contributions & Equilibrium Ratings",
                "Metric Importance & Contributions from Agents",
            ),
        )

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
                ),
                row=1,
                col=1,
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
            ),
            row=1,
            col=1,
        )

        # Task Importance / Metric Ratings
        colors_agents = _get_color_discrete_sequence(len(agents))
        for i, agent in enumerate(agents):
            subset = metric_data[metric_data["agent"] == agent]
            fig.add_trace(
                go.Bar(
                    name=agent,
                    y=subset["metric"],
                    x=subset["contrib"],
                    orientation="h",
                    legendgroup="agents",
                    legendgrouptitle_text="Agents",
                    marker_color=colors_agents[i % len(colors_agents)],
                    hovertemplate=f"<b>Agent: {agent}</b><br>Contrib: %{{x:.2%}}<extra></extra>",
                ),
                row=2,
                col=1,
            )

        # Net Metric Rating
        fig.add_trace(
            go.Scatter(
                name="Net Metric Rating",
                y=metric_ratings_df["metric"],
                x=metric_ratings_df["rating"],
                mode="markers",
                marker=dict(symbol="diamond", size=12, color="black", line=dict(width=1.5, color="white")),
                error_x=dict(type="data", array=metric_ratings_df["CI95"], color="black", thickness=2),
                hovertemplate="<b>%{y}</b><br>Net Rating: %{x:.2%}<br>Std: %{error_x.array:.4f}<extra></extra>",
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        # --- 4. Layout Formatting ---
        fig.update_layout(
            barmode="relative",
            height=max(900, total_items * 35),
            width=1300,
            title_text="Game Theoretic Evaluation Results",
            title_font_size=24,
            bargap=0.15,
            legend=dict(yanchor="top", y=1, xanchor="left", x=1.02, orientation="v", tracegroupgap=20),
            font=dict(family="Inter, sans-serif"),
        )

        # Font Sizing
        fig.update_yaxes(
            categoryorder="array", categoryarray=sorted_agents, tickfont=dict(size=14, color="#1f2937"), row=1, col=1
        )
        fig.update_yaxes(
            categoryorder="array", categoryarray=sorted_tasks, tickfont=dict(size=14, color="#1f2937"), row=2, col=1
        )

        fig.update_xaxes(
            tickformat=".0%",
            title_text="Win Rate Contribution",
            tickfont=dict(size=14, color="#1f2937"),
            title_font=dict(size=16, color="#111827"),
            row=1,
            col=1,
            gridcolor="#F3F4F6",
        )
        fig.update_xaxes(
            tickformat=".0%",
            title_text="Contribution to Metric Rating",
            tickfont=dict(size=14, color="#1f2937"),
            title_font=dict(size=16, color="#111827"),
            row=2,
            col=1,
            gridcolor="#F3F4F6",
        )
        fig.update_yaxes(gridcolor="#F3F4F6")

        _save_figure(fig, output_path, width=1300, height=fig.layout.height)
        return fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Werewolf game records.")
    parser.add_argument("input_dir", help="Directory containing .json game files.")
    parser.add_argument("--error-log", default="game_loading_errors.log", help="Path to log game loading errors.")
    parser.add_argument("--gte-tasks", default="win_dependent", help="GTE tasks configuration.")
    parser.add_argument("--output-prefix", default="", help="Prefix for output files (plots).")

    args = parser.parse_args()

    evaluator = GameSetEvaluator(
        input_dir=args.input_dir,
        gte_tasks=args.gte_tasks,
        error_log_path=args.error_log
    )

    # Run evaluation
    # We use default sample counts for now, could be made configurable
    evaluator.evaluate(gte_samples=3, elo_samples=3, openskill_samples=3)

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    evaluator.print_results()

    # Save plots
    prefix = args.output_prefix
    evaluator.plot_metrics([f"{prefix}metrics.html", f"{prefix}metrics.png"])
    evaluator.plot_gte_evaluation([f"{prefix}gte.html", f"{prefix}gte.png"])
    evaluator.plot_pareto_frontier([f"{prefix}pareto.html", f"{prefix}pareto.png"])
