import functools
import os
from collections import defaultdict, namedtuple
from pathlib import Path
from typing import Dict, List, Tuple, Union, Iterator
from concurrent.futures import ProcessPoolExecutor

import jax.numpy as jnp
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import polarix as plx

    POLARIX_AVAILABLE = True
except ImportError:
    POLARIX_AVAILABLE = False

try:
    from openskill.models import PlackettLuce

    OPENSKILL_AVAILABLE = True
except ImportError:
    OPENSKILL_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.io as pio

    # Set a default sophisticated template
    pio.templates.default = "plotly_white"
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from kaggle_environments.envs.werewolf.eval.loaders import get_game_results, Agent, Role, Player
from kaggle_environments.envs.werewolf.game.consts import Team


def _mean_sem(values: List[float]) -> Tuple[float, float]:
    """Helper to calculate mean and standard error of the mean, handling empty lists."""
    if not values:
        return 0.0, 0.0
    if len(values) < 2:
        return float(np.mean(values)), 0.0
    return float(np.mean(values)), float(np.std(values, ddof=1) / np.sqrt(len(values)))


def calculate_elo_change(p1_elo, p2_elo, result, k=32):
    """
    Calculates the change in Elo rating for Player 1.
    :param p1_elo: Rating of Player 1
    :param p2_elo: Rating of Player 2
    :param result: 1 if Player 1 wins, 0 if Player 1 loses, 0.5 for draw
    :param k: K-factor
    :return: Change in rating for Player 1
    """
    expected = 1 / (1 + 10 ** ((p2_elo - p1_elo) / 400))
    return k * (result - expected)


# --- Plotting utils ---


def _save_figure(fig: "go.Figure", output_path: Union[str, List[str], None], width=None, height=None):
    """
    Saves a Plotly figure to one or multiple files.
    Args:
        output_path: A single filename (str) or a list of filenames (List[str]).
                     e.g., "plot.html" or ["plot.png", "plot.html"]
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
        if ext == '.html':
            fig.write_html(output_path)
        elif ext in ['.png', '.jpg', '.jpeg']:
            # scale=3 ensures high DPI (Retina quality)
            fig.write_image(output_path, width=width, height=height, scale=3)
        elif ext in ['.pdf', '.svg']:
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


LightGame = namedtuple('LightGame', ['players', 'winner_team', 'irp_results', 'vss_results'])


# --- Worker Globals and Functions ---

def _openskill_bootstrap_worker(games_sample, model):
    if model is None:
        return {}
    ratings = GameSetEvaluator._compute_openskill_ratings(games_sample, model)
    return {name: r.mu for name, r in ratings.items()}


def _gte_bootstrap_worker(sampled_games, agents, tasks):
    rnd = np.random.default_rng()

    agent_set = set(agents)
    task_set = set(tasks)
    agent_scores = {agent: {task: [] for task in tasks} for agent in agents}

    for game in sampled_games:
        # 1. Collect basic stats (WinRate, KSR)
        for player in game.players:
            agent_name = player.agent.display_name
            if agent_name in agent_set:
                role_name = player.role.name

                win_rate_task = f'WinRate-{role_name}'
                if win_rate_task in task_set:
                    agent_scores[agent_name][win_rate_task].append(1 if player.role.team == game.winner_team else 0)

                ksr_task = f'KSR-{role_name}'
                if ksr_task in task_set:
                    agent_scores[agent_name][ksr_task].append(1 if player.alive else 0)

                if 'KSR' in task_set:
                    agent_scores[agent_name]['KSR'].append(1 if player.alive else 0)

        # 2. Collect complex stats (IRP, VSS) requiring mini-game iteration
        irp_results, vss_results = game.irp_results, game.vss_results
        if 'IRP' in task_set:
            for agent_name, score in irp_results:
                if agent_name in agent_set:
                    agent_scores[agent_name]['IRP'].append(score)
        if 'VSS' in task_set:
            for agent_name, score in vss_results:
                if agent_name in agent_set:
                    agent_scores[agent_name]['VSS'].append(score)

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
        agents=agents, tasks=tasks, agent_vs_task=mean_matrix, agent_vs_task_stddev=stddev_matrix,
        task_player='metric', normalizer='winrate'
    )
    res = plx.solve(game_plx, plx.ce_maxent, disable_progress_bar=True)
    marginals = plx.marginals_from_joint(res.joint)
    contributions = plx.joint_payoffs_contribution(
        game_plx.payoffs, res.joint, rating_player=1, contrib_player=0
    )
    return res.ratings, res.joint, marginals, contributions, game_plx


def _default_elo():
    return 1200.0


def _default_gte_contrib():
    return 0.0, 0.0


class AgentMetrics:
    """Stores and calculates performance metrics for a single agent."""

    def __init__(self, agent_name: str, openskill_model):
        self.agent_name = agent_name
        self.wins: List[int] = []
        self.wins_by_role: Dict[str, List[int]] = defaultdict(list)
        self.irp_scores: List[int] = []
        self.vss_scores: List[int] = []
        self.survival_scores: List[int] = []
        self.survival_by_role: Dict[str, List[int]] = defaultdict(list)

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
        return _mean_sem(self.wins)

    def get_win_rate_for_role(self, role: str) -> Tuple[float, float]:
        return _mean_sem(self.wins_by_role.get(role, []))

    def get_irp(self) -> Tuple[float, float]:
        return _mean_sem(self.irp_scores)

    def get_vss(self) -> Tuple[float, float]:
        return _mean_sem(self.vss_scores)

    def get_ksr(self) -> Tuple[float, float]:
        return _mean_sem(self.survival_scores)

    def get_ksr_for_role(self, role: str) -> Tuple[float, float]:
        return _mean_sem(self.survival_by_role.get(role, []))


class GameSetEvaluator:
    """Evaluates a set of game replays and calculates metrics for each agent."""

    def __init__(self, input_dir: Union[str, List[str]], gte_tasks: List[str] = None,
                 preserve_full_game_records: bool = False):
        if isinstance(input_dir, str):
            input_dirs = [input_dir]
        else:
            input_dirs = input_dir

        self.games = []
        for directory in input_dirs:
            self.games.extend(get_game_results(directory, preserve_full_record=preserve_full_game_records))

        self.openskill_model = PlackettLuce() if OPENSKILL_AVAILABLE else None

        self.metrics: Dict[str, AgentMetrics] = defaultdict(
            lambda: AgentMetrics(agent_name=None, openskill_model=self.openskill_model))

        # For GTE
        self.gte_game = None
        self.gte_joint = None
        self.gte_ratings = None
        self.gte_marginals = None
        self.gte_contributions_raw = None

        if gte_tasks is None:
            roles = sorted(list(set(p.role.name for g in self.games for p in g.players)))
            self.gte_tasks = ([f'WinRate-{r}' for r in roles] +
                              [f'KSR-{r}' for r in roles] +
                              ['IRP', 'VSS', 'KSR'])
        else:
            self.gte_tasks = gte_tasks

    @staticmethod
    def _compute_elo_ratings(games: List) -> Dict[str, float]:
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

    def _generate_bootstrap_samples(self, light_games_iterator: Iterator[LightGame], num_samples: int) -> Iterator[List[LightGame]]:
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
        if not self.games: return

        def light_games_iter():
            for g in self.games:
                # Re-create lightweight players for this specific bootstrap
                players = [Player(p.id, Agent(p.agent.display_name), Role(p.role.name, p.role.team), p.alive) for p in g.players]
                yield LightGame(players, g.winner_team, None, None)

        samples_iterator = self._generate_bootstrap_samples(light_games_iter(), num_samples)

        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            results = list(tqdm(executor.map(GameSetEvaluator._compute_elo_ratings, samples_iterator), total=num_samples, desc="Elo Bootstrap"))

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
                players = [Player(p.id, Agent(p.agent.display_name), Role(p.role.name, p.role.team), p.alive) for p in g.players]
                yield LightGame(players, g.winner_team, None, None)
        
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
            irp_results, vss_results = game.iterate_voting_mini_game()
            for agent_name, score in irp_results:
                self.metrics[agent_name].irp_scores.append(score)
            for agent_name, score in vss_results:
                self.metrics[agent_name].vss_scores.append(score)

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
                yield LightGame(players, g.winner_team, irp_results, vss_results)

        samples_iterator = self._generate_bootstrap_samples(light_gte_games_iter(), num_samples)
        
        worker_func = functools.partial(_gte_bootstrap_worker, agents=agents, tasks=self.gte_tasks)
        
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            res = list(tqdm(executor.map(worker_func, samples_iterator), total=num_samples, desc="GTE Bootstrap"))
            
        # Unpack
        ratings, joints, marginals, contributions, games = zip(*res)
        self.gte_game = games[0] # Save one game instance for structure

        ratings_mean = [np.mean(r, axis=0) for r in zip(*ratings)]
        ratings_std = [np.std(r, axis=0) for r in zip(*ratings)]

        joints_mean = np.mean(joints, axis=0)
        
        marginals_by_dim = list(zip(*marginals))
        marginals_mean = [np.mean(m, axis=0) for m in marginals_by_dim]
        marginals_std = [np.std(m, axis=0) for m in marginals_by_dim]

        contributions_mean = np.mean(contributions, axis=0)
        contributions_std = np.std(contributions, axis=0)
        
        self.gte_ratings = (ratings_mean, ratings_std)
        self.gte_joint = (joints_mean, None)
        self.gte_marginals = (marginals_mean, marginals_std)
        self.gte_contributions_raw = (contributions_mean, contributions_std)

        for i, agent_name in enumerate(agents):
            self.metrics[agent_name].gte_rating = (ratings_mean[1][i], ratings_std[1][i])
            for j, task_name in enumerate(self.gte_tasks):
                self.metrics[agent_name].gte_contributions[task_name] = (
                    contributions_mean[i, j], contributions_std[i, j])

    def print_results(self):
        # [Keep original code]
        sorted_metrics = sorted(self.metrics.values(), key=lambda m: m.agent_name)
        for stats in sorted_metrics:
            print(f"Agent: {stats.agent_name}")
            win_rate, win_std = stats.get_win_rate()
            print(f"  Overall Win Rate: {win_rate:.2f} ± {win_std * 1.96:.2f} (CI95) ({len(stats.wins)} games)")
            ksr, ksr_std = stats.get_ksr()
            print(f"  Overall Survival Rate: {ksr:.2f} ± {ksr_std * 1.96:.2f} (CI95)")
            print("  Role-Specific Win Rates:")
            for role in sorted(stats.wins_by_role.keys()):
                role_rate, role_std = stats.get_win_rate_for_role(role)
                game_count = len(stats.wins_by_role[role])
                print(f"    {role:<10}: {role_rate:.2f} ± {role_std * 1.96:.2f} (CI95) ({game_count} games)")
            print("  Role-Specific Survival Rates (KSR):")
            for role in sorted(stats.survival_by_role.keys()):
                role_ksr, role_ksr_std = stats.get_ksr_for_role(role)
                game_count = len(stats.survival_by_role[role])
                print(f"    {role:<10}: {role_ksr:.2f} ± {role_ksr_std * 1.96:.2f} (CI95) ({game_count} games)")
            irp, irp_std = stats.get_irp()
            vss, vss_std = stats.get_vss()
            print("  Voting Accuracy (Villager Team):")
            print(f"    IRP (Identification Precision): {irp:.2f} ± {irp_std * 1.96:.2f} (CI95) ({len(stats.irp_scores)} votes)")
            print(f"    VSS (Voting Success Score):     {vss:.2f} ± {vss_std * 1.96:.2f} (CI95) ({len(stats.vss_scores)} votes)")
            print("  Ratings:")
            print(f"    Elo: {stats.elo:.2f} ± {stats.elo_std * 1.96:.2f} (CI95)")
            if OPENSKILL_AVAILABLE and stats.openskill_rating:
                print(
                    f"    TrueSkill: mu={stats.openskill_rating.mu:.2f} ± {stats.openskill_mu_std * 1.96:.2f} (CI95), sigma={stats.openskill_rating.sigma:.2f}")
            if POLARIX_AVAILABLE:
                print("  Game Theoretic Evaluation (GTE):")
                gte_mean, gte_std = stats.gte_rating
                print(f"    Overall GTE Rating: {gte_mean:.2f} ± {gte_std * 1.96:.2f} (CI95)")
                for task in self.gte_tasks:
                    contrib_mean, contrib_std = stats.gte_contributions[task]
                    print(f"    - {task:<30} Contribution: {contrib_mean:.2f} ± {contrib_std * 1.96:.2f} (CI95)")
            print("-" * 30)

    def _prepare_plot_data(self):
        plot_data = []
        for agent_name, metrics in self.metrics.items():
            # 1. Overall
            win_rate, win_std = metrics.get_win_rate()
            ksr, ksr_std = metrics.get_ksr()
            # Multiply by 1.96 for 95% Confidence Interval
            plot_data.append(
                {'agent': agent_name, 'metric': 'Win Rate', 'value': win_rate, 'CI95': win_std * 1.96, 'category': 'Overall'})
            plot_data.append(
                {'agent': agent_name, 'metric': 'Survival Rate', 'value': ksr, 'CI95': ksr_std * 1.96, 'category': 'Overall'})

            # 2. Voting
            irp, irp_std = metrics.get_irp()
            vss, vss_std = metrics.get_vss()
            plot_data.append(
                {'agent': agent_name, 'metric': 'IRP', 'value': irp, 'CI95': irp_std * 1.96, 'category': 'Voting Accuracy'})
            plot_data.append(
                {'agent': agent_name, 'metric': 'VSS', 'value': vss, 'CI95': vss_std * 1.96, 'category': 'Voting Accuracy'})

            # 3. Role Specific Win Rates
            for role in sorted(metrics.wins_by_role.keys()):
                role_rate, role_std = metrics.get_win_rate_for_role(role)
                plot_data.append({'agent': agent_name, 'metric': f'{role}', 'value': role_rate, 'CI95': role_std * 1.96,
                                  'category': 'Role-Specific Win Rate'})

            # 4. Role Specific Survival
            for role in sorted(metrics.survival_by_role.keys()):
                role_ksr, role_ksr_std = metrics.get_ksr_for_role(role)
                plot_data.append({'agent': agent_name, 'metric': f'{role}', 'value': role_ksr, 'CI95': role_ksr_std * 1.96,
                                  'category': 'Role-Specific Survival'})

            # 5. Ratings
            # We group Elo and TrueSkill into a single "Ratings" category for cleaner layout
            plot_data.append({'agent': agent_name, 'metric': 'Elo', 'value': metrics.elo, 'CI95': metrics.elo_std * 1.96,
                              'category': 'Ratings'})

            if OPENSKILL_AVAILABLE and metrics.openskill_rating:
                plot_data.append({'agent': agent_name, 'metric': 'TrueSkill', 'value': metrics.openskill_rating.mu,
                                  'CI95': metrics.openskill_mu_std * 1.96, 'category': 'Ratings'})

        return pd.DataFrame(plot_data)

    def plot_metrics(self, output_path: Union[str, List[str]] = "metrics.html"):
        if not PLOTLY_AVAILABLE:
            print("Warning: `plotly` not found. Cannot plot metrics.")
            return

        df = self._prepare_plot_data()

        category_order = [
            'Overall',
            'Voting Accuracy',
            'Role-Specific Win Rate',
            'Role-Specific Survival',
            'Ratings'
        ]
        present_categories = [cat for cat in category_order if cat in df['category'].unique()]

        max_cols = 0
        category_metrics_map = {}
        for cat in present_categories:
            metrics_in_cat = df[df['category'] == cat]['metric'].unique()
            if cat == 'Ratings':
                metrics_in_cat = sorted(metrics_in_cat, key=lambda x: 0 if x == 'Elo' else 1)
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
            horizontal_spacing=0.04
        )

        colors = _get_color_discrete_sequence(10)

        for row_idx, cat in enumerate(present_categories):
            metrics = category_metrics_map[cat]
            for col_idx, metric in enumerate(metrics):
                metric_data = df[(df['category'] == cat) & (df['metric'] == metric)]
                
                if cat == 'Ratings':
                    # Sort agents by value for rating plots
                    sorted_agents = metric_data.sort_values('value', ascending=True)['agent'].tolist()
                    fig.update_xaxes(categoryorder='array', categoryarray=sorted_agents, row=row_idx + 1, col=col_idx + 1)
                else:
                    sorted_agents = sorted(metric_data['agent'].unique())

                fig.add_trace(
                    go.Bar(
                        name=metric,
                        x=metric_data['agent'],
                        y=metric_data['value'],
                        error_y=dict(type='data', array=metric_data['CI95']),
                        marker_color=metric_data['agent'].apply(lambda x: colors[sorted_agents.index(x) % len(colors)]),
                        showlegend=False,
                        hovertemplate="<b>%{x}</b><br>%{y:.2f} ± %{error_y.array:.2f}<extra></extra>"
                    ),
                    row=row_idx + 1, col=col_idx + 1
                )

                if cat == 'Ratings':
                    fig.update_yaxes(matches=None, row=row_idx + 1, col=col_idx + 1)
                else:
                    fig.update_yaxes(range=[0, 1.05], row=row_idx + 1, col=col_idx + 1)
                fig.update_xaxes(tickangle=45, row=row_idx + 1, col=col_idx + 1)

        # Move Row Titles to Left
        fig.for_each_annotation(lambda a: a.update(
            x=-0.06, xanchor='right', font=dict(size=14, color="#111827", weight="bold"), yanchor='middle'
        ) if a.text in present_categories else None)

        fig.update_layout(
            title_text="Agent Performance Metrics",
            title_font_size=24,
            title_x=0.01,
            height=350 * len(present_categories),
            width=250 * max_cols if max_cols > 2 else 1000,
            font=dict(family="Inter, sans-serif"),
            showlegend=False,
            margin=dict(l=120, r=50)
        )

        _save_figure(fig, output_path, width=fig.layout.width, height=fig.layout.height)
        return fig

    def plot_gte_evaluation(self, output_path: Union[str, List[str]] = "gte_evaluation.html"):
        if not POLARIX_AVAILABLE or not PLOTLY_AVAILABLE:
            print("Warning: `polarix` or `plotly` library not found.")
            return None
        if self.gte_game is None:
            print("GTE evaluation has not been run. Please run .evaluate() first.")
            return None

        # --- 1. Data Preparation ---
        agents = sorted(list(self.metrics.keys()))
        tasks = self.gte_tasks

        ratings_mean = self.gte_ratings[0][1]
        ratings_std = self.gte_ratings[1][1]
        joint_mean = self.gte_joint[0]
        contributions_mean = self.gte_contributions_raw[0]

        agent_rating_map = {agent: ratings_mean[i] for i, agent in enumerate(agents)}
        sorted_agents = sorted(agents, key=lambda x: agent_rating_map[x])

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

        data = pd.DataFrame.from_dict({
            "agent": [rating_actions[i] for i in rating_actions_grid.ravel()],
            "metric": [contrib_actions[i] for i in contrib_actions_grid.ravel()],
            "contrib": contributions_mean.ravel(),
            "support": joint_support.ravel(),
        })

        agent_ratings_df = pd.DataFrame({
            "agent": agents,
            "rating": ratings_mean,
            "CI95": ratings_std * 1.96  # 95% CI
        })

        importance_df = pd.DataFrame({
            'metric': tasks,
            'importance': self.gte_marginals[0][0],
            'CI95': self.gte_marginals[1][0] * 1.96  # 95% CI
        }).sort_values('importance', ascending=True)

        # --- 2. Dynamic Layout Calculation ---
        n_top = len(agents) + 2
        n_bottom = len(tasks) + 2
        total_items = n_top + n_bottom

        h_top = max(0.3, min(0.7, n_top / total_items))
        h_bottom = 1.0 - h_top

        # --- 3. Build Plot ---
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[h_top, h_bottom],
            vertical_spacing=0.1,
            subplot_titles=("Win Rate Contributions & Equilibrium Ratings", "Task Importance")
        )

        # Contributions
        colors = _get_color_discrete_sequence(len(tasks))
        for i, metric in enumerate(tasks):
            subset = data[data['metric'] == metric]
            fig.add_trace(
                go.Bar(
                    name=metric,
                    y=subset['agent'],
                    x=subset['contrib'],
                    orientation='h',
                    legendgroup='metrics',
                    legendgrouptitle_text="Metrics",
                    marker_color=colors[i % len(colors)],
                    hovertemplate=f"<b>Metric: {metric}</b><br>Contrib: %{{x:.2%}}<extra></extra>"
                ),
                row=1, col=1
            )

        # Net Rating
        fig.add_trace(
            go.Scatter(
                name="Net Rating",
                y=agent_ratings_df['agent'],
                x=agent_ratings_df['rating'],
                mode='markers',
                marker=dict(symbol='diamond', size=12, color='black', line=dict(width=1.5, color='white')),
                error_x=dict(type='data', array=agent_ratings_df['CI95'], color='black', thickness=2),
                hovertemplate="<b>%{y}</b><br>Net Rating: %{x:.2%}<br>Std: %{error_x.array:.4f}<extra></extra>"
            ),
            row=1, col=1
        )

        # Task Importance
        fig.add_trace(
            go.Bar(
                name="Importance",
                y=importance_df['metric'],
                x=importance_df['importance'],
                orientation='h',
                marker_color='#6B7280',
                error_x=dict(type='data', array=importance_df['CI95'], color='black'),
                showlegend=False,
                hovertemplate="<b>%{y}</b><br>Importance: %{x:.2%}<extra></extra>"
            ),
            row=2, col=1
        )

        # --- 4. Layout Formatting ---
        fig.update_layout(
            barmode='relative',
            height=max(900, total_items * 35),
            width=1300,
            title_text="Game Theoretic Evaluation Results",
            title_font_size=24,
            bargap=0.15,
            legend=dict(
                yanchor="top", y=1,
                xanchor="left", x=1.02,
                orientation="v",
                tracegroupgap=20
            ),
            font=dict(family="Inter, sans-serif")
        )

        # Font Sizing
        fig.update_yaxes(
            categoryorder='array', categoryarray=sorted_agents,
            tickfont=dict(size=14, color="#1f2937"), row=1, col=1
        )
        fig.update_yaxes(
            tickfont=dict(size=14, color="#1f2937"), row=2, col=1
        )

        fig.update_xaxes(
            tickformat=".0%", title_text="Win Rate Contribution",
            tickfont=dict(size=14, color="#1f2937"), title_font=dict(size=16, color="#111827"),
            row=1, col=1, gridcolor='#F3F4F6'
        )
        fig.update_xaxes(
            title_text="Marginal Probability",
            tickfont=dict(size=14, color="#1f2937"), title_font=dict(size=16, color="#111827"),
            row=2, col=1, gridcolor='#F3F4F6'
        )
        fig.update_yaxes(gridcolor='#F3F4F6')

        _save_figure(fig, output_path, width=1300, height=fig.layout.height)
        return fig


if __name__ == '__main__':
    # Example usage:
    DIR_PATH = Path(os.path.dirname(os.path.abspath(__file__)))
    SMOKE_TEST_DATA_DIR = str(DIR_PATH / "test" / "data" / "w_replace")
    evaluator = GameSetEvaluator(SMOKE_TEST_DATA_DIR)
    evaluator.evaluate(gte_samples=2)
    evaluator.print_results()
    evaluator.plot_metrics(["metrics.html", "metrics.png"])
    chart = evaluator.plot_gte_evaluation(["gte.html", "gte.png"])