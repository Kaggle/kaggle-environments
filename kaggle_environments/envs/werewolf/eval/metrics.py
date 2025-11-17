import functools
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

try:
    import polarix as plx

    POLARIX_AVAILABLE = True
except ImportError:
    POLARIX_AVAILABLE = False

from kaggle_environments.envs.werewolf.eval.loaders import get_games, GameResult


def _mean_std(values: List[float]) -> Tuple[float, float]:
    """Helper to calculate mean and standard deviation, handling empty lists."""
    if not values:
        return 0.0, 0.0
    return np.mean(values), np.std(values)


class AgentMetrics:
    """Stores and calculates performance metrics for a single agent."""

    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.wins: List[int] = []
        self.wins_by_role: Dict[str, List[int]] = defaultdict(list)
        self.irp_scores: List[int] = []
        self.vss_scores: List[int] = []
        self.survival_scores: List[int] = []
        self.survival_by_role: Dict[str, List[int]] = defaultdict(list)

        # For GTE
        self.gte_rating: Tuple[float, float] = (0.0, 0.0)
        self.gte_contributions: Dict[str, Tuple[float, float]] = defaultdict(lambda: (0.0, 0.0))

    def get_win_rate(self) -> Tuple[float, float]:
        return _mean_std(self.wins)

    def get_win_rate_for_role(self, role: str) -> Tuple[float, float]:
        return _mean_std(self.wins_by_role.get(role, []))

    def get_irp(self) -> Tuple[float, float]:
        return _mean_std(self.irp_scores)

    def get_vss(self) -> Tuple[float, float]:
        return _mean_std(self.vss_scores)

    def get_ksr(self) -> Tuple[float, float]:
        return _mean_std(self.survival_scores)

    def get_ksr_for_role(self, role: str) -> Tuple[float, float]:
        return _mean_std(self.survival_by_role.get(role, []))


class GameSetEvaluator:
    """Evaluates a set of game replays and calculates metrics for each agent."""

    def __init__(self, input_dir: str, gte_tasks: List[str] = None):
        """Initializes the evaluator with a set of games and GTE tasks.

        Args:
            input_dir: The path to the directory containing game replay JSONs.
            gte_tasks: A list of strings specifying which metrics to include in the
                Game Theoretic Evaluation. If None, a default set of tasks is used.
                Available options are:
                - 'WinRate-{role}': Win rate for a specific role (e.g., 'WinRate-Doctor').
                - 'KSR-{role}': Key Role Survival rate for a specific role.
                - 'KSR': Overall survival rate across all roles.
                - 'IRP': Identification Precision score.
                - 'VSS': Voting Success Score.
        """
        self.games = [GameResult(g) for g in get_games(input_dir)]
        self.metrics: Dict[str, AgentMetrics] = defaultdict(lambda: AgentMetrics(agent_name=None))

        if gte_tasks is None:
            # Default tasks for GTE
            roles = sorted(list(set(p.role.name for g in self.games for p in g.players)))
            self.gte_tasks = ([f'WinRate-{r}' for r in roles] +
                              [f'KSR-{r}' for r in roles] +
                              ['IRP', 'VSS', 'KSR'])
        else:
            self.gte_tasks = gte_tasks

    def evaluate(self, gte_samples=100):
        """Processes all games and aggregates the metrics."""
        for game in self.games:
            # --- Win Rate & Survival Metrics ---
            for player in game.players:
                agent_name = player.agent.display_name
                if self.metrics[agent_name].agent_name is None:
                    self.metrics[agent_name].agent_name = agent_name

                won = 1 if player.role.team == game.winner_team else 0
                survived = 1 if player.alive else 0

                self.metrics[agent_name].wins.append(won)
                self.metrics[agent_name].wins_by_role[player.role.name].append(won)
                self.metrics[agent_name].survival_scores.append(survived)
                self.metrics[agent_name].survival_by_role[player.role.name].append(survived)

            # --- Voting Accuracy Metrics ---
            irp_results, vss_results = game.iterate_voting_mini_game()
            for agent_name, score in irp_results:
                self.metrics[agent_name].irp_scores.append(score)
            for agent_name, score in vss_results:
                self.metrics[agent_name].vss_scores.append(score)

        self._run_gte_evaluation(num_samples=gte_samples)

    def _run_gte_evaluation(self, num_samples: int):
        if not POLARIX_AVAILABLE:
            print("Warning: `polarix` library not found. Skipping Game Theoretic Evaluation.")
            return

        agents = sorted(list(self.metrics.keys()))
        rnd = np.random.default_rng(42)

        ratings, _, _, contributions, _ = self._bootstrap_stats(
            rnd, self.games, agents, self.gte_tasks, num_samples=num_samples
        )

        ratings_mean, ratings_std = ratings
        contributions_mean, contributions_std = contributions

        for i, agent_name in enumerate(agents):
            self.metrics[agent_name].gte_rating = (ratings_mean[1][i], ratings_std[1][i])
            for j, task_name in enumerate(self.gte_tasks):
                self.metrics[agent_name].gte_contributions[task_name] = (
                    contributions_mean[i, j], contributions_std[i, j])

    @staticmethod
    def _bootstrap_solve(rnd, games, agents, tasks):
        # Convert lists to sets for O(1) lookups
        agent_set = set(agents)
        task_set = set(tasks)

        sampled_games = rnd.choice(games, size=len(games), replace=True)
        agent_scores = {agent: {task: [] for task in tasks} for agent in agents}

        for game in sampled_games:
            for player in game.players:
                agent_name = player.agent.display_name
                if agent_name in agent_set:
                    role_name = player.role.name
                    # Win Rate & KSR
                    win_rate_task = f'WinRate-{role_name}'
                    if win_rate_task in task_set:
                        agent_scores[agent_name][win_rate_task].append(1 if player.role.team == game.winner_team else 0)
                    ksr_task = f'KSR-{role_name}'
                    if ksr_task in task_set:
                        agent_scores[agent_name][ksr_task].append(1 if player.alive else 0)
                    if 'KSR' in task_set:
                        agent_scores[agent_name]['KSR'].append(1 if player.alive else 0)

            irp_results, vss_results = game.iterate_voting_mini_game()
            if 'IRP' in task_set:
                for agent_name, score in irp_results:
                    if agent_name in agent_set:
                        agent_scores[agent_name]['IRP'].append(score)
            if 'VSS' in task_set:
                for agent_name, score in vss_results:
                    if agent_name in agent_set:
                        agent_scores[agent_name]['VSS'].append(score)

        mean_matrix = np.zeros((len(agents), len(tasks)))
        stddev_matrix = np.zeros((len(agents), len(tasks)))
        for i, agent in enumerate(agents):
            for j, task in enumerate(tasks):
                scores = agent_scores[agent][task]
                if scores:
                    mean_matrix[i, j] = np.mean(scores)
                    stddev_matrix[i, j] = np.std(scores)

        for j in range(mean_matrix.shape[1]):  # Iterate over tasks (columns)
            # If all agents have the same score for a task, add noise to avoid ptp=0 error
            if np.ptp(mean_matrix[:, j]) < 1e-9:
                mean_matrix[:, j] += rnd.random(mean_matrix.shape[0]) * 1e-6
                stddev_matrix[:, j] += rnd.random(stddev_matrix.shape[0]) * 1e-6

        game = plx.agent_vs_task_game(
            agents=agents, tasks=tasks, agent_vs_task=mean_matrix, agent_vs_task_stddev=stddev_matrix,
            task_player='metric', normalizer='winrate'
        )
        res = plx.solve(game, plx.ce_maxent, disable_progress_bar=True)
        marginals = plx.marginals_from_joint(res.joint)
        contributions = plx.joint_payoffs_contribution(
            game.payoffs, res.joint, rating_player=1, contrib_player=0
        )
        return res.ratings, res.joint, marginals, contributions, game

    def _bootstrap_stats(self, rnd, games, agents, tasks, num_samples=10):
        rnds = [np.random.default_rng(s) for s in rnd.integers(1_000_000, size=num_samples)]
        solve_func = functools.partial(self._bootstrap_solve, games=games, agents=agents, tasks=tasks)
        res = list(map(solve_func, rnds))

        ratings, joints, marginals, contributions, games = zip(*res)
        ratings_mean = [np.mean(r, axis=0) for r in zip(*ratings)]
        ratings_std = [np.std(r, axis=0) for r in zip(*ratings)]
        contributions_mean = np.mean(contributions, axis=0)
        contributions_std = np.std(contributions, axis=0)
        return (ratings_mean, ratings_std), None, None, (contributions_mean, contributions_std), games[0]

    def print_results(self):
        """Prints a formatted summary of the evaluation results."""
        sorted_metrics = sorted(self.metrics.values(), key=lambda m: m.agent_name)

        for stats in sorted_metrics:
            print(f"Agent: {stats.agent_name}")

            win_rate, win_std = stats.get_win_rate()
            print(f"  Overall Win Rate: {win_rate:.2f} ± {win_std:.2f} ({len(stats.wins)} games)")

            ksr, ksr_std = stats.get_ksr()
            print(f"  Overall Survival Rate: {ksr:.2f} ± {ksr_std:.2f}")

            print("  Role-Specific Win Rates:")
            for role in sorted(stats.wins_by_role.keys()):
                role_rate, role_std = stats.get_win_rate_for_role(role)
                game_count = len(stats.wins_by_role[role])
                print(f"    {role:<10}: {role_rate:.2f} ± {role_std:.2f} ({game_count} games)")

            print("  Role-Specific Survival Rates (KSR):")
            for role in sorted(stats.survival_by_role.keys()):
                role_ksr, role_ksr_std = stats.get_ksr_for_role(role)
                game_count = len(stats.survival_by_role[role])
                print(f"    {role:<10}: {role_ksr:.2f} ± {role_ksr_std:.2f} ({game_count} games)")

            irp, irp_std = stats.get_irp()
            vss, vss_std = stats.get_vss()
            print("  Voting Accuracy (Villager Team):")
            print(f"    IRP (Identification Precision): {irp:.2f} ± {irp_std:.2f} ({len(stats.irp_scores)} votes)")
            print(f"    VSS (Voting Success Score):     {vss:.2f} ± {vss_std:.2f} ({len(stats.vss_scores)} votes)")

            if POLARIX_AVAILABLE:
                print("  Game Theoretic Evaluation (GTE):")
                gte_mean, gte_std = stats.gte_rating
                print(f"    Overall GTE Rating: {gte_mean:.2f} ± {gte_std:.2f}")
                for task in self.gte_tasks:
                    contrib_mean, contrib_std = stats.gte_contributions[task]
                    print(f"    - {task:<30} Contribution: {contrib_mean:.2f} ± {contrib_std:.2f}")

            print("-" * 30)


if __name__ == '__main__':
    # Example usage:
    evaluator = GameSetEvaluator("kaggle_environments/envs/werewolf/eval/test/data/w_replace")
    evaluator.evaluate()
    evaluator.print_results()
