import functools
from collections import defaultdict
from typing import Dict, List, Tuple, Union
from pathlib import Path
import os

import numpy as np

try:
    import polarix as plx

    POLARIX_AVAILABLE = True
except ImportError:
    POLARIX_AVAILABLE = False

try:
    import altair as alt
    import pandas as pd
    import chex
    import jax.numpy as jnp
    ALTAIR_AVAILABLE = True
except ImportError:
    ALTAIR_AVAILABLE = False


from kaggle_environments.envs.werewolf.eval.loaders import get_games, GameResult


def _mean_std(values: List[float]) -> Tuple[float, float]:
    """Helper to calculate mean and standard deviation, handling empty lists."""
    if not values:
        return 0.0, 0.0
    return np.mean(values), np.std(values)


# --- Plotting utils from game_theoretic.py ---

_HEIGHT = 25
_WIDTH = 600


def _gte_rating_contribution_chart(
    game,
    joint: chex.Array,
    contributions: chex.Array,
    *,
    rating_player: int,
    contrib_player: int,
    rating_metadata: "pd.DataFrame | None" = None,
    contrib_metadata: "pd.DataFrame | None" = None,
    top_k: int = 100,
) -> "alt.Chart":
  """Plots the rating contribution of a player to another player's ratings."""
  if game.players is None:
    raise ValueError("Game must have player names explicitly defined.")

  rating_name = game.players[rating_player]
  contrib_name = game.players[contrib_player]
  rating_actions = game.actions[rating_player]
  contrib_actions = game.actions[contrib_player]

  if rating_metadata is not None:
    nunique_by_rating_name = rating_metadata.groupby(rating_name).nunique(False)
    if np.any(nunique_by_rating_name != 1):
      raise ValueError(
          "Rating metadata must be unique per rating action, but is not"
          f" ({nunique_by_rating_name.reset_index()})."
      )

  joint_support = jnp.sum(
      jnp.moveaxis(joint, (rating_player, contrib_player), (0, 1)),
      axis=tuple(range(2, len(joint.shape))),
  )  # [num_rating_actions, num_contrib_actions]
  rating_actions_grid, contrib_actions_grid = np.meshgrid(
      jnp.arange(len(game.actions[rating_player])),
      jnp.arange(len(game.actions[contrib_player])),
      indexing="ij",
  )

  data = pd.DataFrame.from_dict({
      game.players[rating_player]: rating_actions_grid.ravel(),
      game.players[contrib_player]: contrib_actions_grid.ravel(),
      "contrib": contributions.ravel(),
      "support": joint_support.ravel(),
  })

  # Computes rating player's action ratings to order rows by.
  sorted_rating_actions = (
      (data[[rating_name, "contrib"]].groupby(rating_name).sum().reset_index())
      .sort_values(by="contrib", ascending=False)[rating_name]
      .values
  )

  # Computes the height of the rating chart.
  num_actions = len(rating_actions)
  top_k = min(top_k, num_actions)
  num_actions_to_display = min(num_actions, top_k)
  height = num_actions_to_display * _HEIGHT

  sorted_rating_actions = sorted_rating_actions[:top_k]
  data = data[data[rating_name].isin(sorted_rating_actions)]
  data[rating_name] = data[rating_name].apply(lambda a: rating_actions[a])
  data[contrib_name] = data[contrib_name].apply(lambda a: contrib_actions[a])
  sorted_rating_actions = [rating_actions[a] for a in sorted_rating_actions]

  # Interval for selecting a subset of rating actions.
  ratings_data = (
      data[[rating_name, "contrib", "support"]]
      .groupby(rating_name)
      .sum()
      .reset_index()
  )

  if rating_metadata is not None:
    ratings_data = pd.merge(
        ratings_data,
        rating_metadata,
        on=rating_name,
        how="left",
        suffixes=(None, "_metadata"),
        validate="many_to_one",
    )
  ratings_data = ratings_data.rename(columns={"contrib": "rating"})
  ratings_data["rating_str"] = ratings_data["rating"].apply(lambda c: f"{c:.2%}")
  ratings_data["rating_ci_str"] = ratings_data["rating_ci"].apply(lambda c: f"{c:.2%}")

  if contrib_metadata is not None:
    merge_on = contrib_name
    if rating_name in contrib_metadata.columns:
      merge_on = (rating_name, contrib_name)
    data = pd.merge(
        data,
        contrib_metadata,
        on=merge_on,
        how="left",
        suffixes=(None, "_metadata"),
        validate="many_to_one",
    )

  color = alt.Color(
      f"{contrib_name}:N",
      scale=alt.Scale(scheme="category10"),
      legend=alt.Legend(title=contrib_name.capitalize()),
  )
  grouping = list(
      filter(lambda c: c not in ["contrib", "support"], data.columns)
  )

  x = alt.X(
      "sum(contrib):Q",
      title=(
          "Win (loss) rate contribution to agent ratings (stars), broken down"
          " by role"
      ),
      axis=alt.Axis(grid=False, format="%"),
  )
  y = alt.Y(
      f"{rating_name}:N",
      sort=sorted_rating_actions,
      title=None,
      axis=alt.Axis(
          grid=True,
          labels=True,
      ),
  )
  data = data[[*grouping, "contrib"]].groupby(grouping).sum().reset_index()

  data["contrib_str"] = data["contrib"].apply(lambda c: f"{c:.2%}")

  category_chart = alt.Chart(data)
  bars = category_chart.mark_bar().encode(
      x=x,
      y=y,
      color=color,
      tooltip=[
          alt.Tooltip(rating_name),
          alt.Tooltip(contrib_name),
          alt.Tooltip("contrib_with_ci:N", title="Role contribution"),
      ],
  ).transform_calculate(
      contrib_with_ci=alt.datum.contrib_str + " (± " + alt.datum.contrib_ci + ")"
  )

  # Order pos and neg contributions separately to address vega/altair bug.
  bars = alt.layer(
      bars.transform_filter(alt.datum.contrib < 0).encode(
          order=alt.Order("contrib:Q", sort="descending"),
      ),
      bars.transform_filter(alt.datum.contrib >= 0).encode(
          order=alt.Order("contrib:Q", sort="ascending"),
      ),
  ).resolve_scale(color="shared")
  rule = (
      alt.Chart(pd.DataFrame({"x": [1e-4]}))
      .mark_rule(opacity=0.5, size=1, strokeDash=[2, 2])
      .encode(x="x:Q")
  )

  overlay_points = (
      alt.Chart(ratings_data)
      .mark_point(
          shape=(
              "M0,.5L.6,.8L.5,.1L1,-.3L.3,-.4L0,-1L-.3,-.4L-1,-.3L-.5,.1L-.6,.8L0,.5Z"
          ),
          stroke="black",
          fill="gold",
          size=200,
          strokeWidth=2,
      )
      .encode(
          y=alt.Y(
              f"{rating_name}:N",
              sort=sorted_rating_actions,
              title=None,
              axis=alt.Axis(labels=True, grid=True),
          ),
          x=alt.X(
              "rating:Q",
              title=alt.Undefined,
              axis=alt.Axis(grid=True),
          ),
          tooltip=[
              alt.Tooltip(rating_name),
              alt.Tooltip(
                  'ratings_with_ci:N',
                  title="Relative winrate vs equilibrium",
              ),
              alt.Tooltip(
                  "support",
                  format=".2%",
                  title="Probability of play at equilibrium",
              ),
          ],
      ).transform_calculate(
          ratings_with_ci=alt.datum.rating_str + " (± " + alt.datum.rating_ci_str + ")"
      )
  )

  overlay_points_ci = (
      alt.Chart(ratings_data).mark_errorbar(
          color="black", ticks=True, size=0.8 * height // num_actions
      )
      .encode(
          x=alt.X("ratings_lo:Q"),
          x2=alt.X2("ratings_hi:Q"),
          y=alt.Y(
              f"{rating_name}:N",
              sort=sorted_rating_actions,
              title=None,
              axis=alt.Axis(labels=True, grid=True),
          ),
          strokeWidth=alt.value(1),
          tooltip=alt.Tooltip('ratings_with_ci:N'),
      )
      .transform_calculate(
          ratings_lo="datum.rating - datum.rating_ci",
          ratings_hi="datum.rating + datum.rating_ci",
          ratings_with_ci=alt.datum.rating_str + " (± " + alt.datum.rating_ci_str + ")"
      )
      .properties(height=height)
  )

  chart = (
      (
          alt.layer(bars, rule, overlay_points_ci, overlay_points)
          .resolve_scale(x="shared", y="shared", color="independent")
          .properties(
              width=_WIDTH,
              height=height,
              title=alt.TitleParams(
                  "Game-theoretic Ratings: win probabilities against the"
                  " equilibrium strategy",
                  subtitle=(
                      "An equilibrium is a mixture of current best agents and"
                      " most discriminative roles."
                  ),
              ),
          )
      )
  )

  return chart


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

    def __init__(self, input_dir: Union[str, List[str]], gte_tasks: List[str] = None,
                 preserve_full_game_records: bool = False):
        """Initializes the evaluator with a set of games and GTE tasks.

        Args:
            input_dir: The path to the directory or a list of paths to directories
                containing game replay JSONs.
            gte_tasks: A list of strings specifying which metrics to include in the
                Game Theoretic Evaluation. If None, a default set of tasks is used.
                Available options are:
                - 'WinRate-{role}': Win rate for a specific role (e.g., 'WinRate-Doctor').
                - 'KSR-{role}': Key Role Survival rate for a specific role.
                - 'KSR': Overall survival rate across all roles.
                - 'IRP': Identification Precision score.
                - 'VSS': Voting Success Score.
            preserve_full_game_records: If True, the full JSON for each game
                is stored in memory. Defaults to False.
        """
        if isinstance(input_dir, str):
            input_dirs = [input_dir]
        else:
            input_dirs = input_dir

        all_games = []
        for directory in input_dirs:
            all_games.extend(get_games(directory))

        self.games = [GameResult(g, preserve_full_record=preserve_full_game_records) for g in all_games]
        self.metrics: Dict[str, AgentMetrics] = defaultdict(lambda: AgentMetrics(agent_name=None))

        # For GTE
        self.gte_game = None
        self.gte_joint = None
        self.gte_ratings = None
        self.gte_contributions_raw = None

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

        ratings, joints, _, contributions, self.gte_game = self._bootstrap_stats(
            rnd, self.games, agents, self.gte_tasks, num_samples=num_samples
        )
        self.gte_ratings = ratings
        self.gte_joint = joints
        self.gte_contributions_raw = contributions

        ratings_mean, ratings_std = self.gte_ratings
        contributions_mean, contributions_std = self.gte_contributions_raw

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
        joints_mean = np.mean(joints, axis=0)
        joints_std = np.std(joints, axis=0)
        contributions_mean = np.mean(contributions, axis=0)
        contributions_std = np.std(contributions, axis=0)
        return (ratings_mean, ratings_std), (joints_mean, joints_std), None, (contributions_mean, contributions_std), games[0]

    def plot_gte_evaluation(self, top_k: int = 100):
        if not POLARIX_AVAILABLE or not ALTAIR_AVAILABLE:
            print("Warning: `polarix` or `altair` library not found. Cannot plot GTE results.")
            return None
        if self.gte_game is None:
            print("GTE evaluation has not been run. Please run .evaluate() first.")
            return None

        agents = sorted(list(self.metrics.keys()))
        tasks = self.gte_tasks

        ratings_mean, ratings_std = self.gte_ratings
        # GTE returns ratings for task player, agent player 1, agent player 2. We want agent player 1.
        ratings_ci = ratings_std[1] * 1.96

        joint_avg, _ = self.gte_joint

        contribution_avg, contribution_std = self.gte_contributions_raw
        contribution_ci = contribution_std * 1.96

        contribs_metadata_dict = {"agent": [], "metric": [], "contrib_ci": []}
        for i, model in enumerate(agents):
            for j, task in enumerate(tasks):
                ci = contribution_ci[i, j]
                contribs_metadata_dict["agent"].append(model)
                contribs_metadata_dict["metric"].append(task)
                contribs_metadata_dict["contrib_ci"].append(f"{ci:.2%}")

        ratings_metadata_dict = {"agent": [], "rating_ci": []}
        for i, model in enumerate(agents):
            ci = ratings_ci[i]
            ratings_metadata_dict["agent"].append(model)
            ratings_metadata_dict["rating_ci"].append(ci)

        # The game object has 'metric' and 'agent' as players.
        # rating_player=1 is 'agent', contrib_player=0 is 'metric'.
        chart = _gte_rating_contribution_chart(
            game=self.gte_game,
            joint=joint_avg,
            contributions=contribution_avg,
            contrib_metadata=pd.DataFrame(contribs_metadata_dict),
            rating_metadata=pd.DataFrame(ratings_metadata_dict),
            rating_player=1,
            contrib_player=0,
            top_k=top_k)

        return chart

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
    DIR_PATH = Path(os.path.dirname(os.path.abspath(__file__)))
    SMOKE_TEST_DATA_DIR = str(DIR_PATH / "test" / "data" / "w_replace")
    evaluator = GameSetEvaluator(SMOKE_TEST_DATA_DIR)
    evaluator.evaluate()
    evaluator.print_results()
    chart = evaluator.plot_gte_evaluation()
    if chart:
        chart.save("gte_evaluation.html")
        print("\nGTE evaluation chart saved to gte_evaluation.html")
