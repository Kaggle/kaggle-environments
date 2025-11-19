import functools
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np

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
    import altair as alt
    import pandas as pd
    import chex
    import jax.numpy as jnp

    ALTAIR_AVAILABLE = True
except ImportError:
    ALTAIR_AVAILABLE = False

from kaggle_environments.envs.werewolf.eval.loaders import get_game_results
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
        self.gte_contributions: Dict[str, Tuple[float, float]] = defaultdict(lambda: (0.0, 0.0))

        # Ratings
        self.elo: float = 1200.0
        self.elo_std: float = 0.0
        self.openskill_model = openskill_model
        self.openskill_rating = None

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
            # Default tasks for GTE
            roles = sorted(list(set(p.role.name for g in self.games for p in g.players)))
            self.gte_tasks = ([f'WinRate-{r}' for r in roles] +
                              [f'KSR-{r}' for r in roles] +
                              ['IRP', 'VSS', 'KSR'])
        else:
            self.gte_tasks = gte_tasks

    def _compute_elo_ratings(self, games: List) -> Dict[str, float]:
        """Computes Elo ratings for a given sequence of games."""
        elos = defaultdict(lambda: 1200.0)

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

    def _bootstrap_elo(self, num_samples=100):
        """Estimates Elo standard error via bootstrapping."""
        if not self.games:
            return

        rnd = np.random.default_rng(42)
        bootstrapped_elos = defaultdict(list)

        # We need to know all agents to initialize lists, in case an agent isn't picked in a sample
        all_agents = list(self.metrics.keys())
        
        for _ in range(num_samples):
            sampled_games = rnd.choice(self.games, size=len(self.games), replace=True)
            sample_elos = self._compute_elo_ratings(sampled_games)
            
            for agent in all_agents:
                # If agent wasn't in the sample, they stay at 1200 (or we could skip, 
                # but sticking to 1200 might bias if they rarely play. 
                # Better to track only if they played, but for simplicity we assume 1200).
                # However, typically we only care about variance of active play.
                # Let's use the calculated value or 1200 default.
                bootstrapped_elos[agent].append(sample_elos.get(agent, 1200.0))

        for agent, values in bootstrapped_elos.items():
            if len(values) > 1:
                self.metrics[agent].elo_std = float(np.std(values, ddof=1))

    def evaluate(self, gte_samples=3, elo_samples=100):
        """Processes all games and aggregates the metrics."""
        for game in self.games:
            # --- Win Rate & Survival Metrics ---
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

            # --- Voting Accuracy Metrics ---
            irp_results, vss_results = game.iterate_voting_mini_game()
            for agent_name, score in irp_results:
                self.metrics[agent_name].irp_scores.append(score)
            for agent_name, score in vss_results:
                self.metrics[agent_name].vss_scores.append(score)

        # --- Rating Updates (Point Estimates) ---
        # 1. Elo
        final_elos = self._compute_elo_ratings(self.games)
        for agent, rating in final_elos.items():
            self.metrics[agent].elo = rating
        
        # 2. TrueSkill (OpenSkill)
        # OpenSkill is order dependent too, but we just run it once sequentially here.
        if OPENSKILL_AVAILABLE and self.openskill_model:
             for game in self.games:
                villager_agents = []
                werewolf_agents = []
                for player in game.players:
                    agent_name = player.agent.display_name
                    if player.role.team == Team.VILLAGERS:
                        villager_agents.append(agent_name)
                    else:
                        werewolf_agents.append(agent_name)
                
                team_v = [self.metrics[a].openskill_rating for a in villager_agents]
                team_w = [self.metrics[a].openskill_rating for a in werewolf_agents]

                teams = None
                if game.winner_team == Team.VILLAGERS:
                    teams = [team_v, team_w]
                elif game.winner_team == Team.WEREWOLVES:
                    teams = [team_w, team_v]

                if teams:
                    new_ratings = self.openskill_model.rate(teams)
                    openskill_ratings = [rate for team in new_ratings for rate in team]
                    for rating in openskill_ratings:
                        self.metrics[rating.name].openskill_rating = rating

        # --- Bootstrapping for Errors ---
        self._bootstrap_elo(num_samples=elo_samples)
        self._run_gte_evaluation(num_samples=gte_samples)

    def _run_gte_evaluation(self, num_samples: int):
        if not POLARIX_AVAILABLE:
            print("Warning: `polarix` library not found. Skipping Game Theoretic Evaluation.")
            return

        agents = sorted(list(self.metrics.keys()))
        rnd = np.random.default_rng(42)

        ratings, joints, marginals, contributions, self.gte_game = self._bootstrap_stats(
            rnd, self.games, agents, self.gte_tasks, num_samples=num_samples
        )
        self.gte_ratings = ratings
        self.gte_joint = joints
        self.gte_marginals = marginals
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
                    if len(scores) > 1:
                        stddev_matrix[i, j] = np.std(scores, ddof=1) / np.sqrt(len(scores))
                    else:
                        stddev_matrix[i, j] = 0.0

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

        marginals_by_dim = list(zip(*marginals))
        marginals_mean = [np.mean(m, axis=0) for m in marginals_by_dim]
        marginals_std = [np.std(m, axis=0) for m in marginals_by_dim]

        contributions_mean = np.mean(contributions, axis=0)
        contributions_std = np.std(contributions, axis=0)
        return (ratings_mean, ratings_std), (joints_mean, joints_std), (marginals_mean, marginals_std), (contributions_mean, contributions_std), \
        games[0]

    def plot_gte_evaluation(self, top_k: int = 100, output_path="gte_evaluation.html"):
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
        rating_chart = _gte_rating_contribution_chart(
            game=self.gte_game,
            joint=joint_avg,
            contributions=contribution_avg,
            contrib_metadata=pd.DataFrame(contribs_metadata_dict),
            rating_metadata=pd.DataFrame(ratings_metadata_dict),
            rating_player=1,
            contrib_player=0,
            top_k=top_k)

        # --- Chart 2: Task Importance (Marginal Probability) ---
        # marginals[0] is for Player 0 (Metric/Task)
        # self.gte_marginals is (marginals_mean, marginals_std)
        # marginals_mean is [mean_p0, mean_p1], marginals_std is [std_p0, std_p1]
        task_marginals_mean = self.gte_marginals[0][0]
        task_marginals_std = self.gte_marginals[1][0]

        task_importance_df = pd.DataFrame({
            'metric': tasks,
            'importance': task_marginals_mean,
            'std': task_marginals_std
        })
        # 95% CI
        task_importance_df['ci'] = task_importance_df['std'] * 1.96
        task_importance_df['min_val'] = task_importance_df['importance'] - task_importance_df['ci']
        task_importance_df['max_val'] = task_importance_df['importance'] + task_importance_df['ci']

        base = alt.Chart(task_importance_df).encode(
            y=alt.Y('metric:N', sort='-x', title="Task"),
            x=alt.X('importance:Q', title="Marginal Probability (Importance)"),
        )

        bars = base.mark_bar().encode(
            tooltip=['metric', 'importance', 'std']
        )

        error_bars = alt.Chart(task_importance_df).mark_rule(color='black').encode(
            y=alt.Y('metric:N', sort='-x'),
            x=alt.X('min_val:Q'),
            x2=alt.X2('max_val:Q'),
            tooltip=['metric', 'importance', 'std']
        )

        importance_chart = (bars + error_bars).properties(
            title="Task Importance (Marginal Probability in Equilibrium)"
        )

        final_chart = alt.vconcat(rating_chart, importance_chart).resolve_scale(color='independent')

        if output_path:
            final_chart.save(output_path)
        return final_chart

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

            print("  Ratings:")
            print(f"    Elo: {stats.elo:.2f}")
            if OPENSKILL_AVAILABLE and stats.openskill_rating:
                print(f"    TrueSkill: mu={stats.openskill_rating.mu:.2f}, sigma={stats.openskill_rating.sigma:.2f}")

            if POLARIX_AVAILABLE:
                print("  Game Theoretic Evaluation (GTE):")
                gte_mean, gte_std = stats.gte_rating
                print(f"    Overall GTE Rating: {gte_mean:.2f} ± {gte_std:.2f}")
                for task in self.gte_tasks:
                    contrib_mean, contrib_std = stats.gte_contributions[task]
                    print(f"    - {task:<30} Contribution: {contrib_mean:.2f} ± {contrib_std:.2f}")

            print("-" * 30)

    def _prepare_plot_data(self):
        plot_data = []
        for agent_name, metrics in self.metrics.items():
            # Overall metrics
            win_rate, win_std = metrics.get_win_rate()
            ksr, ksr_std = metrics.get_ksr()
            irp, irp_std = metrics.get_irp()
            vss, vss_std = metrics.get_vss()
            plot_data.append(
                {'agent': agent_name, 'metric': 'Win Rate', 'value': win_rate, 'std': win_std, 'category': 'Overall'})
            plot_data.append({'agent': agent_name, 'metric': 'Survival Rate (KSR)', 'value': ksr, 'std': ksr_std,
                              'category': 'Overall'})
            plot_data.append(
                {'agent': agent_name, 'metric': 'IRP', 'value': irp, 'std': irp_std, 'category': 'Voting Accuracy'})
            plot_data.append(
                {'agent': agent_name, 'metric': 'VSS', 'value': vss, 'std': vss_std, 'category': 'Voting Accuracy'})

            # Role-specific metrics
            for role in sorted(metrics.wins_by_role.keys()):
                role_rate, role_std = metrics.get_win_rate_for_role(role)
                plot_data.append(
                    {'agent': agent_name, 'metric': f'Win Rate ({role})', 'value': role_rate, 'std': role_std,
                     'category': 'Role-Specific Win Rate'})
            for role in sorted(metrics.survival_by_role.keys()):
                role_ksr, role_ksr_std = metrics.get_ksr_for_role(role)
                plot_data.append(
                    {'agent': agent_name, 'metric': f'KSR ({role})', 'value': role_ksr, 'std': role_ksr_std,
                     'category': 'Role-Specific Survival'})

            # Ratings
            plot_data.append(
                {'agent': agent_name, 'metric': 'Elo', 'value': metrics.elo, 'std': metrics.elo_std, 'category': 'Elo Rating'})
            if OPENSKILL_AVAILABLE and metrics.openskill_rating:
                plot_data.append(
                    {'agent': agent_name, 'metric': 'TrueSkill (mu)', 'value': metrics.openskill_rating.mu,
                     'std': metrics.openskill_rating.sigma, 'category': 'TrueSkill Rating'})

        return pd.DataFrame(plot_data)

    def plot_metrics(self, output_path="metrics.html"):
        if not ALTAIR_AVAILABLE:
            print("Warning: `altair` and `pandas` not found. Cannot plot metrics.")
            return

        df = self._prepare_plot_data()

        charts = []
        for category in df['category'].unique():
            chart_data = df[df['category'] == category]

            base = alt.Chart(chart_data).encode(
                x=alt.X('agent:N', title="Agent"),
                y=alt.Y('value:Q', title="Score", scale=alt.Scale(zero=False)),
                color='agent:N'
            )

            bars = base.mark_bar().encode(
                tooltip=['agent', 'metric', 'value', 'std']
            )

            error_bars = base.mark_errorbar(extent='ci').encode(
                y='value:Q',
                yError='std:Q',
                color=alt.value('black')
            )

            chart = (bars + error_bars).properties(
                title=category
            ).facet(
                column=alt.Column('metric:N', title=None)
            )
            charts.append(chart)

        if charts:
            final_chart = alt.vconcat(*charts).resolve_scale(
                color='shared'
            )
            final_chart.save(output_path)
            print(f"Metrics chart saved to {output_path}")


if __name__ == '__main__':
    # Example usage:
    DIR_PATH = Path(os.path.dirname(os.path.abspath(__file__)))
    SMOKE_TEST_DATA_DIR = str(DIR_PATH / "test" / "data" / "w_replace")
    evaluator = GameSetEvaluator(SMOKE_TEST_DATA_DIR)
    evaluator.evaluate(gte_samples=2)
    evaluator.print_results()
    evaluator.plot_metrics()
    chart = evaluator.plot_gte_evaluation()
    if chart:
        chart.save("gte_evaluation.html")
        print("\nGTE evaluation chart saved to gte_evaluation.html")
