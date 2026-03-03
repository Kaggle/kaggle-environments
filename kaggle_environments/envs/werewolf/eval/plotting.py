import os
# Workaround for broken google.colab import
import sys
from typing import Union, List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
# Plotting imports
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns
from kaggle_environments.envs.werewolf.game.consts import Team
from matplotlib.gridspec import GridSpec
from plotly.subplots import make_subplots

try:
    import google.colab
except AttributeError:
    sys.modules["google.colab"] = None
except ImportError:
    pass

pio.templates.default = "plotly_white"

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})
sns.set_theme(style="ticks", palette="colorblind")


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


import hashlib
import re
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def _canonical_name(name: str) -> str:
    return re.sub(r"\s+", " ", str(name).strip())

def _make_big_palette():
    cols = []
    import seaborn as sns
    cols.extend(sns.color_palette("colorblind", 10))
    cols.extend(sns.color_palette("husl", 10))
    for cmap_name in ["tab20", "tab20b", "tab20c"]:
        cmap = plt.get_cmap(cmap_name)
        cols.extend([cmap(i) for i in range(cmap.N)])
    return [mcolors.to_hex(c) for c in cols]

_COLOR_PALETTE = _make_big_palette()

_MODEL_TO_COLOR = {
    # Good Guy Team (Cool, calming blues/cyans)
    _canonical_name("WinRate-Villager"): "#4A81BF", # Solid Blue
    _canonical_name("WinRate-Seer"): "#78B2D4",     # Lighter Sky Blue
    _canonical_name("WinRate-Doctor"): "#13A699",   # Vibrant Teal
    
    # Bad Guy Team (Aggressive, striking reds)
    _canonical_name("WinRate-Werewolf"): "#D94F53", # Flat Carmine Red
    
    # Neutral/Mechanics
    _canonical_name("Voting"): "#F28E2B",           # Muted Orange
    _canonical_name("Overall"): "#95A5A6",          # Slate Grey
}
_USED_COLOR_IDX = set()

def get_model_color(name: str):
    """Return a stable, collision-free color for name (Hex string)."""
    key = _canonical_name(name)
    if key in _MODEL_TO_COLOR:
        return _MODEL_TO_COLOR[key]

    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    idx = int(h[:8], 16) % len(_COLOR_PALETTE)

    start = idx
    while idx in _USED_COLOR_IDX:
        idx = (idx + 1) % len(_COLOR_PALETTE)
        if idx == start:
            hue = (int(h[:8], 16) % 360) / 360.0
            _MODEL_TO_COLOR[key] = mcolors.to_hex(mcolors.hsv_to_rgb((hue, 0.65, 0.85)))
            return _MODEL_TO_COLOR[key]

    _USED_COLOR_IDX.add(idx)
    _MODEL_TO_COLOR[key] = _COLOR_PALETTE[idx]
    return _MODEL_TO_COLOR[key]


def _save_figure_mpl(fig, output_path, width=None, height=None):
    if not output_path: return
    if isinstance(output_path, (list, tuple)):
        for path in output_path: _save_figure_mpl(fig, path, width, height)
        return
    import os
    ext = os.path.splitext(output_path)[1].lower()
    try:
        if ext == ".html":
            output_path = output_path.replace(".html", ".png")
        if width and height:
            fig.set_size_inches(width / 100.0, height / 100.0)
        fig.savefig(output_path, bbox_inches="tight", dpi=300)
        print(f"Saved chart to {output_path}")
        plt.close(fig)
    except Exception as e:
        print(f"Error saving to {output_path}: {e}")


def plot_metrics(evaluator, output_path: Union[str, List[str]] = "metrics.html"):
    df = evaluator._prepare_plot_data()

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
        horizontal_spacing=0.1,
    )

    # Create a consistent color map for all agents and a default sorting order
    agent_gte_ratings = {name: metrics.gte_rating[0] for name, metrics in evaluator.metrics.items()}
    all_agents_sorted = sorted(agent_gte_ratings, key=agent_gte_ratings.get)

    agent_color_map = {agent: get_model_color(agent) for agent in all_agents_sorted}

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

    title_text = f"Agent Performance Metrics<br><sup>Total Games: {len(evaluator.games)} | Bootstrap Samples: Elo={getattr(evaluator, 'elo_samples', 'N/A')}, OpenSkill={getattr(evaluator, 'openskill_samples', 'N/A')}</sup>"
    fig.update_layout(
        title_text=title_text,
        title_font_size=24,
        title_x=0.01,
        height=350 * len(present_categories),
        width=350 * max_cols if max_cols > 2 else 1000,
        font=dict(family="Inter, sans-serif"),
        showlegend=False,
        margin=dict(l=120, r=50),
    )

    import re
    import numpy as np

    if output_path:
        paths = [output_path] if isinstance(output_path, str) else output_path
        for path in paths:
            base, ext = os.path.splitext(path)
            for cat in present_categories:
                safe_cat = re.sub(r'[^a-zA-Z0-9_\-]', '_', cat.strip())
                row_output_html = f"{base}_{safe_cat}.html"
                row_output_png = f"{base}_{safe_cat}.png"

                cat_metrics = category_metrics_map[cat]
                row_titles = np.pad(cat_metrics, (0, max_cols - len(cat_metrics)), constant_values="").tolist()

                row_fig = make_subplots(
                    rows=1,
                    cols=max_cols,
                    subplot_titles=row_titles,
                    horizontal_spacing=0.04,
                )

                for col_idx, metric in enumerate(cat_metrics):
                    metric_data = df[(df["category"] == cat) & (df["metric"] == metric)]

                    if cat == "Ratings":
                        sorted_agents_by_value = metric_data.sort_values("value", ascending=True)["agent"].tolist()
                        row_fig.update_xaxes(
                            categoryorder="array", categoryarray=sorted_agents_by_value, row=1, col=col_idx + 1
                        )
                    else:
                        row_fig.update_xaxes(
                            categoryorder="array", categoryarray=all_agents_sorted, row=1, col=col_idx + 1
                        )

                    row_fig.add_trace(
                        go.Bar(
                            name=metric,
                            x=metric_data["agent"],
                            y=metric_data["value"],
                            error_y=dict(type="data", array=metric_data["CI95"]),
                            marker_color=metric_data["agent"].map(agent_color_map),
                            showlegend=False,
                            hovertemplate="<b>%{x}</b><br>%{y:.2f} ± %{error_y.array:.2f}<extra></extra>",
                        ),
                        row=1,
                        col=col_idx + 1,
                    )

                    if cat == "Ratings":
                        row_fig.update_yaxes(matches=None, row=1, col=col_idx + 1)
                    else:
                        row_fig.update_yaxes(range=[0, 1.05] if cat != "Cost" else None, row=1, col=col_idx + 1)
                    row_fig.update_xaxes(tickangle=45, row=1, col=col_idx + 1)

                row_fig.update_layout(
                    title_text=f"{cat} Metrics",
                    title_font_size=20,
                    title_x=0.01,
                    height=400,
                    width=250 * max_cols if max_cols > 2 else 1000,
                    font=dict(family="Inter, sans-serif"),
                    showlegend=False,
                    margin=dict(l=120, r=50),
                )

                _save_figure(row_fig, row_output_html, width=row_fig.layout.width, height=row_fig.layout.height)
                _save_figure(row_fig, row_output_png, width=row_fig.layout.width, height=row_fig.layout.height)

    _save_figure(fig, output_path, width=fig.layout.width, height=fig.layout.height)
    return fig


def plot_pareto_frontier(evaluator, output_path: Union[str, List[str]] = "pareto_frontier.html"):
    # Gather data
    data = []
    for agent_name, metrics in evaluator.metrics.items():
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
        title=f"Cost-Performance Pareto Frontier (GTE)<br><sup>Total Games: {len(evaluator.games)} | GTE Bootstrap Samples: {getattr(evaluator, 'gte_samples', 'N/A')}</sup>",
        xaxis_title="Average Cost per Game ($)",
        yaxis_title="GTE Overall Rating",
        template="plotly_white",
        font=dict(family="Inter, sans-serif"),
        width=900,
        height=600,
    )

    _save_figure(fig, output_path, width=900, height=600)
    return fig


def plot_tournament_graph(evaluator, output_path: Union[str, List[str]] = "tournament_graph.html"):
    """Plots a directed tournament graph where directed edges indicate the margin of victory."""
    from collections import defaultdict

    all_agents = list(evaluator.metrics.keys())
    all_agents.sort(key=lambda x: getattr(evaluator.metrics[x], 'gte_rating', (0.0, 0.0))[0], reverse=True)
    votes = defaultdict(int)

    for g in evaluator.games:
        team_v = {p.agent.display_name for p in g.players if p.role.team == Team.VILLAGERS}
        team_w = {p.agent.display_name for p in g.players if p.role.team == Team.WEREWOLVES}

        winner_names = team_v if g.winner_team == Team.VILLAGERS else team_w
        loser_names = team_w if g.winner_team == Team.VILLAGERS else team_v

        for w in winner_names:
            for l in loser_names:
                votes[(w, l)] += 1

    G = nx.DiGraph()
    G.add_nodes_from(all_agents)

    for a in all_agents:
        for b in all_agents:
            if a == b:
                continue
            margin = votes[(a, b)] - votes[(b, a)]
            if margin > 0:
                G.add_edge(a, b, weight=margin)

    pos = nx.circular_layout(G)

    edge_x = []
    edge_y = []
    annotations = []

    for u, v, d in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]

        # Draw line
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

        # Arrow
        annotations.append(
            dict(
                ax=x0, ay=y0, axref='x', ayref='y',
                x=x1, y=y1, xref='x', yref='y',
                showarrow=True,
                arrowhead=4,
                arrowsize=1.0,
                arrowwidth=1.5,
                arrowcolor='#a3a3a3',
                opacity=0.8,
                text=""
            )
        )
        # Text at midpoint
        annotations.append(
            dict(
                x=(x0 + x1) / 2,
                y=(y0 + y1) / 2,
                xref='x',
                yref='y',
                text=f"{d['weight']}",
                hovertext=f"{u} -> {v}: {d['weight']}",
                showarrow=False,
                font=dict(size=10, color="#ffffff"),
                bgcolor="#2c3e50",
                bordercolor="#ffffff",
                borderwidth=1,
                borderpad=3,
                opacity=0.9
            )
        )

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    node_x = []
    node_y = []
    node_text = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition="top center",
        marker=dict(
            showscale=False,
            colorscale='YlGnBu',
            size=30,
            color='#1f77b4',
            line_width=2,
            line_color='#ffffff')
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=dict(text='Tournament Graph (Directed Edge Indicates Win Margin)',
                                   font=dict(size=18, family="Inter, sans-serif")),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=40, l=40, r=40, t=60),
                        annotations=annotations,
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        width=1000,
                        height=800,
                        plot_bgcolor="#f8f9fa",
                        font=dict(family="Inter, sans-serif")
                    )
                    )

    _save_figure(fig, output_path, width=fig.layout.width, height=fig.layout.height)
    return fig


def plot_pairwise_winrates(evaluator, output_path: Union[str, List[str]] = "pairwise_winrates.html"):
    """Plots a heatmap of pairwise winrates between all distinct agents."""
    import numpy as np
    from collections import defaultdict

    all_agents = list(evaluator.metrics.keys())
    all_agents.sort(key=lambda x: getattr(evaluator.metrics[x], 'gte_rating', (0.0, 0.0))[0], reverse=True)
    counts = defaultdict(int)
    wins = defaultdict(int)

    for g in evaluator.games:
        team_v = [p.agent.display_name for p in g.players if p.role.team == Team.VILLAGERS]
        team_w = [p.agent.display_name for p in g.players if p.role.team == Team.WEREWOLVES]

        for v in team_v:
            for w in team_w:
                counts[(v, w)] += 1
                counts[(w, v)] += 1
                if g.winner_team == Team.VILLAGERS:
                    wins[(v, w)] += 1
                elif g.winner_team == Team.WEREWOLVES:
                    wins[(w, v)] += 1

    n = len(all_agents)
    matrix = np.full((n, n), np.nan)

    for i, a in enumerate(all_agents):
        for j, b in enumerate(all_agents):
            if a == b:
                matrix[i, j] = 0.5  # Self vs Self trivially tied
                continue
            if counts[(a, b)] > 0:
                matrix[i, j] = wins[(a, b)] / counts[(a, b)]

    # Center the color scale at 0.5 and exhibit the individual cell values
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=all_agents,
            y=all_agents,
            colorscale="RdBu",
            zmid=0.5,
            zmin=0.0,
            zmax=1.0,
            text=np.around(matrix, 2),
            texttemplate="%{text:.2f}",
            textfont={"size": 10},
            hovertemplate="Row: %{y}<br>Col: %{x}<br>Win Rate: %{z:.2f}<extra></extra>"
        )
    )

    fig.update_layout(
        title="Pairwise Head-to-Head Win Rates (Row vs Column)",
        xaxis_title="Opponent Agent",
        yaxis_title="Agent",
        width=900,
        height=900,
        font=dict(family="Inter, sans-serif")
    )

    _save_figure(fig, output_path, width=900, height=900)
    return fig


def plot_gte_evaluation(evaluator, output_path: Union[str, List[str]] = "gte_evaluation.html"):
    if evaluator.gte_game is None:
        print("GTE evaluation has not been run. Please run .evaluate() first.")
        return None

    # --- 1. Data Preparation ---
    # Use agents from the GTE game structure if available
    # If rating_player=1 (Agents), we should look at actions[1]
    game = evaluator.gte_game
    if game and getattr(game, 'actions', None) and len(game.actions) > 1:
        agents = game.actions[1]
    else:
        print("Warning: GTE Game structure missing agent list. Falling back to metrics keys (risky).")
        agents = sorted(list(evaluator.metrics.keys()))

    tasks = evaluator.gte_tasks

    ratings_mean = evaluator.gte_ratings[0][1]
    ratings_std = evaluator.gte_ratings[1][1]

    joint_mean = evaluator.gte_joint[0]
    r2m_contributions_mean = evaluator.gte_contributions_raw[0]

    agent_rating_map = {agent: ratings_mean[i] for i, agent in enumerate(agents)}
    sorted_agents = sorted(agents, key=lambda x: agent_rating_map[x], reverse=True)

    game = evaluator.gte_game
    rating_player = 1  # Agents
    contrib_player = 0  # Metrics

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
        print(
            f"    !!! ERROR: Mismatched GTE rating lengths: Agents={len(agents)}, Ratings={len(ratings_mean)}. Skipping plot.")
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
                marker_color=get_model_color(metric),
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
        title_text=f"Game Theoretic Evaluation: Agent Ratings<br><sup>Total Games: {len(evaluator.games)} | GTE Bootstrap Samples: {getattr(evaluator, 'gte_samples', 'N/A')}</sup>",
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


def plot_gte_metrics_analysis(evaluator, output_path: Union[str, List[str]] = "gte_metrics.html"):
    """Plots GTE Metric analysis: Weights (Marginals) and Ratings (Payoffs)."""
    if not hasattr(evaluator, "gte_game") or evaluator.gte_game is None:
        print("GTE evaluation not run or failed. Skipping metric analysis plot.")
        return None

    if not evaluator.gte_game or not getattr(evaluator.gte_game, 'actions', None) or len(
            evaluator.gte_game.actions) < 2:
        print("    !!! WARNING: Cannot plot GTE metrics analysis: Metadata actions missing or incomplete.")
        return None

    # Tasks are Player 0
    agents = evaluator.gte_game.actions[1]
    tasks = evaluator.gte_game.actions[0]

    # Unpack Data
    # Marginals (Weights) - Player 0
    try:
        metric_weights_mean = evaluator.gte_marginals[0][0]
        metric_weights_std = evaluator.gte_marginals[1][0]

        # Ratings (Values) - Player 0
        metric_ratings_mean = evaluator.gte_ratings[0][0]
        metric_ratings_std = evaluator.gte_ratings[1][0]
    except (IndexError, TypeError) as e:
        print(f"    !!! WARNING: GTE results structure mismatch in plot: {e}")
        return None

    tasks = evaluator.gte_tasks

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
            marker_color="#3b82f6",  # Blue
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


def plot_metrics_paper(evaluator, output_path="metrics.png"):
    df = evaluator._prepare_plot_data()
    category_order = ["Overall", "Voting Accuracy", "Win-Dependent Metrics", "Dominance Metrics",
                      "Role-Specific Win Rate", "Role-Specific KSR", "Win-Dependent KSR", "Ratings", "Cost"]
    present_categories = [cat for cat in category_order if cat in df["category"].unique()]
    agent_gte_ratings = {name: metrics.gte_rating[0] for name, metrics in evaluator.metrics.items()}
    all_agents_sorted = sorted(agent_gte_ratings, key=agent_gte_ratings.get)
    agent_color_map = {agent: get_model_color(agent) for agent in all_agents_sorted}

    import re
    if output_path:
        paths = [output_path] if isinstance(output_path, str) else output_path
        for path in paths:
            base, ext = os.path.splitext(path)
            for cat in present_categories:
                cat_df = df[df["category"] == cat]
                metrics_in_cat = sorted(cat_df["metric"].unique())
                if cat == "Ratings":
                    metrics_in_cat = sorted(metrics_in_cat, key=lambda x: 0 if x == "Elo" else 1)

                safe_cat = re.sub(r'[^a-zA-Z0-9_\\-]', '_', cat.strip())
                row_output_png = f"{base}_{safe_cat}.png"

                fig, axes = plt.subplots(1, len(metrics_in_cat), figsize=(4 * len(metrics_in_cat), 5), squeeze=False)
                axes = axes.flatten()
                for i, metric in enumerate(metrics_in_cat):
                    ax = axes[i]
                    metric_data = cat_df[cat_df["metric"] == metric]
                    order = all_agents_sorted
                    if cat == "Ratings": order = metric_data.sort_values("value")["agent"].tolist()
                    palette_colors = [agent_color_map[a] for a in order]
                    sns.barplot(data=metric_data, x="agent", y="value", hue="agent", order=order, palette=palette_colors, ax=ax, edgecolor='black', linewidth=0.5, legend=False)
                    for j, agent in enumerate(order):
                        agent_data = metric_data[metric_data["agent"] == agent]
                        if not agent_data.empty:
                            val = agent_data["value"].values[0]
                            err = agent_data["CI95"].values[0]
                            ax.errorbar(j, val, yerr=err, color='black', capsize=3)
                    # Remove title to rely on paper caption
                    ax.set_xlabel("")
                    ax.set_ylabel(metric)
                    ax.set_xticks(range(len(order)))
                    ax.set_xticklabels(order, rotation=45, ha='right', rotation_mode="anchor")
                    if cat != "Cost" and cat != "Ratings": ax.set_ylim(0, 1.05)
                fig.tight_layout()
                _save_figure_mpl(fig, row_output_png)
    return None


def plot_pareto_frontier_paper(evaluator, output_path="pareto_frontier.png"):
    data = []
    for agent_name, metrics in evaluator.metrics.items():
        gte_mean, _ = getattr(metrics, 'gte_rating', (0.0, 0.0))
        avg_cost, _ = metrics.get_avg_cost()
        if avg_cost <= 0: continue
        data.append({"agent": agent_name, "gte_rating": gte_mean, "cost": avg_cost})
    if not data: return None

    df = pd.DataFrame(data)
    sorted_df = df.sort_values(by=["cost", "gte_rating"], ascending=[True, False])
    frontier_points = []
    current_max_rating = -float("inf")
    for index, row in sorted_df.iterrows():
        if row["gte_rating"] > current_max_rating:
            frontier_points.append(row)
            current_max_rating = row["gte_rating"]
    frontier_df = pd.DataFrame(frontier_points)

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(data=df, x="cost", y="gte_rating", s=100, color="#6366F1", ax=ax, zorder=3)
    ax.plot(frontier_df["cost"], frontier_df["gte_rating"], color="#10B981", linestyle="--", linewidth=2, zorder=2,
            label="Pareto Frontier")
    for i, row in df.iterrows(): ax.text(row["cost"], row["gte_rating"] + 0.02, row["agent"], ha='center', va='bottom',
                                         fontsize=9)
    # No titles as requested
    ax.set_xlabel("Average Cost per Game ($)")
    ax.set_ylabel("GTE Overall Rating")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    _save_figure_mpl(fig, output_path)
    return fig


def plot_tournament_graph_paper(evaluator, output_path="tournament_graph.png"):
    all_agents = list(evaluator.metrics.keys())
    all_agents.sort(key=lambda x: getattr(evaluator.metrics[x], 'gte_rating', (0.0, 0.0))[0], reverse=True)
    from collections import defaultdict
    votes = defaultdict(int)

    for g in evaluator.games:
        team_v = {p.agent.display_name for p in g.players if p.role.team == Team.VILLAGERS}
        team_w = {p.agent.display_name for p in g.players if p.role.team == Team.WEREWOLVES}
        winner_names = team_v if g.winner_team == Team.VILLAGERS else team_w
        loser_names = team_w if g.winner_team == Team.VILLAGERS else team_v
        for w in winner_names:
            for l in loser_names: votes[(w, l)] += 1

    G = nx.DiGraph()
    G.add_nodes_from(all_agents)
    for a in all_agents:
        for b in all_agents:
            if a == b: continue
            margin = votes[(a, b)] - votes[(b, a)]
            if margin > 0: G.add_edge(a, b, weight=margin)

    fig, ax = plt.subplots(figsize=(10, 8))
    pos = nx.circular_layout(G)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color="#1f77b4", node_size=2000, edgecolors="white", linewidths=2)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=9, font_color="black")
    edge_labels = {}
    for u, v, d in G.edges(data=True):
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], ax=ax, arrowstyle="->", arrowsize=20, edge_color="#a3a3a3",
                               alpha=0.8)
        edge_labels[(u, v)] = d['weight']
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax, font_size=8, label_pos=0.3)
    # No title
    ax.axis("off")
    _save_figure_mpl(fig, output_path)
    return fig


def plot_pairwise_winrates_paper(evaluator, output_path="pairwise_winrates.png"):
    """Plots a combined heatmap of pairwise winrates and bootstrapped GTE distributions."""
    all_agents = list(evaluator.metrics.keys())
    all_agents.sort(key=lambda x: getattr(evaluator.metrics[x], 'gte_rating', (0.0, 0.0))[0], reverse=True)
    from collections import defaultdict
    counts = defaultdict(int)
    wins = defaultdict(int)

    for g in evaluator.games:
        team_v = [p.agent.display_name for p in g.players if p.role.team == Team.VILLAGERS]
        team_w = [p.agent.display_name for p in g.players if p.role.team == Team.WEREWOLVES]
        for v in team_v:
            for w in team_w:
                counts[(v, w)] += 1
                counts[(w, v)] += 1
                if g.winner_team == Team.VILLAGERS:
                    wins[(v, w)] += 1
                elif g.winner_team == Team.WEREWOLVES:
                    wins[(w, v)] += 1

    n = len(all_agents)
    win_rate_matrix = pd.DataFrame(np.nan, index=all_agents, columns=all_agents)
    for i, a in enumerate(all_agents):
        for j, b in enumerate(all_agents):
            if a == b: continue
            if counts[(a, b)] > 0: win_rate_matrix.loc[a, b] = wins[(a, b)] / counts[(a, b)]

    order = all_agents
    win_rate_ord = win_rate_matrix.reindex(index=order, columns=order)
    diag_mask = np.eye(len(order), dtype=bool)

    fig = plt.figure(figsize=(16, 7), constrained_layout=True)
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.15, 1.0])

    ax_hm = fig.add_subplot(gs[0, 0])
    ax_vi = fig.add_subplot(gs[0, 1])

    sns.heatmap(win_rate_ord, ax=ax_hm, mask=diag_mask, annot=True, fmt=".0%", cmap="RdBu", vmin=0, vmax=1, center=0.5,
                linewidths=0.3, linecolor="white", cbar_kws={"shrink": 0.9, "label": "Win rate"}, annot_kws={"size": 8})
    # Removing titles as requested
    ax_hm.set_xlabel("Opponent")
    ax_hm.set_ylabel("Player")
    ax_hm.set_xticklabels(ax_hm.get_xticklabels(), rotation=40, ha="right", rotation_mode="anchor")
    ax_hm.tick_params(axis="x", pad=6)
    ax_hm.tick_params(axis="y", rotation=0)

    poker_violin_color = "#2A9D8F"
    plot_df_rows = []
    if getattr(evaluator, "gte_bootstrapped_agent_ratings", None) is not None:
        samples = evaluator.gte_bootstrapped_agent_ratings
        if getattr(evaluator, "gte_game", None) and getattr(evaluator.gte_game, "actions", None) and len(
                evaluator.gte_game.actions) > 1:
            gte_agents = evaluator.gte_game.actions[1]
            for agent_name in all_agents:
                if agent_name in gte_agents:
                    idx = list(gte_agents).index(agent_name)
                    for val in samples[:, idx]: plot_df_rows.append({"Model": agent_name, "Bootstrapped GTE": val})

    if plot_df_rows:
        plot_df = pd.DataFrame(plot_df_rows)
        sns.violinplot(data=plot_df, x="Bootstrapped GTE", y="Model", order=order, ax=ax_vi, inner="quartile",
                       linewidth=1, cut=0, scale="width", color=poker_violin_color)
        ax_vi.axvline(0, linestyle="--", linewidth=1)
        # Removing title
        ax_vi.set_xlabel("Bootstrapped GTE Rating")
        ax_vi.set_ylabel("")
        ax_vi.grid(axis="x", alpha=0.25)
        xmax = np.percentile(plot_df["Bootstrapped GTE"], 99)
        xmin = np.percentile(plot_df["Bootstrapped GTE"], 1)
        pad = 0.1 * (xmax - xmin + 1e-9)
        ax_vi.set_xlim(xmin - pad, xmax + pad)
    else:
        ax_vi.text(0.5, 0.5, "No GTE Bootstrap Samples Available", ha="center", va="center")

    _save_figure_mpl(fig, output_path, width=1600, height=700)
    return fig


def plot_gte_evaluation_paper(evaluator, output_path="gte_evaluation.png"):
    if getattr(evaluator, "gte_game", None) is None: return None
    game = evaluator.gte_game
    if game and getattr(game, "actions", None) and len(game.actions) > 1:
        agents = game.actions[1]
    else:
        agents = sorted(list(evaluator.metrics.keys()))

    tasks = evaluator.gte_tasks
    ratings_mean = evaluator.gte_ratings[0][1]
    ratings_std = evaluator.gte_ratings[1][1]
    r2m_contributions_mean = evaluator.gte_contributions_raw[0]

    agent_rating_map = {agent: ratings_mean[i] for i, agent in enumerate(agents)}
    sorted_agents = sorted(agents, key=lambda x: agent_rating_map[x], reverse=True)

    fig, ax = plt.subplots(figsize=(12, max(6, len(agents) * 0.6)))
    y_pos = np.arange(len(sorted_agents))
    lefts = np.zeros(len(sorted_agents))
    for j, task in enumerate(tasks):
        task_idx = tasks.index(task)
        contribs = []
        for agent in sorted_agents:
            agent_idx = list(agents).index(agent)
            contribs.append(r2m_contributions_mean[agent_idx, task_idx])
        ax.barh(y_pos, contribs, left=lefts, color=get_model_color(task), label=task, edgecolor="white", height=0.4, alpha=0.6)
        lefts += np.array(contribs)
    # Draw horizontal violin plot for Net Rating bootstrap samples
    plot_df_rows = []
    if getattr(evaluator, "gte_bootstrapped_agent_ratings", None) is not None:
        samples = evaluator.gte_bootstrapped_agent_ratings
        for agent in sorted_agents:
            agent_idx = list(agents).index(agent)
            for val in samples[:, agent_idx]:
                plot_df_rows.append({"Model": agent, "Bootstrapped GTE": val})

    if plot_df_rows:
        plot_df = pd.DataFrame(plot_df_rows)
        sns.violinplot(
            data=plot_df,
            x="Bootstrapped GTE",
            y="Model",
            order=sorted_agents,
            ax=ax,
            inner="quartile",
            linewidth=1,
            cut=0,
            scale="width",
            color="#2A9D8F",
            alpha=0.85,
            zorder=10
        )
        ax.axvline(0, linestyle="--", linewidth=1, color="black", alpha=0.5)
    else:
        ratings_sorted = [agent_rating_map[a] for a in sorted_agents]
        std_sorted = [ratings_std[list(agents).index(a)] for a in sorted_agents]
        ax.errorbar(ratings_sorted, y_pos, xerr=np.array(std_sorted) * 1.96, fmt="D", color="black", capsize=3,
                    label="Net Rating")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_agents)
    if not ax.yaxis_inverted():
        ax.invert_yaxis()
    ax.set_xlabel("Win Rate Contribution (and Net Rating)")
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    # Removing title
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.tight_layout()
    _save_figure_mpl(fig, output_path)
    return fig


def plot_gte_metrics_analysis_paper(evaluator, output_path="gte_metrics.png"):
    if not hasattr(evaluator, "gte_game") or getattr(evaluator, "gte_game", None) is None: return None
    if not getattr(evaluator.gte_game, "actions", None) or len(evaluator.gte_game.actions) < 2: return None

    tasks = evaluator.gte_tasks
    metric_weights_mean = evaluator.gte_marginals[0][0]
    metric_weights_std = evaluator.gte_marginals[1][0]
    metric_ratings_mean = evaluator.gte_ratings[0][0]
    metric_ratings_std = evaluator.gte_ratings[1][0]

    weights_df = pd.DataFrame({"metric": tasks, "weight": metric_weights_mean, "CI95": metric_weights_std * 1.96})
    ratings_df = pd.DataFrame({"metric": tasks, "rating": metric_ratings_mean, "CI95": metric_ratings_std * 1.96})

    weights_df = weights_df.sort_values("weight", ascending=True)
    sorted_tasks = weights_df["metric"].tolist()
    ratings_df = ratings_df.set_index("metric").reindex(sorted_tasks).reset_index()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, max(8, len(tasks) * 0.8)))
    ax1.barh(weights_df["metric"], weights_df["weight"], color="#3b82f6", xerr=weights_df["CI95"], capsize=3,
             edgecolor="black")
    ax1.set_xlabel("Marginal Probability")

    ratings_df_sorted = ratings_df.sort_values("rating", ascending=True)
    ax2.errorbar(ratings_df_sorted["rating"], ratings_df_sorted["metric"], xerr=ratings_df_sorted["CI95"], fmt="D",
                 color="#ef4444", capsize=3)
    ax2.set_xlabel("Nash Value (Rating)")
    # Removing titles
    fig.tight_layout()
    _save_figure_mpl(fig, output_path)
    return fig
