"""Physical route, heading, and teacher-stage figures using validated saved records."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D
from cmnav.analysis.navigation_records import ACTIONS
from cmnav.analysis.report_contracts import MODES, INTERVALS, CONDITIONS
from cmnav.reporting.labels import NAVIGATION_LABELS as LABELS, NAVIGATION_COLORS as COLORS, ACTION_COLORS
from cmnav.reporting.labels import FIGURE_LABELS, GCML_LABELS, GCML_COLORS


def draw_graph(axis: Any, positions: np.ndarray, edges: list[list[int]]) -> None:
    segments = [positions[edge][:, [0, 2]] for edge in edges]
    axis.add_collection(LineCollection(segments, colors="#C2C8CC", linewidths=0.7,
                                       alpha=0.55, zorder=1))
    axis.scatter(positions[:, 0], positions[:, 2], s=6, color="#A5ADB3", alpha=0.6, zorder=2)
    axis.autoscale_view()
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("World x (m)")
    axis.set_ylabel("World z (m)")
    axis.grid(alpha=0.15)


def heading_arrows(axis: Any, data: dict[str, Any], length: float,
                   bounds: tuple[float, float, float, float] | None = None) -> None:
    positions, rotations = data["positions"], data["rotations"]
    candidates = list(range(0, len(positions), max(1, len(positions) // 10)))
    chosen = []
    for index in candidates:
        x, _, z = positions[index]
        if bounds is not None and not (bounds[0] <= x <= bounds[1] and bounds[2] <= z <= bounds[3]):
            continue
        if any(np.linalg.norm(positions[index, [0, 2]] - positions[other, [0, 2]]) < length * 1.5
               for other in chosen):
            continue
        chosen.append(index)
    if not chosen:
        return
    # Rotate Habitat body forward (0,0,-1) by its saved agent-to-world xyzw quaternion.
    x, y, z, w = rotations[chosen].T
    forward_x = -2.0 * (x * z + w * y)
    forward_z = -(1.0 - 2.0 * (x * x + y * y))
    axis.quiver(positions[chosen, 0], positions[chosen, 2], forward_x * length,
                forward_z * length, angles="xy", scale_units="xy", scale=1,
                width=0.004, color="#7A3100", alpha=0.75, zorder=6)


def render_navigation_audit(data: dict[str, Any], output: Path) -> dict[str, str]:
    run = data["run"]
    task_index = data["task_index"]
    task = data["task"]
    config = data["config"]
    graph_positions = data["graph_positions"]
    edges = data["edges"]
    methods = data["methods"]
    cml = data["cml"]
    episode_id = data["episode_id"]
    goal = data["goal"]
    start = data["start"]
    decisions = data["decisions"]
    pair = data["pair"]
    bounds = data["bounds"]
    height_note = data["height_note"]
    graph_height_span = data["graph_height_span"]
    execution_height_span = data["execution_height_span"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 12, "axes.titleweight": "bold",
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "savefig.facecolor": "white"})
    figure = plt.figure(figsize=(16, 12), facecolor="white")
    grid = figure.add_gridspec(2, 2, left=0.07, right=0.97, top=0.87, bottom=0.12,
                               hspace=0.36, wspace=0.28, height_ratios=[1.15, 1.0])
    overview, zoom = figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1])
    distance_axis = figure.add_subplot(grid[1, 0])
    detail_grid = grid[1, 1].subgridspec(2, 1, height_ratios=[1.5, 1], hspace=0.5)
    action_axis, table_axis = figure.add_subplot(detail_grid[0]), figure.add_subplot(detail_grid[1])
    figure.suptitle("PointNav execution audit", x=0.07, y=0.975, ha="left", fontsize=24, fontweight="bold")
    figure.text(0.07, 0.935,
                f"Episode {episode_id}  |  CML dimension {config['cml_dimension']}  |  "
                f"Budget {config['max_steps']} primitive actions  |  Oracle-state development diagnostic",
                fontsize=12, color="#48515A")
    figure.text(0.07, 0.907, f"Saved run: {run.name}", fontsize=10, color="#67727B")

    draw_graph(overview, graph_positions, edges)
    overview.set_title("A  Actual routes on the observed abstract graph", loc="left", pad=12)
    for method in reversed(list(methods)):
        data = methods[method]
        overview.plot(data["positions"][:, 0], data["positions"][:, 2], color=COLORS[method],
                      linewidth=2.0 if method == "cml" else 1.65,
                      linestyle="-" if method != "oracle_follower" else "--",
                      alpha=0.92, label=LABELS[method], zorder=4 if method == "cml" else 3)
    overview.scatter(start[0], start[2], marker="*", color="#222222", s=130, label="Start", zorder=7)
    overview.scatter(goal[0], goal[2], marker="D", color="#222222", s=55, label="Goal", zorder=7)
    end = cml["positions"][-1]
    overview.scatter(end[0], end[2], marker="X", color=COLORS["cml"], edgecolors="white",
                     linewidths=0.6, s=95, label="CML end", zorder=8)
    overview.add_patch(Rectangle((bounds[0], bounds[2]), bounds[1] - bounds[0], bounds[3] - bounds[2],
                                fill=False, edgecolor="#7A3100", linestyle="--", linewidth=1.2, zorder=7))
    heading_arrows(overview, cml, 0.3)
    overview.legend(loc="best", framealpha=0.94, fontsize=9, ncol=2)

    draw_graph(zoom, graph_positions, edges)
    zoom.set_xlim(bounds[:2])
    zoom.set_ylim(bounds[2:])
    zoom.set_title("B  CML choices at the most-used node pair", loc="left", pad=12)
    for method, data in methods.items():
        zoom.plot(data["positions"][:, 0], data["positions"][:, 2], color=COLORS[method],
                  linewidth=1.5, linestyle="-" if method == "cml" else "--", alpha=0.45, zorder=3)
    heading_arrows(zoom, cml, 0.065, bounds)
    if pair:
        for node in pair:
            x, z = graph_positions[node, [0, 2]]
            zoom.scatter(x, z, s=100, color="white", edgecolors="#444444", zorder=8)
            zoom.annotate(f"node {node}", (x, z), xytext=(0, -20), textcoords="offset points",
                          ha="center", fontsize=10, fontweight="bold", zorder=10)
        for row in decisions["ranked_edges"]:
            if sorted((row["source"], row["target"])) != pair:
                continue
            a = graph_positions[row["source"], [0, 2]]
            b = graph_positions[row["target"], [0, 2]]
            delta = b - a
            length = np.linalg.norm(delta)
            if length <= 1e-12:
                continue
            perpendicular = np.asarray([-delta[1], delta[0]]) / length
            label_position = (a + b) / 2 - perpendicular * max(0.13, length * 0.5)
            zoom.add_patch(FancyArrowPatch(a, b, connectionstyle="arc3,rad=0.25", arrowstyle="-|>",
                                          mutation_scale=15, linewidth=2.2, color=COLORS["cml"],
                                          shrinkA=7, shrinkB=7, zorder=9))
            zoom.text(*label_position, f"edge {row['edge_index']}\n{row['attempts']} chosen / {row['completed']} arrived",
                      ha="center", va="center", fontsize=9, color="#7A3100",
                      bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 3}, zorder=10)
    zoom.text(0.03, 0.97, f"{decisions['dominant_pair_attempts']} / {len(decisions['selected_edge_sequence'])} edge choices on this pair\n"
              "Curved arrows show graph choices, not physical paths.", transform=zoom.transAxes,
              ha="left", va="top", fontsize=9, bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none"}, zorder=11)

    for method, data in methods.items():
        distance_axis.plot(np.arange(len(data["distances"])), data["distances"], color=COLORS[method],
                           linewidth=1.9, label=LABELS[method])
        distance_axis.scatter(len(data["distances"]) - 1, data["distances"][-1], color=COLORS[method], s=25)
    distance_axis.axvline(config["max_steps"], color="#777777", linestyle=":", linewidth=1)
    distance_axis.set_title("C  Distance to final task goal at every action", loc="left", pad=12)
    distance_axis.set_xlabel("Executed primitive actions")
    distance_axis.set_ylabel("Geodesic distance (m)")
    distance_axis.set_ylim(bottom=0)
    distance_axis.grid(alpha=0.2)
    distance_axis.legend(loc="best", fontsize=9)

    method_names = list(methods)
    y = np.arange(len(method_names))
    left = np.zeros(len(method_names))
    for action, color in zip(ACTIONS, ACTION_COLORS):
        values = np.asarray([methods[method]["stats"]["actions"][action] for method in method_names])
        action_axis.barh(y, values, left=left, color=color, height=0.62,
                         label=action.replace("MOVE_", "").replace("TURN_", "").title())
        for index, value in enumerate(values):
            if value >= max(10, config["max_steps"] * 0.04):
                action_axis.text(left[index] + value / 2, index, str(value), ha="center", va="center", fontsize=9)
        left += values
    for index, total in enumerate(left):
        action_axis.text(total + config["max_steps"] * 0.015, index, str(int(total)), va="center", fontsize=9)
    action_axis.set_yticks(y, [LABELS[method] for method in method_names])
    action_axis.invert_yaxis()
    action_axis.set_xlim(0, max(config["max_steps"], left.max()) * 1.12)
    action_axis.set_xlabel("Primitive action counts, including turns and STOP")
    action_axis.set_title("D  Where the action budget went", loc="left", pad=25)
    action_axis.legend(ncol=4, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.01), frameon=False)
    action_axis.grid(axis="x", alpha=0.2)
    action_axis.set_axisbelow(True)
    table_axis.axis("off")
    cells = []
    for method in method_names:
        stats = methods[method]["stats"]
        cells.append([LABELS[method], "Yes" if stats["success"] else "No",
                      f"{stats['path_length_metres']:.2f}", str(stats["collisions"]),
                      f"{stats['graph_hops_completed']}/{stats['graph_hops_attempted']}" if method != "oracle_follower" else "N/A"])
    table = table_axis.table(cellText=cells,
                             colLabels=["Method", "Success", "Path (m)", "Collisions", "Hops done/tried"],
                             colWidths=[0.22, 0.16, 0.19, 0.17, 0.26], loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#D5DBDF")
        if row == 0:
            cell.set_facecolor("#EAF0F4")
            cell.set_text_props(weight="bold")
    figure.text(0.07, 0.071,
                "Gray background: observed abstract nodes/edges, NOT an occupancy map or floor plan. "
                "Colored routes join actual consecutive saved positions; small brown arrows show CML headings.", fontsize=9, color="#48515A")
    figure.text(0.07, 0.047, height_note + "  Attempt counts may include an unfinished hop; saved model is unchanged.",
                fontsize=9, color="#48515A")

    output.mkdir(parents=True, exist_ok=True)
    prefix = output / f"{task}_navigation_audit"
    png, svg, diagnostics_path = prefix.with_suffix(".png"), prefix.with_suffix(".svg"), prefix.with_suffix(".json")
    figure.savefig(png, dpi=180)
    figure.savefig(svg)
    plt.close(figure)
    return {"png": str(png), "svg": str(svg), "diagnostics_json": str(diagnostics_path)}


def render_subgoal_task(output: Path, index: int, episode: str, loaded: dict[str, dict[str, Any]],
                rows: dict[str, dict[str, Any]], positions: np.ndarray, edges: list[list[int]],
                run_name: str) -> Path:
    first = loaded[CONDITIONS[0]]
    start = first["positions"][0]
    final_goal = np.asarray(first["execution"][0]["goal_position"], dtype=float)
    segments = [(positions[source, [0, 2]], positions[target, [0, 2]]) for source, target in edges]
    all_xz = np.concatenate([positions[:, [0, 2]], final_goal[None, [0, 2]],
                             *[record["positions"][:, [0, 2]] for record in loaded.values()]], axis=0)
    low, high = all_xz.min(axis=0), all_xz.max(axis=0)
    padding = max(float(np.max(high - low)) * .04, .25)
    graph_height = float(np.ptp(positions[:, 1]))
    figure, axes = plt.subplots(2, 4, figsize=(19, 11.2), squeeze=False)
    for row_index, mode in enumerate(MODES):
        for column, interval in enumerate(INTERVALS):
            condition = f"{mode}_{interval if interval is not None else 'final'}"
            data, record = loaded[condition], rows[condition]
            axis = axes[row_index, column]
            axis.add_collection(LineCollection(segments, colors="#c9ced3", linewidths=.65, alpha=.72, zorder=1))
            axis.scatter(positions[:, 0], positions[:, 2], s=3, c="#c1c7cd", zorder=1)
            route = data["positions"]
            axis.plot(route[:, 0], route[:, 2], color="#2474b5", linewidth=2.3, alpha=.95, zorder=3)
            if record["stages"] is not None:
                nodes = record["stages"]["nodes"]
                node_positions = positions[nodes]
                axis.scatter(node_positions[:, 0], node_positions[:, 2], s=82, facecolors="none",
                             edgecolors="#e28b16", linewidths=1.7, zorder=4)
                for ordinal, position in enumerate(node_positions, 1):
                    axis.annotate(str(ordinal), (position[0], position[2]), xytext=(4, 5), textcoords="offset points",
                                  fontsize=7, color="#9b5a08", weight="bold", zorder=6)
            axis.scatter(start[0], start[2], s=35, color="#20252b", marker="o", zorder=7)
            axis.scatter(final_goal[0], final_goal[2], s=130, color="#23966f", edgecolors="#164d3c",
                         linewidths=.7, marker="*", zorder=7)
            axis.scatter(route[-1, 0], route[-1, 2], s=55, color="#2474b5", marker="x", linewidths=1.8, zorder=8)
            axis.set_xlim(low[0] - padding, high[0] + padding)
            axis.set_ylim(low[1] - padding, high[1] + padding)
            axis.set_aspect("equal", adjustable="box")
            axis.grid(alpha=.12)
            axis.set_xlabel("World X (m)")
            axis.set_ylabel("World Z (m)")
            title = "Final goal only" if interval is None else f"Teacher subgoal every {interval} graph edges"
            stages = record["stages"]
            stage_text = "" if stages is None else f" | stages {stages['completed']}/{stages['attempted']}/{stages['total']}"
            axis.set_title(f"{title}\nSuccess {int(record['success'])} | SPL {record['spl']:.3f} | {record['steps']} actions\n"
                           f"{record['end_reason']}{stage_text}", fontsize=9.2, pad=9)
            if column == 0:
                axis.text(-.23, .5, FIGURE_LABELS[mode], rotation=90, transform=axis.transAxes,
                          ha="center", va="center", fontsize=12, weight="bold")
    handles = [Line2D([0], [0], color="#c9ced3", lw=1.2, label="Observed graph (directions omitted)"),
               Line2D([0], [0], color="#2474b5", lw=2.3, label="Actual physical execution"),
               Line2D([0], [0], color="#e28b16", marker="o", markerfacecolor="none", lw=0,
                      label="Planned teacher subgoal (not an executed route)"),
               Line2D([0], [0], color="#20252b", marker="o", lw=0, label="Start"),
               Line2D([0], [0], color="#23966f", marker="*", markersize=10, lw=0, label="Final goal"),
               Line2D([0], [0], color="#2474b5", marker="x", lw=0, label="Last recorded pose")]
    figure.suptitle(f"Task {index + 1} | Official episode {episode} | Recorded execution across subgoal conditions",
                   y=.99, fontsize=15, weight="bold")
    figure.text(.5, .958, "Same frozen GCML checkpoint and total action budget. Subgoals are supplied by a privileged teacher.",
                ha="center", fontsize=10, color="#505962")
    figure.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(.5, .034), frameon=False, fontsize=9)
    figure.text(.5, .018, f"Stages: completed / attempted / all planned. X/Z projection; graph vertical span {graph_height:.2f} m. "
                f"Run {run_name}", ha="center", fontsize=8, color="#606972")
    figure.subplots_adjust(left=.08, right=.985, top=.88, bottom=.12, hspace=.42, wspace=.29)
    destination = output / f"task_{index:04d}_routes.png"
    figure.savefig(destination, dpi=150, facecolor="white")
    plt.close(figure)
    return destination


def render_gcml_task(output: Path, i: int, episode: str, loaded: dict[str, Any],
                     methods: list[str], positions: np.ndarray, edges: list[list[int]]) -> Path:
    segments = [(positions[u, [0, 2]], positions[v, [0, 2]]) for u, v in edges]
    first = loaded[methods[0]]
    start = first["positions"][0]
    goal = np.asarray(first["execution"][0]["goal_position"])
    fig, axes = plt.subplots(2, len(methods), figsize=(4.5 * len(methods), 6.8),
                             gridspec_kw={"height_ratios": [1.35, 1]}, squeeze=False)
    max_steps = max(data["stats"]["steps"] for data in loaded.values())
    max_distance = max(float(np.nanmax(data["distances"])) for data in loaded.values())
    for j, method in enumerate(methods):
        data = loaded[method]
        stats = data["stats"]
        axis = axes[0, j]
        axis.add_collection(LineCollection(segments, color="#d5d9dc", linewidth=.7))
        route = data["positions"]
        axis.plot(route[:, 0], route[:, 2], color=GCML_COLORS.get(method), linewidth=2)
        axis.scatter(start[0], start[2], marker="o", c="#222222", s=36, label="Start", zorder=5)
        axis.scatter(goal[0], goal[2], marker="*", c="#ffbb22", edgecolor="#222222", s=140, label="Goal", zorder=5)
        axis.scatter(route[-1, 0], route[-1, 2], marker="x", c=GCML_COLORS.get(method), s=65, zorder=6)
        axis.autoscale_view()
        axis.set_aspect("equal")
        axis.set_title(f"{GCML_LABELS.get(method, method)}\nSuccess {int(stats['success'])}; {stats['steps']} steps")
        axis.set_xlabel("World X (m)")
        axis.set_ylabel("World Z (m)")
        axis.legend(fontsize=8)
        distance = axes[1, j]
        distance.plot(data["distances"], color=GCML_COLORS.get(method))
        distance.set_xlabel("Environment action")
        distance.set_ylabel("Goal geodesic distance (m)")
        distance.set_title(stats["end_reason"])
        distance.set_xlim(0, max_steps)
        distance.set_ylim(0, max_distance * 1.05)
        distance.grid(alpha=.2)
    fig.suptitle(f"Task {i+1} | Official episode {episode} | Actual executed poses\n"
                 "Gray: observed directed graph projected to X/Z; overlapping heights are possible. Privileged diagnostic.", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, .88], h_pad=3.0)
    fig.savefig(output / f"task_{i:04d}_routes.png", dpi=150)
    plt.close(fig)
    return output / f"task_{i:04d}_routes.png"
