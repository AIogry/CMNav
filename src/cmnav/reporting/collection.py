"""Saved collection pose plots with no policy or dataset modification."""
from __future__ import annotations

import os
from cmnav.analysis.records import read_rows

def plot_paths(run, result):
    os.environ["MPLCONFIGDIR"] = str(run.parents[1] / "cache" / "matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.get_cmap("tab10")
    for index, trajectory in enumerate(result["trajectories"]):
        trajectory_id = trajectory["trajectory_id"]
        positions = [row["position"] for row in read_rows(run / "audit" / f"{trajectory_id}.jsonl")]
        xs, zs = [p[0] for p in positions], [p[2] for p in positions]
        color = colors(index % 10)
        ax.plot(xs, zs, color=color, alpha=0.8, linewidth=1.0,
                label=f"{trajectory_id} ({trajectory['path_length_3d_m']:.1f} m)")
        ax.scatter(xs[0], zs[0], marker="^", color=[color], edgecolor="black", linewidth=0.6, s=70, zorder=3)
        ax.scatter(xs[-1], zs[-1], marker="s", color=[color], edgecolor="white", linewidth=0.6, s=32, zorder=3)
    handles, labels = ax.get_legend_handles_labels()
    handles.extend([Line2D([], [], marker="^", color="black", linestyle="None", markersize=7),
                    Line2D([], [], marker="s", color="black", linestyle="None", markersize=5)])
    labels.extend(["Start", "End"])
    ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8, frameon=False)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("World X (m)")
    ax.set_ylabel("World Z (m)")
    ax.grid(True, linewidth=0.5, alpha=0.25)
    ax.set_title(f"Collection {result['run_id']}: {result['aggregate']['trajectories']} exploration trajectories\n"
                 f"Audit pose projection, {result['aggregate']['actions']} total actions", fontsize=12)
    fig.text(0.5, 0.02, "Ground-truth diagnostics only. Projection omits height; no floor layout or area coverage is implied.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    destination = run / "audit" / "trajectory_paths_xz.png"
    fig.savefig(destination, dpi=160)
    plt.close(fig)
    return destination
