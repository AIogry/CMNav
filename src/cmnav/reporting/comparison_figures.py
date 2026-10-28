"""Figures for dimension, checkpoint, gate and readout comparisons."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from cmnav.analysis.report_contracts import NOISE_CONDITIONS
from cmnav.reporting.labels import GCML_LABELS as LABELS, GCML_COLORS as COLORS


def dimension_figures(directory: Path, data: dict[str, Any]) -> list[str]:
    directory.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    dimensions, methods = data["dimensions"], data["methods"]
    settings, planning = data["shared_settings"], data["planning_settings"]
    task_count = settings["task_count"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for dim in dimensions:
        rows = data["results"][str(dim)]["training"]["epochs_detail"]
        for axis, key, title in zip(axes, ("prediction_mse", "inverse_mse", "affordance_mse"),
                                    ("Q/V transition error", "W inverse error", "G affordance error")):
            axis.semilogy([row["epoch"] for row in rows], [max(row[key], 1e-20) for row in rows], label=f"d={dim}")
            axis.set(title=title, xlabel="Epoch", ylabel="Mean batch pre-update MSE")
            axis.grid(alpha=.25)
            axis.legend()
    fig.suptitle(f"Frozen batches, {settings['epochs']} epochs, seed {settings['seed']}; dimension comparison")
    fig.tight_layout()
    curve = directory / "training_curves.png"
    fig.savefig(curve, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    x = np.arange(len(dimensions))
    width = .8 / len(methods)
    for index, method in enumerate(methods):
        success = [sum(row["metrics"]["success"] for row in data["results"][str(dim)]["tasks"] if row["method"] == method)
                   for dim in dimensions]
        axes[0, 0].bar(x + (index - (len(methods) - 1) / 2) * width, success, width,
                       label=LABELS[method], color=COLORS[method])
    axes[0, 0].set(title="Full-goal navigation", ylabel=f"Successful tasks / {task_count}", ylim=(0, task_count + .5), xticks=x,
                    xticklabels=[str(dim) for dim in dimensions], xlabel="Dimension")
    axes[0, 0].legend(fontsize=8)
    for index, (group, metric, label, color) in enumerate((
            ("supervised_sources", "false_positive", "Source FP", "#d17a22"),
            ("supervised_sources", "false_negative", "Source FN", "#c73e33"),
            ("graph_sinks", "false_positive", "Sink FP", "#7469a9"))):
        counts = [data["results"][str(dim)]["readout"][group][metric] for dim in dimensions]
        axes[0, 1].bar(x + (index - 1) * .24, counts, .24, label=label, color=color)
    axes[0, 1].set(title=f"G errors at threshold {planning['gcml_affordance_threshold']:g}", ylabel="Error count (symlog)", xticks=x,
                    xticklabels=[str(dim) for dim in dimensions], xlabel="Dimension")
    axes[0, 1].set_yscale("symlog", linthresh=1)
    axes[0, 1].legend(fontsize=8)
    for key, label in (("accuracy", "All reference states"), ("branching_accuracy", "Multiple outgoing edges")):
        scores = [data["results"][str(dim)]["long_goal_W"]["summary"][key] for dim in dimensions]
        axes[1, 0].plot(dimensions, scores, "o-", label=label)
    axes[1, 0].set(title="W legal argmax follows a cost-optimal edge", ylabel="Accuracy", xlabel="Dimension", ylim=(0, 1.05))
    axes[1, 0].legend(fontsize=8)
    for method in methods:
        if method not in {"gcml", "gcml_graph_affordance"}:
            continue
        lengths = []
        for dim in dimensions:
            values = [row["first_candidates"]["lengths"]["mean"] for row in data["results"][str(dim)]["tasks"]
                      if row["method"] == method and row["first_candidates"]["lengths"]["mean"] is not None]
            lengths.append(float(np.mean(values)) if values else np.nan)
        axes[1, 1].plot(dimensions, lengths, "o-", color=COLORS[method], label=LABELS[method])
    axes[1, 1].set(title="First planning decision: candidate length", ylabel="Mean imagined edges", xlabel="Dimension")
    axes[1, 1].legend(fontsize=8)
    for axis in axes.flat:
        axis.grid(axis="y", alpha=.2)
    fig.suptitle(f"Same {task_count} tasks; one training seed per dimension")
    fig.tight_layout()
    comparison = directory / "dimension_comparison.png"
    fig.savefig(comparison, dpi=180)
    plt.close(fig)
    return [str(curve), str(comparison)]


def plot_training(study: dict[str, Any], destination: Path) -> None:
    history = study["training"]["epochs_detail"]
    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(2, 3, figsize=(15, 8))
    for axis, metric, title in zip(axes[0], ("prediction_mse", "inverse_mse", "affordance_mse"),
                                    ("Q/V transition loss", "W inverse loss", "Raw G affordance loss")):
        for weighted, style, label in ((False, "-", "Mean of trajectory means"), (True, "--", "Transition weighted")):
            values = [row.get("transition_weighted", {}).get(metric) if weighted else row.get(metric) for row in history]
            array = np.array([float(value) if value is not None and value > 0 else np.nan for value in values])
            if np.isfinite(array).any():
                axis.semilogy(epochs, array, style, label=label)
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("MSE (pre-update batches)")
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    for name in ("Q", "V", "W", "G"):
        values = [row.get("parameter_frobenius_norms", {}).get(name, np.nan) for row in history]
        axes[1, 0].plot(epochs, values, label=name)
    axes[1, 0].set_title("Parameter Frobenius norms")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].legend()
    for variant, style in (("raw", "-"), ("minimum_change", "--"), ("minimum_norm", ":")):
        for key, marker in (("fp", "o"), ("fn", "s")):
            values = [point["readout"]["variants"][variant]["all_states"][key] for point in study["checkpoints"]]
            axes[1, 1].plot([point["epoch"] for point in study["checkpoints"]], values, style + marker,
                            label=f"{variant}: {key.upper()}", alpha=.8)
    axes[1, 1].set_title("Frozen-checkpoint G errors on all known states")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Count (source + sink labels)")
    axes[1, 1].legend(fontsize=7)
    for name in ("W", "G"):
        for bound, style in (("min", "-"), ("max", "--")):
            values = [row.get(f"effective_readout_rate_{bound}", {}).get(name, np.nan) for row in history]
            axes[1, 2].plot(epochs, values, style, label=f"{name} {bound}")
    axes[1, 2].set_title("Effective readout step sizes")
    axes[1, 2].set_xlabel("Epoch")
    axes[1, 2].legend(fontsize=8)
    for axis in axes[1]:
        axis.grid(alpha=.2)
    figure.suptitle(f"GCML d={study['protocol']['dimension']} | one CUDA training seed {study['protocol']['training_seed']}\n"
                   "Training-batch losses and frozen-checkpoint readout diagnostics are different measurements", fontsize=12)
    figure.tight_layout(rect=[0, 0, 1, .94])
    figure.savefig(destination, dpi=150, facecolor="white")
    plt.close(figure)


def plot_checkpoints(study: dict[str, Any], groups: list[dict[str, Any]], destination: Path) -> None:
    lookup = {(group["condition"]["name"], group["condition"]["epoch"]): group for group in groups}
    seed = study["protocol"]["planning_seeds"][0]
    epochs = study["protocol"]["checkpoints"]
    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    for name in study["protocol"]["checkpoint_conditions"]:
        values = [next(block["successes"] for block in lookup[name, epoch]["per_seed"] if block["planning_seed"] == seed) for epoch in epochs]
        axis.plot(epochs, values, marker="o", label=name)
    axis.set(xlabel="Training epoch", ylabel="Successful fixed tasks / 5", ylim=(-.15, 5.3), yticks=range(6), xticks=epochs)
    axis.grid(alpha=.2)
    axis.legend(fontsize=9)
    axis.set_title(f"Checkpoint progression | one planning seed {seed}\nSame five full-goal tasks; no teacher subgoals; no selected best epoch")
    figure.tight_layout()
    figure.savefig(destination, dpi=150, facecolor="white")
    plt.close(figure)


def plot_mechanisms(study: dict[str, Any], groups: list[dict[str, Any]], destination: Path) -> None:
    final = {group["condition"]["name"]: group for group in groups if group["condition"]["epoch"] == study["protocol"]["epochs"]}
    figure, axes = plt.subplots(2, 2, figsize=(15, 9))
    colors = ("#277da1", "#f8961e", "#43aa8b", "#9b5de5")
    for index, seed in enumerate(study["protocol"]["planning_seeds"]):
        x, y = [], []
        for sigma, name in NOISE_CONDITIONS:
            seed_block = next((block for block in final[name]["per_seed"] if block["planning_seed"] == seed), None)
            if seed_block:
                x.append(sigma)
                y.append(seed_block["successes"])
        axes[0, 0].plot(x, y, "o-", label=f"Planning seed {seed}", alpha=.8, color=colors[index % len(colors)])
    axes[0, 0].set_title("Noise sweep | minimum-change fitted G")
    axes[0, 0].set_xlabel("Gaussian noise scale")
    axes[0, 0].legend(fontsize=8)

    def bars(axis: Any, names: list[str], labels: list[str], title: str) -> None:
        x = np.arange(len(names))
        means = [final[name]["mean_successes_per_seed"] for name in names]
        axis.bar(x, means, color="#81a6bd", alpha=.75, width=.65)
        for j, name in enumerate(names):
            values = [block["successes"] for block in final[name]["per_seed"]]
            offsets = np.linspace(-.17, .17, len(values)) if len(values) > 1 else [0.]
            for k, (offset, value) in enumerate(zip(offsets, values)):
                axis.scatter(j + offset, value, s=24, color=colors[k % len(colors)], zorder=3)
        axis.set_xticks(x)
        axis.set_xticklabels(labels, fontsize=8)
        axis.set_title(title)

    bars(axes[0, 1], ["raw_imagination", "raw_pure_imagination", "fit_imagination", "fit_pure_imagination"],
         ["Raw G\nobserved first", "Raw G\nlearned first", "Fitted G\nobserved first", "Fitted G\nlearned first"],
         "First-action information permission | noise 0.2")
    bars(axes[1, 0], ["graph_greedy", "raw_learned_greedy", "fit_learned_greedy", "fit_imagination", "minimum_norm_imagination", "fit_graph_imagination"],
         ["Graph\ngreedy", "Raw G\ngreedy", "Fitted G\ngreedy", "Min-change\nimagination", "Min-norm\nimagination", "Graph aided\nimagination"],
         "Greedy controls and alternative readouts")
    bars(axes[1, 1], ["fit_noise_0.05", "fit_snap_0.05", "fit_cache_0.05", "fit_snap_cache_0.05"],
         ["Neither", "Snap only", "Cache only", "Snap + cache"], "Snap/cache 2 x 2 | noise 0.05")
    for axis in axes.flat:
        axis.set_ylim(-.15, 5.35)
        axis.set_yticks(range(6))
        axis.set_ylabel("Successful fixed tasks / 5")
        axis.grid(axis="y", alpha=.2)
        axis.set_axisbelow(True)
    figure.suptitle("Final checkpoint mechanism comparisons\nBars: mean successful tasks per planning seed; dots: seed repeats of the same five tasks (not independent tasks)", fontsize=12)
    figure.tight_layout(rect=[0, 0, 1, .94], h_pad=3)
    figure.savefig(destination, dpi=150, facecolor="white")
    plt.close(figure)


def readout_figure(directory: Path, data: dict[str, Any]) -> Path:
    manifest, initial, groups, rates = (data[key] for key in ("manifest", "initial", "groups", "rates"))
    palette = ("#0072B2", "#D55E00", "#009E73")
    colors = {rate: palette[index % len(palette)] for index, rate in enumerate(rates)}
    threshold = initial["sources"]["threshold"]
    panels = (("sources", "positive_rmse", "Source positive-label RMSE"),
              ("sources", "fn", f"Missed source edges at threshold {threshold:g}"),
              ("sinks", "fp", f"Sink false positives at threshold {initial['sinks']['threshold']:g}"))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for mode in ("sources_only", "sources_plus_sinks"):
        for rate in rates:
            name = f"{mode}_lr_{rate:g}"
            rows = groups[name]
            label = f"{rate:g}, {'+ sinks' if mode == 'sources_plus_sinks' else 'sources only'}"
            for ax, (scope, metric, title) in zip(axes, panels):
                x = [0] + [row["epoch"] for row in rows]
                y = [initial[scope][metric]] + [row["metrics"][scope][metric] for row in rows]
                ax.plot(x, y, color=colors[rate],
                        linestyle="-" if mode == "sources_plus_sinks" else "--",
                        linewidth=1.7, label=label)
                ax.set_title(title, fontsize=11)
                ax.set_xlabel("Additional G-only epochs")
                ax.grid(alpha=.2)
    axes[0].set_yscale("log")
    axes[0].set_ylabel(f"RMSE ({initial['sources']['positive_labels']} true outgoing edges)")
    axes[1].set_ylabel(f"Count / {initial['sources']['positive_labels']} positive source labels")
    axes[2].set_ylabel(f"Count / {initial['sinks']['negative_labels']} sink-negative labels")
    axes[2].legend(fontsize=8, frameon=False)
    fig.suptitle("Frozen Q/V/W: same initial G and source replay budget", fontsize=13)
    target = directory / "readout_learning_curves.png"
    fig.savefig(target, dpi=180)
    plt.close(fig)
    return target
