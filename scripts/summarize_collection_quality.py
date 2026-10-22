"""Audit-only diagnostics for a completed collection, never learning inputs.

Usage: python scripts/summarize_collection_quality.py <run-directory>
Use --overwrite to explicitly replace only this script's diagnostic outputs.
The collector, recorded trajectories, config, and start selection stay unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

EPSILON_M = 0.001
CELL_M = 0.5


def longest_run(values):
    best_start, best_length = None, 0
    current_start, current_length = None, 0
    for index, value in enumerate(values):
        if value:
            if current_start is None:
                current_start = index
            current_length += 1
            if current_length > best_length:
                best_start, best_length = current_start, current_length
        else:
            current_start, current_length = None, 0
    return {"start_transition_index": best_start, "length_actions": best_length}


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(run):
    dataset = json.loads((run / "dataset.json").read_text())
    if dataset["status"] != "complete":
        raise ValueError("Only analyze completed collections")
    reports, union_cells, files, all_positions = [], set(), {}, []
    for entry in dataset["trajectories"]:
        trajectory_id = entry["trajectory_id"]
        audit_path = run / "audit" / f"{trajectory_id}.jsonl"
        transition_path = run / entry["path"] / "transitions.jsonl"
        audits, transitions = read_rows(audit_path), read_rows(transition_path)
        files[str(audit_path.relative_to(run))] = digest(audit_path)
        files[str(transition_path.relative_to(run))] = digest(transition_path)
        if len(audits) != len(transitions) + 1:
            raise ValueError(f"Audit alignment mismatch: {trajectory_id}")
        positions = [row["position"] for row in audits]
        all_positions.extend(positions)
        for index, row in enumerate(audits):
            assert row["observation_index"] == index
            assert len(row["position"]) == 3 and all(math.isfinite(v) for v in row["position"])
        for index, row in enumerate(transitions):
            assert row["index"] == index and row["observation_index"] == index
            assert row["next_observation_index"] == index + 1
        actions = [row["action"] for row in transitions]
        distances = [math.dist(before, after) for before, after in zip(positions, positions[1:])]
        forward_indices = [index for index, action in enumerate(actions) if action == "MOVE_FORWARD"]
        forward_distances = [distances[index] for index in forward_indices]
        collision_flags = [bool(row["collisions"]["is_collision"]) for row in audits[1:]]
        cumulative_counts = [row["collisions"]["count"] for row in audits]
        collision_deltas = [after - before for before, after in zip(cumulative_counts, cumulative_counts[1:])]
        assert all(delta == int(flag) for delta, flag in zip(collision_deltas, collision_flags))
        failed_forward = [distance <= EPSILON_M for distance in forward_distances]
        failed_run_filtered = longest_run(failed_forward)
        filtered_start = failed_run_filtered.pop("start_transition_index")
        failed_run_filtered["start_forward_attempt_index"] = filtered_start
        if filtered_start is not None:
            failed_run_filtered["start_transition_index"] = forward_indices[filtered_start]
            failed_run_filtered["end_transition_index_inclusive"] = forward_indices[
                filtered_start + failed_run_filtered["length_actions"] - 1]
        failed_run_filtered["length_forward_attempts"] = failed_run_filtered.pop("length_actions")
        cells = [(math.floor(p[0] / CELL_M), math.floor(p[2] / CELL_M)) for p in positions]
        union_cells.update(cells)
        prefix_cells = []
        seen = set()
        for observation_index, cell in enumerate(cells):
            seen.add(cell)
            if observation_index % 100 == 0 or observation_index == len(transitions):
                prefix_cells.append({"after_actions": observation_index, "distinct_cells": len(seen)})
        forward_collisions = sum(collision_flags[i] for i in forward_indices)
        turn_collisions = sum(flag for action, flag in zip(actions, collision_flags) if action != "MOVE_FORWARD")
        no_translation = [distance <= EPSILON_M for distance in distances]
        turn_flags = [action in {"TURN_LEFT", "TURN_RIGHT"} for action in actions]
        consecutive_failed = [action == "MOVE_FORWARD" and distance <= EPSILON_M
                              for action, distance in zip(actions, distances)]
        no_translation_span = longest_run(no_translation)
        if no_translation_span["start_transition_index"] is not None:
            start = no_translation_span["start_transition_index"]
            stop = start + no_translation_span["length_actions"]
            no_translation_span["action_counts"] = dict(Counter(actions[start:stop]))
        report = {
            "trajectory_id": trajectory_id,
            "action_counts": dict(Counter(actions)),
            "actions": len(actions), "observations": len(audits),
            "collision_flags_total": sum(collision_flags),
            "forward_attempts": len(forward_indices),
            "forward_collisions": forward_collisions,
            "forward_collision_rate": forward_collisions / len(forward_indices) if forward_indices else None,
            "nonforward_collision_flags": turn_collisions,
            "no_translation_forward_attempts": sum(failed_forward),
            "no_translation_forward_fraction": sum(failed_forward) / len(forward_indices) if forward_indices else None,
            "longest_contiguous_failed_forward_actions": longest_run(consecutive_failed),
            "longest_failed_forward_attempt_streak_ignoring_turns": failed_run_filtered,
            "longest_no_translation_action_span_including_turns": no_translation_span,
            "longest_turn_only_action_span": longest_run(turn_flags),
            "path_length_3d_m": sum(distances),
            "forward_path_length_3d_m": sum(forward_distances),
            "mean_forward_displacement_m": sum(forward_distances) / len(forward_distances) if forward_distances else None,
            "end_to_start_displacement_3d_m": math.dist(positions[0], positions[-1]),
            "max_distance_from_start_3d_m": max(math.dist(positions[0], p) for p in positions),
            "position_min_xyz_m": [min(p[axis] for p in positions) for axis in range(3)],
            "position_max_xyz_m": [max(p[axis] for p in positions) for axis in range(3)],
            "span_xyz_m": [max(p[axis] for p in positions) - min(p[axis] for p in positions) for axis in range(3)],
            "distinct_horizontal_cells": len(set(cells)),
            "cumulative_distinct_horizontal_cells": prefix_cells,
        }
        reports.append(report)
    total_forward = sum(row["forward_attempts"] for row in reports)
    total_collision = sum(row["forward_collisions"] for row in reports)
    total_no_translation_forward = sum(row["no_translation_forward_attempts"] for row in reports)
    return {
        "schema_version": 1, "run_id": dataset["run_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "access": "audit_only_not_learning_input",
        "analysis": {
            "script_path": str(Path(__file__).resolve()),
            "script_sha256": digest(Path(__file__)),
            "script_version": "1",
            "python_version": sys.version,
        },
        "definitions": {
            "source": "Saved simulator-truth poses/collisions joined with recorded actions; post-collection only",
            "forward_collision_rate": "Forward MOVE_FORWARD actions whose post-action collision flag is true / total MOVE_FORWARD attempts; turns are not in denominator",
            "path_length": "Sum of 3D Euclidean displacement between consecutive audited poses, in metres",
            "no_translation_threshold_m": EPSILON_M,
            "failed_forward_attempt": "MOVE_FORWARD with 3D translation <= threshold; turns never count as failed movement",
            "failed_forward_streak_ignoring_turns": "Consecutive failures in forward-attempt subsequence; intervening turns excluded, successful forward resets streak",
            "no_translation_span": "Consecutive actions with <= threshold translation, INCLUDING rotations; not by itself proof of a stuck policy",
            "horizontal_cell_size_m": CELL_M,
            "horizontal_cells": "floor(x/cell_size), floor(z/cell_size), origin (0,0); uses observation endpoints; ignores height and visibility",
            "cell_interpretation": "Descriptive location diversity only; NOT navigable-area, observed-surface, room, or map coverage; grid alignment and repeated/same-height floors can affect counts",
            "threshold_interpretation": "Diagnostic thresholds only, not a prespecified dataset acceptance rule; no trajectories discarded or resampled",
        },
        "aggregate": {
            "trajectories": len(reports),
            "actions": sum(row["actions"] for row in reports),
            "observations": sum(row["observations"] for row in reports),
            "action_counts": dict(sum((Counter(row["action_counts"]) for row in reports), Counter())),
            "forward_attempts": total_forward,
            "forward_collisions": total_collision,
            "forward_collision_rate": total_collision / total_forward if total_forward else None,
            "no_translation_forward_attempts": total_no_translation_forward,
            "no_translation_forward_fraction": total_no_translation_forward / total_forward if total_forward else None,
            "path_length_3d_m": sum(row["path_length_3d_m"] for row in reports),
            "distinct_horizontal_cells_union": len(union_cells),
            "distinct_horizontal_cells_per_trajectory_sum": sum(row["distinct_horizontal_cells"] for row in reports),
            "span_xyz_m": [max(p[axis] for p in all_positions) - min(p[axis] for p in all_positions) for axis in range(3)],
        },
        "trajectories": reports,
        "input_sha256": files,
    }


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
    ax.set_title("MP3D 17DRP5sb8fy: 10 exploration trajectories\nAudit pose projection, 500 actions each", fontsize=12)
    fig.text(0.5, 0.02, "Ground-truth diagnostics only. Projection omits height; no floor layout or area coverage is implied.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    destination = run / "audit" / "trajectory_paths_xz.png"
    fig.savefig(destination, dpi=160)
    plt.close(fig)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace this script's audit report/plot only")
    args = parser.parse_args()
    run = args.run_directory.resolve()
    destination = run / "audit" / "quality_report.json"
    plot_destination = run / "audit" / "trajectory_paths_xz.png"
    if not args.overwrite and (destination.exists() or plot_destination.exists()):
        parser.error("Diagnostic output exists; pass --overwrite to replace it explicitly")
    result = analyze(run)
    with destination.open("w" if args.overwrite else "x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    plot = plot_paths(run, result)
    print(json.dumps({"report": str(destination), "plot": str(plot), "aggregate": result["aggregate"],
                      "trajectories": result["trajectories"]}, ensure_ascii=False, indent=2))
