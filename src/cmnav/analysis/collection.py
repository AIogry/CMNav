"""Explicitly privileged collection audits and descriptive saved-pose metrics.

Simulator truth is audit-only. Validation replays the saved RGB-D policy;
quality summaries do not resample trajectories or affect learning inputs.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from cmnav.policies.depth_walk import DepthWalkPolicy
from cmnav.data.trajectories import TrajectoryReader
from cmnav.analysis.records import (read_json, read_rows, digest, require,
                                    implementation_provenance, capture_analysis_sources)
from cmnav.analysis.statistics import longest_run

EPSILON_M = 0.001
CELL_M = 0.5

def analyze_quality(run):
    dataset = read_json(run / "dataset.json")
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
            "implementation": implementation_provenance(Path(__file__)),
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

def inspect_run(directory):
    manifest = read_json(directory / "dataset.json")
    require(manifest["status"] == "complete", "Collection is not complete")
    plan_path = directory / "audit/plan.json"
    plan = read_json(plan_path)["payload"]
    require(digest(plan_path) == manifest["plan_sha256"], "Plan fingerprint mismatch")
    config = read_json(directory / "audit/config.json")
    budget = config["collection"]["steps_per_trajectory"]
    starts = plan["starts"]
    require(len(starts) == len(manifest["trajectories"]), "Incomplete start coverage")
    habitat = read_json(directory / "audit/habitat_config.json")["habitat"]
    require(habitat["task"]["lab_sensors"] == {}, "Goal sensor leaked into exploration")
    require(set(habitat["task"]["measurements"]) == {"collisions"}, "Unexpected task metrics")
    require(set(habitat["task"]["actions"]) == {"move_forward", "turn_left", "turn_right"},
            "Unexpected exploration actions")
    reports, signatures = [], []
    for entry, start in zip(manifest["trajectories"], starts):
        require(entry["trajectory_id"] == start["trajectory_id"], "Wrong trajectory order")
        location = (directory / entry["path"]).resolve()
        require(location.is_relative_to(directory), "Trajectory path escapes collection")
        require(digest(location / "manifest.json") == entry["manifest_sha256"],
                "Dataset/trajectory manifest mismatch")
        reader = TrajectoryReader(location)
        require(len(reader) == budget == entry["actions"], "Incorrect action budget")
        require(reader.summary["observation_count"] == budget + 1, "Not T+1 observations")
        metadata = reader.metadata
        require(metadata["scene_id"] == start["scene_key"], "Scene identity mismatch")
        require(metadata["collection_id"] == manifest["run_id"], "Collection identity mismatch")
        require(metadata["source_episode_id"] == start["source_episode_id"], "Wrong source start")
        require(metadata["plan_sha256"] == manifest["plan_sha256"], "Trajectory/plan mismatch")
        records = read_rows(directory / "audit" / f"{start['trajectory_id']}.jsonl")
        require([record["observation_index"] for record in records] == list(range(budget + 1)),
                "Audit is not aligned to all observations")
        require(np.allclose(records[0]["position"], start["start_position"], atol=1e-5, rtol=0),
                "Reset position differs from the manifest")
        actual_q = np.array(records[0]["rotation_xyzw"])
        target_q = np.array(start["start_rotation"])
        require(min(np.linalg.norm(actual_q - target_q), np.linalg.norm(actual_q + target_q)) < 1e-5,
                "Reset orientation differs from the manifest")
        require(records[0]["collisions"]["count"] == 0, "Collision count was not reset")
        frame_hashes, rgb_hashes, depth_hashes, preview = [], [], [], []
        preview_indices = set(np.linspace(0, budget, min(budget + 1, 200), dtype=int).tolist())
        depth_min, depth_max = float("inf"), -float("inf")
        spec = metadata["sensor_spec"]
        for index in range(budget + 1):
            obs = reader.observation(index)
            require(set(obs) == {"rgb", "depth"}, "Unexpected learning observation field")
            require(obs["rgb"].shape == (spec["height"], spec["width"], 3), "RGB shape mismatch")
            require(obs["depth"].shape == (spec["height"], spec["width"]), "Depth shape mismatch")
            require(obs["rgb"].dtype == np.uint8 and obs["depth"].dtype == np.float32,
                    "Unexpected sensor dtype")
            require(np.isfinite(obs["depth"]).all(), "Nonfinite depth")
            depth_min = min(depth_min, float(obs["depth"].min()))
            depth_max = max(depth_max, float(obs["depth"].max()))
            rgb_hash = hashlib.sha256(obs["rgb"].tobytes()).hexdigest()
            depth_hash = hashlib.sha256(obs["depth"].tobytes()).hexdigest()
            rgb_hashes.append(rgb_hash)
            depth_hashes.append(depth_hash)
            frame_hashes.append([rgb_hash, depth_hash])
            if index in preview_indices:
                preview.append(Image.fromarray(obs["rgb"]))
        require(depth_min >= spec["min_depth"] and depth_max <= spec["max_depth"],
                "Depth outside configured metric interval")
        require(len(set(rgb_hashes)) > 1, "This verification trajectory has no RGB change")
        policy = DepthWalkPolicy(metadata["policy_seed"], metadata["policy"])
        actions = []
        for index, transition in enumerate(reader.transitions()):
            require(transition["collection_id"] == manifest["run_id"]
                    and transition["trajectory_id"] == start["trajectory_id"], "Lost provenance")
            require(transition["observation_index"] == index
                    and transition["next_observation_index"] == index + 1, "Broken alignment")
            require(transition["action"] == policy.act(transition["obs"]),
                    "Action does not match RGB-D-only seeded policy replay")
            require(not transition["terminated"], "Exploration unexpectedly terminated")
            require(transition["truncated"] == (index == budget - 1), "Wrong truncation boundary")
            require(transition["end_reason"] == ("collector_budget" if index == budget - 1 else None),
                    "Wrong boundary reason")
            require(transition["action"] != "STOP", "Artificial STOP in exploration")
            actions.append(transition["action"])
        window_count = 0
        for offset, window in enumerate(reader.windows(8, 4)):
            require([item["index"] for item in window] == list(range(offset * 4, offset * 4 + 8)),
                    "Window crosses or loses trajectory indices")
            window_count += 1
        require(window_count == max(0, (budget - 8) // 4 + 1), "Wrong offline window count")
        positions = np.array([record["position"] for record in records])
        require(np.isfinite(positions).all(), "Nonfinite audit pose")
        distance = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
        require(distance > 0, "This verification trajectory did not translate")
        preview_path = directory / "audit" / f"{start['trajectory_id']}.preview.gif"
        preview[0].save(preview_path, save_all=True, append_images=preview[1:], duration=200, loop=0)
        reports.append({
            "trajectory_id": start["trajectory_id"], "source_episode_id": start["source_episode_id"],
            "actions": len(reader), "observations": budget + 1, "action_counts": dict(Counter(actions)),
            "rgb_shape": [spec["height"], spec["width"], 3], "depth_dtype": "float32",
            "depth_range_m": [depth_min, depth_max], "unique_rgb_frames": len(set(rgb_hashes)),
            "unique_depth_frames": len(set(depth_hashes)), "offline_windows_length8_stride4": window_count,
            "reset_matches_manifest": True, "policy_replay_matches": True,
            "budget_truncated_without_stop": True, "rgbd_only_learning_observations": True,
            "audit_path_length_m": distance, "audit_collision_count": records[-1]["collisions"]["count"],
            "preview": str(preview_path), "preview_observation_indices": sorted(preview_indices),
        })
        signatures.append({"source_episode_id": start["source_episode_id"], "actions": actions,
                           "rgbd_frame_hashes": frame_hashes, "audit_records": records})
    total_actions = sum(item["actions"] for item in reports)
    require(total_actions == manifest["expected_actions"] == plan["budget_actions"], "Total budget differs")
    return {"run_directory": str(directory), "status": "passed", "plan_sha256": manifest["plan_sha256"],
            "total_actions": total_actions, "total_observations": sum(item["observations"] for item in reports),
            "trajectories": reports}, signatures

def audit_main(argv=None):
    parser = argparse.ArgumentParser(description="Validate saved exploration trajectories and optional exact repeats.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--compare-run", type=Path)
    args = parser.parse_args(argv)
    directory = args.run_dir.resolve()
    report, signatures = inspect_run(directory)
    if args.compare_run:
        repeat, repeat_signatures = inspect_run(args.compare_run.resolve())
        require(report["plan_sha256"] == repeat["plan_sha256"], "Runs use different plans")
        require(signatures == repeat_signatures, "Repeated run differs in actions, RGB-D, or audit poses")
        report["repeat_run"] = repeat
        report["repeat_exact_match"] = {"actions": True, "rgbd_arrays": True, "audit_records": True}
    report["audit_script_sha256"] = digest(Path(__file__))
    report["analysis_implementation"] = implementation_provenance(Path(__file__))
    report["analysis_source_snapshot"] = capture_analysis_sources(directory / "audit")
    output = directory / "audit/validation_report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    print(f"Validation report: {output}")

def quality_main(argv=None):
    parser = argparse.ArgumentParser(description="Summarize audit-only saved collection quality.")
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace this entrypoint's audit report/plot only")
    args = parser.parse_args(argv)
    run = args.run_directory.resolve()
    destination = run / "audit" / "quality_report.json"
    plot_destination = run / "audit" / "trajectory_paths_xz.png"
    if not args.overwrite and (destination.exists() or plot_destination.exists()):
        parser.error("Diagnostic output exists; pass --overwrite to replace it explicitly")
    result = analyze_quality(run)
    result["analysis"]["source_snapshot"] = capture_analysis_sources(run / "audit")
    with destination.open("w" if args.overwrite else "x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    from cmnav.reporting.collection import plot_paths
    plot = plot_paths(run, result)
    print(json.dumps({"report": str(destination), "plot": str(plot), "aggregate": result["aggregate"],
                      "trajectories": result["trajectories"]}, ensure_ascii=False, indent=2))
