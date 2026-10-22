"""Audit complete development collections, optionally compare repeat runs.

This explicitly privileged verification script reads audit/ simulator truth.
It is not a training reader or a component of the exploration policy.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from cmnav.exploration_policy import DepthWalkPolicy
from cmnav.trajectories import TrajectoryReader


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
        records = [json.loads(line) for line in
                   (directory / "audit" / f"{start['trajectory_id']}.jsonl").read_text().splitlines()]
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--compare-run", type=Path)
    args = parser.parse_args()
    directory = args.run_dir.resolve()
    report, signatures = inspect_run(directory)
    if args.compare_run:
        repeat, repeat_signatures = inspect_run(args.compare_run.resolve())
        require(report["plan_sha256"] == repeat["plan_sha256"], "Runs use different plans")
        require(signatures == repeat_signatures, "Repeated run differs in actions, RGB-D, or audit poses")
        report["repeat_run"] = repeat
        report["repeat_exact_match"] = {"actions": True, "rgbd_arrays": True, "audit_records": True}
    report["audit_script_sha256"] = digest(Path(__file__))
    output = directory / "audit/validation_report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    print(f"Validation report: {output}")


if __name__ == "__main__":
    main()
