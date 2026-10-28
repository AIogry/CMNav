"""Frozen start plans and metadata for goal-independent exploration.

Creating or validating a plan reads the configured sources and asset hashes;
it never initializes the simulator or executes a collection policy.
"""
from __future__ import annotations

import hashlib
import json
import random
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cmnav.configuration.exploration import data_path
from cmnav.data.pointnav import SCENES_PATH_PREFIX, read_episode_dataset, validate_pose
from cmnav.runtime.artifacts import (
    atomic_write_json,
    canonical_json_bytes,
    require_within,
    sha256_file,
)


def _plan_payload(config: dict[str, Any]) -> dict[str, Any]:
    """Deterministically rebuild selection and fingerprints before collection."""
    dataset = config["environment_config"]["dataset"]
    source = Path(dataset["episodes_path"])
    scene_root = Path(dataset["scenes_dir"])
    root = read_episode_dataset(source)
    files = {source}
    template = root.get("content_scenes_path", "{data_path}/content/{scene}.json.gz")
    selected = sorted(set(dataset["scene_ids"]))
    rng = random.Random(config["seed"])
    starts = []
    for scene in selected:
        candidates = [episode for episode in root["episodes"]
                      if isinstance(episode, dict)
                      and Path(str(episode.get("scene_id", ""))).stem == scene]
        try:
            shard = require_within(Path(template.format(data_path=str(source.parent), scene=scene)),
                                   source.parent)
        except (AttributeError, KeyError, IndexError) as exc:
            raise ValueError("Invalid content_scenes_path") from exc
        if shard.exists():
            files.add(shard)
            for episode in read_episode_dataset(shard)["episodes"]:
                if (not isinstance(episode, dict)
                        or Path(str(episode.get("scene_id", ""))).stem != scene):
                    raise ValueError(f"Scene mismatch in {shard}")
                candidates.append(episode)
        by_id = {}
        for episode in candidates:
            identity = str(episode.get("episode_id", ""))
            if not identity or identity in by_id:
                raise ValueError(f"Missing/duplicate episode ID in scene {scene}")
            by_id[identity] = episode
        count = config["collection"]["trajectories_per_scene"]
        if len(by_id) < count:
            raise ValueError(f"Scene {scene} has {len(by_id)} starts; need {count}")
        for identity in rng.sample(sorted(by_id), count):
            episode = by_id[identity]
            raw_scene = episode["scene_id"]
            relative = raw_scene.removeprefix(SCENES_PATH_PREFIX)
            asset = require_within(scene_root / relative, scene_root)
            if not asset.is_file():
                raise ValueError(f"Missing scene: {asset}")
            # Pin the mesh and available navigation/semantic resources as well.
            files.add(asset)
            for sibling in (asset.with_suffix(".navmesh"), asset.with_suffix(".house"),
                            asset.with_name(asset.stem + "_semantic.ply")):
                if sibling.is_file():
                    files.add(require_within(sibling, scene_root))
            index = len(starts)
            starts.append({
                "trajectory_id": f"traj_{index:06d}", "scene_key": scene,
                "scene_id": str(asset), "source_episode_id": identity,
                "start_position": validate_pose(episode.get("start_position"), 3, "start_position"),
                "start_rotation": validate_pose(episode.get("start_rotation"), 4, "start_rotation"),
                "policy_seed": (config["seed"] + index) % 2**32,
            })
    if dataset.get("scene_dataset_config"):
        files.add(Path(dataset["scene_dataset_config"]))
    return {
        "protocol": config["protocol"], "purpose": config["purpose"],
        "config_sha256": hashlib.sha256(canonical_json_bytes(config)).hexdigest(),
        "selection": "seeded_sample_without_replacement_from_official_train_episodes",
        "selection_version": 1,
        "selection_scope": "episode IDs; distinct IDs may share physical start poses",
        "budget_actions": len(starts) * config["collection"]["steps_per_trajectory"],
        "fingerprints": {str(path): sha256_file(path) for path in sorted(files)},
        "starts": starts,
    }


def create_collection_plan(config: dict[str, Any]) -> Path:
    """Write an audit-only manifest; this command never initializes Habitat."""
    plan = {"schema_version": 1, "plan_id": uuid.uuid4().hex,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "access": "simulator_setup_and_audit_only", "payload": _plan_payload(config)}
    path = data_path(Path(config["project_root"]), config["collection"]["plan_dir"])
    path = path / f"{plan['plan_id']}.json"
    atomic_write_json(path, plan)
    return path


def trajectory_metadata(config: dict[str, Any], start: dict[str, Any], plan_hash: str,
                        collection_id: str) -> dict[str, Any]:
    """Describe learning-visible collection semantics without privileged poses."""
    environment = config["environment_config"]["environment"]
    return {
        "trajectory_id": start["trajectory_id"], "scene_id": start["scene_key"],
        "collection_id": collection_id,
        "source_episode_id": start["source_episode_id"],
        "start_selection": "seeded official train episode IDs without replacement",
        "collection_protocol": config["protocol"], "purpose": config["purpose"],
        "policy": config["policy"], "policy_seed": start["policy_seed"],
        "plan_sha256": plan_hash,
        "requested_actions": config["collection"]["steps_per_trajectory"],
        "sensor_spec": environment["rgbd"],
        "sensor_interpretation": {
            "projection": "pinhole", "hfov_unit": "horizontal degrees",
            "depth_storage": "Habitat metric depth clipped to configured min/max; unnormalized",
            "depth_validity": "nonpositive is unknown; max_depth may be saturated",
            "pixel_intrinsics": "derive under an explicitly documented pixel convention",
        },
        "coordinate_convention": {"body_axes": "x right, y up, -z forward",
                                  "camera_translation_body_m":
                                  [0.0, environment["rgbd"]["sensor_height"], 0.0],
                                  "camera_rotation_body_euler_rad": [0.0, 0.0, 0.0]},
        "action_spec": {"forward_step_m": environment["forward_step_size"],
                        "turn_angle_degrees": environment["turn_angle"],
                        "actions": ["MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"],
                        "displacement": "commanded; actual displacement may differ"},
    }


def load_collection_plan(config: dict[str, Any], plan_path: str | Path
                         ) -> tuple[dict[str, Any], str]:
    """Validate the frozen selection/assets and return its exact file hash."""
    plan_path = data_path(Path(config["project_root"]), plan_path)
    plan_bytes = plan_path.read_bytes()
    plan = json.loads(plan_bytes)
    if (not isinstance(plan, dict) or plan.get("schema_version") != 1
            or plan.get("access") != "simulator_setup_and_audit_only"
            or plan.get("payload") != _plan_payload(config)):
        raise ValueError("Plan/config/assets mismatch; explicitly generate a new plan")
    return plan, hashlib.sha256(plan_bytes).hexdigest()
