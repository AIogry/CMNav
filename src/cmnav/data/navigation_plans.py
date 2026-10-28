"""Frozen official PointNav task subsets for same-scene development.

These manifests contain privileged start and goal geometry. They belong to the
environment service and audit layer, never to a policy's observation stream.
Both subsets come from the official training split: ``evaluation`` here is a
development holdout of tasks, not held-out-scene or official benchmark testing.
This module creates task definitions, not RGB-D samples or demonstrations.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import string
import uuid
from pathlib import Path
from typing import Any

from cmnav.configuration.environment import EXPECTED_HABITAT_VERSION, validate_config
from cmnav.configuration.exploration import data_path
from cmnav.data.pointnav import (
    MAX_JSON_BYTES,
    SCENES_PATH_PREFIX,
    read_episode_dataset,
    validate_pose,
)
from cmnav.runtime.artifacts import canonical_json_bytes, require_within, sha256_file


SCHEMA_VERSION = 1
ACCESS = "environment_service_only"
PROTOCOL = "same_scene_development"


def _positive_integer(value: Any, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _contract(base_config: dict[str, Any]) -> dict[str, Any]:
    validate_config(base_config)
    dataset = copy.deepcopy(base_config["dataset"])
    if dataset["name"] != "mp3d" or dataset["split"] != "train":
        raise ValueError("Navigation development plans require the MP3D train split")
    for scene in dataset["scene_ids"]:
        if (not scene or any(character not in string.ascii_letters + string.digits + "_-"
                             for character in scene)):
            raise ValueError("Navigation plans require explicit bare scene IDs, not wildcards")
    for key in ("episodes_path", "scenes_dir", "scene_dataset_config"):
        if key in dataset:
            path = Path(dataset[key]).expanduser()
            if not path.is_absolute():
                raise ValueError(f"dataset.{key} must be an absolute, resolved path")
            dataset[key] = str(path.resolve())
    dataset["scene_ids"] = sorted(dataset["scene_ids"])
    environment = copy.deepcopy(base_config["environment"])
    # Changing execution devices does not change task or sensor semantics.
    environment.pop("gpu_device_id")
    return {
        "dataset": dataset,
        "environment": environment,
        "environment_seed": base_config["seed"],
        "habitat_target_version": EXPECTED_HABITAT_VERSION,
        "success_distance_m": 0.2,
        "success_requires_stop": True,
        "coordinate_convention": "Habitat world: y up; body: x right, -z forward; quaternion xyzw",
    }


def _hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _asset(raw_scene: Any, scene_root: Path) -> Path:
    if not isinstance(raw_scene, str) or not raw_scene or "\x00" in raw_scene:
        raise ValueError("Episode scene_id must be a nonempty path string")
    asset = require_within(scene_root / raw_scene.removeprefix(SCENES_PATH_PREFIX), scene_root)
    if asset.suffix.lower() != ".glb" or not asset.is_file():
        raise ValueError(f"Missing MP3D .glb scene asset: {asset}")
    return asset


def _geometry(episode: dict[str, Any], asset: Path) -> dict[str, Any]:
    start = validate_pose(episode.get("start_position"), 3, "start_position")
    validate_pose(episode.get("start_rotation"), 4, "start_rotation")
    goals = episode.get("goals")
    if not isinstance(goals, list) or len(goals) != 1 or not isinstance(goals[0], dict):
        raise ValueError("Development PointNav plans require exactly one goal per task")
    goal = goals[0]
    position = validate_pose(goal.get("position"), 3, "goal.position")
    radius = goal.get("radius")
    if radius is not None and (type(radius) not in (int, float)
                               or not math.isfinite(radius) or radius <= 0):
        raise ValueError("A supplied goal radius must be a positive finite number")
    # Conservatively collapse the same geometric route even if headings/radii
    # differ. Raw headings are retained for the actual selected environment task.
    return {"scene_id": str(asset),
            "start_position": [item if item else 0.0 for item in start],
            "goal_position": [item if item else 0.0 for item in position]}


def _source_candidates(dataset: dict[str, Any]) -> tuple[list[dict[str, Any]], set[Path]]:
    source = Path(dataset["episodes_path"])
    root = read_episode_dataset(source)
    files = {source}
    template = root.get("content_scenes_path", "{data_path}/content/{scene}.json.gz")
    if not isinstance(template, str) or template.count("{scene}") != 1:
        raise ValueError("content_scenes_path must contain exactly one '{scene}' field")
    try:
        for _, field, spec, conversion in string.Formatter().parse(template):
            if field is not None and (field not in ("data_path", "scene") or spec or conversion):
                raise ValueError("Unsupported content_scenes_path field")
    except ValueError as exc:
        raise ValueError("Invalid content_scenes_path") from exc
    result = []
    selected = set(dataset["scene_ids"])
    for episode in root["episodes"]:
        if not isinstance(episode, dict):
            raise ValueError(f"Malformed episode in {source}")
        if Path(str(episode.get("scene_id", ""))).stem in selected:
            result.append({"episode": episode, "source_file": str(source)})
    for scene in dataset["scene_ids"]:
        shard = require_within(Path(template.format(data_path=str(source.parent), scene=scene)),
                               source.parent)
        if not shard.exists():
            continue
        files.add(shard)
        for episode in read_episode_dataset(shard)["episodes"]:
            if (not isinstance(episode, dict)
                    or Path(str(episode.get("scene_id", ""))).stem != scene):
                raise ValueError(f"Scene mismatch or malformed episode in {shard}")
            result.append({"episode": episode, "source_file": str(shard)})
    return result, files


def _payload(base_config: dict[str, Any], *, learning_episodes: int,
             evaluation_episodes: int, seed: int) -> dict[str, Any]:
    learning_episodes = _positive_integer(learning_episodes, "learning_episodes")
    evaluation_episodes = _positive_integer(evaluation_episodes, "evaluation_episodes")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Navigation plan seed must be an integer in [0, 2**32)")
    contract = _contract(base_config)
    dataset = contract["dataset"]
    candidates, files = _source_candidates(dataset)
    scene_root = Path(dataset["scenes_dir"])
    by_identity = {}
    for candidate in candidates:
        episode = candidate["episode"]
        identity = episode.get("episode_id")
        if type(identity) not in (str, int) or not str(identity):
            raise ValueError("Every source episode needs a nonempty string or integer ID")
        scene = Path(str(episode.get("scene_id", ""))).stem
        key = (scene, str(identity))
        if key in by_identity:
            raise ValueError(f"Duplicate source episode identity: {key}")
        asset = _asset(episode.get("scene_id"), scene_root)
        # Ensure all stored official metadata is finite JSON, not merely its pose.
        canonical_json_bytes(episode)
        by_identity[key] = {
            "source_episode_id": str(identity), "scene_key": scene,
            "source_file": candidate["source_file"],
            "task_geometry_sha256": _hash(_geometry(episode, asset)),
            "episode": copy.deepcopy(episode),
        }
    for scene in dataset["scene_ids"]:
        if not any(key[0] == scene for key in by_identity):
            raise ValueError(f"No official training tasks available for selected scene {scene}")
    # Source order cannot change the representative chosen for duplicate geometry.
    by_geometry = {}
    for key in sorted(by_identity):
        record = by_identity[key]
        by_geometry.setdefault(record["task_geometry_sha256"], record)
    count = learning_episodes + evaluation_episodes
    if len(by_geometry) < count:
        raise ValueError(f"Need {count} distinct task geometries; found {len(by_geometry)}")
    selected = random.Random(seed).sample(sorted(by_geometry), count)
    records = [by_geometry[key] for key in selected]
    for record in records:
        asset = _asset(record["episode"]["scene_id"], scene_root)
        files.add(asset)
        for sibling in (asset.with_suffix(".navmesh"), asset.with_suffix(".house"),
                        asset.with_name(asset.stem + "_semantic.ply")):
            if sibling.is_file():
                files.add(require_within(sibling, scene_root))
    if dataset.get("scene_dataset_config"):
        files.add(Path(dataset["scene_dataset_config"]))
    return {
        "protocol": PROTOCOL,
        "purpose": "pipeline_development; no official held-out generalization claim",
        "source_split": "train", "seed": seed,
        "selection_version": 1,
        "selection": "seeded_without_replacement_after_exact_start_goal_geometry_deduplication",
        "geometry_scope": "scene + exact start/goal positions; heading and radius ignored for deduplication",
        "split_scope": "disjoint source task IDs and exact (scene,start,goal) pairs; visual/spatial overlap permitted",
        "evaluation_use": "frozen development tasks; do not adapt collection to evaluation outcomes",
        "learning_episodes": learning_episodes, "evaluation_episodes": evaluation_episodes,
        "environment_contract": contract,
        "candidate_count": len(by_identity), "unique_geometry_count": len(by_geometry),
        "fingerprints": {str(path): sha256_file(path) for path in sorted(files)},
        "learning": records[:learning_episodes], "evaluation": records[learning_episodes:],
    }


def create_navigation_plan(base_config: dict[str, Any], output_dir: Path, *,
                           learning_episodes: int, evaluation_episodes: int, seed: int) -> Path:
    """Freeze task selection under project/data without importing Habitat.

    Counts are totals across the configured scenes. Repeating identical inputs
    produces the same plan ID and raises FileExistsError at the same destination;
    existing plans are never overwritten or silently replaced.
    """
    payload = _payload(base_config, learning_episodes=learning_episodes,
                       evaluation_episodes=evaluation_episodes, seed=seed)
    plan = {"schema_version": SCHEMA_VERSION, "access": ACCESS,
            "plan_id": _hash(payload), "payload": payload}
    contents = canonical_json_bytes(plan)
    if len(contents) > MAX_JSON_BYTES:
        raise ValueError(f"Navigation plan exceeds {MAX_JSON_BYTES} bytes; request fewer episodes")
    directory = data_path(Path(base_config["project_root"]), output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{plan['plan_id']}.json"
    temporary = directory / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        # Atomic publication, with no replace/overwrite even under concurrent use.
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _reject_constant(value: str) -> None:
    raise ValueError(f"Nonfinite JSON constant is not allowed: {value}")


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _validate_plan(plan: dict[str, Any], base_config: dict[str, Any]) -> None:
    if (not isinstance(plan, dict)
            or set(plan) != {"schema_version", "access", "plan_id", "payload"}
            or type(plan["schema_version"]) is not int or plan["schema_version"] != SCHEMA_VERSION
            or plan["access"] != ACCESS or not isinstance(plan["payload"], dict)):
        raise ValueError("Invalid navigation plan schema or access scope")
    payload = plan["payload"]
    if plan["plan_id"] != _hash(payload):
        raise ValueError("Navigation plan content does not match its plan ID")
    if payload.get("environment_contract") != _contract(base_config):
        raise ValueError("Navigation plan and current environment contract differ")
    expected = _payload(base_config, learning_episodes=payload.get("learning_episodes"),
                        evaluation_episodes=payload.get("evaluation_episodes"),
                        seed=payload.get("seed"))
    if expected != payload:
        raise ValueError("Navigation plan differs from frozen selection or source/assets")


def load_navigation_plan(path: str | Path, base_config: dict[str, Any]) -> dict[str, Any]:
    """Validate a frozen plan against its exact source, assets and selection."""
    source = Path(path).expanduser().resolve()
    with source.open("rb") as stream:
        raw = stream.read(MAX_JSON_BYTES + 1)
    if len(raw) > MAX_JSON_BYTES:
        raise ValueError(f"Navigation plan exceeds {MAX_JSON_BYTES} bytes: {source}")
    plan = json.loads(raw, parse_constant=_reject_constant, object_pairs_hook=_unique_keys)
    _validate_plan(plan, base_config)
    return plan


def episode_lists(plan: dict[str, Any], base_config: dict[str, Any]) -> tuple[list[Any], list[Any]]:
    """Build ordered Habitat episodes from a previously loaded plan, lazily.

    These objects include world-space goals and must stay inside the environment
    service. Use the environment adapter to obtain policy-safe observations.
    """
    _validate_plan(plan, base_config)
    from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1

    groups = []
    for key in ("learning", "evaluation"):
        dataset = PointNavDatasetV1(config=None)
        raw = {"episodes": [record["episode"] for record in plan["payload"][key]]}
        dataset.from_json(canonical_json_bytes(raw).decode("utf-8"),
                          scenes_dir=base_config["dataset"]["scenes_dir"])
        scene_config = base_config["dataset"].get("scene_dataset_config")
        if scene_config:
            for episode in dataset.episodes:
                episode.scene_dataset_config = scene_config
        groups.append(list(dataset.episodes))
    return groups[0], groups[1]
