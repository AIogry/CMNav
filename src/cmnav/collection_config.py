"""Configuration for goal-independent exploration, separate from smoke runs."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from cmnav.config import ConfigError, _keys, _number, load_config


def data_path(root: Path, value: str | Path) -> Path:
    """Resolve collection artifacts inside the project's data directory."""
    path = (root / Path(value).expanduser()).resolve()
    try:
        path.relative_to(root / "data")
    except ValueError as exc:
        raise ConfigError(f"Collection data must remain under {root / 'data'}: {path}") from exc
    return path


def load_collection_config(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    try:
        config = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"Cannot read collection config {source}: {exc}") from exc
    _keys(config, {"schema_version", "protocol", "purpose", "base_config", "seed",
                   "collection", "policy"}, "collection config")
    if type(config["schema_version"]) is not int or config["schema_version"] != 1:
        raise ConfigError("Collection schema_version must be 1")
    if config["protocol"] != "exploration_v1" or config["purpose"] != "development":
        raise ConfigError("This collector implements exploration_v1 development data only")
    if not isinstance(config["base_config"], str) or not config["base_config"].strip():
        raise ConfigError("base_config must be a nonempty path")
    _number(config["seed"], "seed", 0, 2**32 - 1, integer=True)
    collection = config["collection"]
    _keys(collection, {"trajectories_per_scene", "steps_per_trajectory", "output_dir",
                       "plan_dir"}, "collection")
    for key in ("trajectories_per_scene", "steps_per_trajectory"):
        _number(collection[key], key, 1, integer=True)
    for key in ("output_dir", "plan_dir"):
        if not isinstance(collection[key], str) or not collection[key].strip():
            raise ConfigError(f"collection.{key} must be a nonempty path")
    policy = config["policy"]
    _keys(policy, {"name", "obstacle_distance", "depth_quantile", "min_valid_fraction",
                   "min_forward_steps", "max_forward_steps", "min_turn_steps",
                   "max_turn_steps"}, "collection policy")
    if policy["name"] != "depth_walk_v1":
        raise ConfigError("Only policy.name=depth_walk_v1 is implemented")
    _number(policy["obstacle_distance"], "obstacle_distance", 0.01)
    _number(policy["depth_quantile"], "depth_quantile", 0, 1)
    _number(policy["min_valid_fraction"], "min_valid_fraction", 0.01, 1)
    for key in ("min_forward_steps", "max_forward_steps", "min_turn_steps", "max_turn_steps"):
        _number(policy[key], key, 1, integer=True)
    for label in ("forward", "turn"):
        if policy[f"min_{label}_steps"] > policy[f"max_{label}_steps"]:
            raise ConfigError(f"min_{label}_steps exceeds max_{label}_steps")
    base_path = (source.parent / config["base_config"]).resolve()
    base = load_config(base_path)
    if base["environment"]["rgbd"]["normalize_depth"]:
        raise ConfigError("Exploration requires unnormalized metric depth")
    if base["environment"]["rgbd"]["min_depth"] != 0:
        raise ConfigError("depth_walk_v1 requires min_depth=0 to preserve unknown depth")
    if policy["obstacle_distance"] >= base["environment"]["rgbd"]["max_depth"]:
        raise ConfigError("obstacle_distance must be below the depth sensor's max_depth")
    if base["dataset"]["split"] != "train":
        raise ConfigError("Development collection uses train starts; never val/test task starts")
    if any(not scene or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
                            for c in scene) for scene in base["dataset"]["scene_ids"]):
        raise ConfigError("Collection requires explicit, bare scene IDs (no wildcard)")
    root = Path(base["project_root"])
    base["seed"] = config["seed"]
    base["environment"]["max_episode_steps"] = collection["steps_per_trajectory"]
    for key in ("output_dir", "plan_dir"):
        collection[key] = str(data_path(root, collection[key]))
    config["base_config"] = str(base_path)
    config["environment_config"] = base
    config["project_root"] = str(root)
    return config
