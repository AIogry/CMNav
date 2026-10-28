"""Small, validated experiment configuration independent of Habitat/Hydra."""
from __future__ import annotations

import copy
import string
from pathlib import Path
from typing import Any

import yaml

from cmnav.configuration import EXPECTED_HABITAT_VERSION
from cmnav.configuration.validation import ConfigError, validate_keys, validate_number


def validate_config(config: dict[str, Any]) -> None:
    validate_keys(config, {"schema_version", "project_root", "seed", "protocol", "dataset",
                           "environment", "policy", "run"}, "config")
    if type(config["schema_version"]) is not int or config["schema_version"] != 1:
        raise ConfigError("schema_version must be 1")
    if config["protocol"] != "environment_smoke":
        raise ConfigError("Only protocol=environment_smoke is implemented in milestone 1")
    if not isinstance(config["project_root"], str) or not config["project_root"]:
        raise ConfigError("project_root must be a nonempty path")
    validate_number(config["seed"], "seed", 0, 2**32 - 1, integer=True)
    dataset = config["dataset"]
    validate_keys(dataset, {"name", "split", "episodes_path", "scenes_dir", "scene_ids"}, "dataset",
                  optional={"scene_dataset_config"})
    if dataset["name"] not in ("mp3d", "habitat_test"):
        raise ConfigError("dataset.name must be mp3d or habitat_test")
    for key in ("split", "episodes_path", "scenes_dir"):
        if not isinstance(dataset[key], str) or not dataset[key].strip():
            raise ConfigError(f"dataset.{key} must be a nonempty string")
    if "scene_dataset_config" in dataset and (
        not isinstance(dataset["scene_dataset_config"], str)
        or not dataset["scene_dataset_config"].strip()
    ):
        raise ConfigError("dataset.scene_dataset_config must be a nonempty path when provided")
    scenes = dataset["scene_ids"]
    if not isinstance(scenes, list) or not scenes or any(
        not isinstance(scene, str) or not scene.strip() for scene in scenes
    ):
        raise ConfigError("dataset.scene_ids must be a nonempty list of strings")
    if len(set(scenes)) != len(scenes) or ("*" in scenes and len(scenes) != 1):
        raise ConfigError("scene_ids must be unique; '*' cannot be mixed with other scenes")
    env = config["environment"]
    validate_keys(env, {"max_episode_steps", "forward_step_size", "turn_angle", "gpu_device_id",
                        "goal_sensor", "goal_dimensions", "rgbd"}, "environment")
    validate_number(env["max_episode_steps"], "max_episode_steps", 2, integer=True)
    validate_number(env["forward_step_size"], "forward_step_size", 0.001)
    validate_number(env["turn_angle"], "turn_angle", 1, 180, integer=True)
    validate_number(env["gpu_device_id"], "gpu_device_id", 0, integer=True)
    if env["goal_sensor"] != "fixed" or type(env["goal_dimensions"]) is not int or env["goal_dimensions"] != 3:
        raise ConfigError("Milestone 1 requires a fixed, 3D Cartesian PointGoal")
    sensor = env["rgbd"]
    validate_keys(sensor, {"width", "height", "hfov", "sensor_height", "min_depth",
                           "max_depth", "normalize_depth"}, "environment.rgbd")
    for key in ("width", "height"):
        validate_number(sensor[key], key, 1, 4096, integer=True)
    validate_number(sensor["hfov"], "hfov", 1, 179, integer=True)
    validate_number(sensor["sensor_height"], "sensor_height", 0)
    validate_number(sensor["min_depth"], "min_depth", 0)
    validate_number(sensor["max_depth"], "max_depth", 0.001)
    if sensor["max_depth"] <= sensor["min_depth"]:
        raise ConfigError("max_depth must exceed min_depth")
    if sensor["normalize_depth"] is not False:
        raise ConfigError("normalize_depth must be false so saved depth is in metres")
    policy = config["policy"]
    validate_keys(policy, {"name", "forward_probability"}, "policy")
    if policy["name"] != "random":
        raise ConfigError("Only the random environment-check policy is implemented")
    validate_number(policy["forward_probability"], "forward_probability", 0, 1)
    run = config["run"]
    validate_keys(run, {"episodes", "output_dir", "save_every", "preview_fps"}, "run")
    for key in ("episodes", "save_every", "preview_fps"):
        validate_number(run[key], f"run.{key}", 1, integer=True)
    if not isinstance(run["output_dir"], str) or not run["output_dir"]:
        raise ConfigError("run.output_dir must be a nonempty path")


def load_config(path: str | Path, *, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    try:
        config = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"Cannot read config {source}: {exc}") from exc
    if overrides:
        config = copy.deepcopy(config)
        if not isinstance(config, dict):
            raise ConfigError("config must be a mapping")
        for key, value in overrides.items():
            if value is None:
                continue
            parts = key.split(".")
            target = config
            for part in parts[:-1]:
                if not isinstance(target.get(part), dict):
                    raise ConfigError(f"Cannot override {key}")
                target = target[part]
            if parts[-1] not in target:
                raise ConfigError(f"Unknown override {key}")
            target[parts[-1]] = value
    validate_config(config)
    root = (source.parent / Path(config["project_root"]).expanduser()).resolve()
    config["project_root"] = str(root)
    paths = [("dataset", "episodes_path"), ("dataset", "scenes_dir"), ("run", "output_dir")]
    if "scene_dataset_config" in config["dataset"]:
        paths.append(("dataset", "scene_dataset_config"))
    for section, key in paths:
        value = config[section][key]
        if key == "episodes_path":
            try:
                for _, field, spec, conversion in string.Formatter().parse(value):
                    if field is not None and (field != "split" or spec or conversion):
                        raise ValueError("unsupported format field")
                value = value.format(split=config["dataset"]["split"])
            except (KeyError, ValueError, IndexError) as exc:
                raise ConfigError("episodes_path may only use the {split} placeholder") from exc
        config[section][key] = str((root / Path(value).expanduser()).resolve())
    return config
