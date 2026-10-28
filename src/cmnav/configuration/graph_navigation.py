"""Configuration for the privileged, frozen-map PointNav pipeline diagnostic.

This protocol uses oracle state association and local control. It is an
engineering diagnostic, not a visual-navigation result or the formal online-map
experiment. Only the wrapper's seed and action budget override the base config.
Random or edge-balanced walks replay the observed graph for CML training;
they do not collect new physical trajectories or RGB-D frames.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from cmnav.configuration.environment import load_config, validate_config
from cmnav.configuration.validation import ConfigError, validate_keys, validate_number


PROTOCOL = "oracle_state_graph_diagnostic_v1"
CML_TRAINING_PROTOCOL = "observed_graph_random_walk_torch_v1"
CML_BALANCED_TRAINING_PROTOCOL = "observed_graph_balanced_edge_walk_torch_v1"
METHODS = frozenset({"oracle_follower", "dijkstra", "cml", "random_graph"})
_FIELDS = {
    "schema_version", "protocol", "base_config", "seed", "learning_episodes",
    "evaluation_episodes", "max_steps", "merge_radius", "attachment_radius",
    "local_goal_radius", "max_local_steps", "max_graph_hops", "cml_dimension",
    "cml_epochs", "learning_rate", "state_learning_rate", "inverse_learning_rate",
    "methods", "output_dir",
}
_CML_DEFAULTS = {
    "cml_random_walks": 200,
    "cml_walk_length": 32,
    "cml_device": "cpu",
    "cml_sampling": "random_walks",
    "cml_edge_visits": 20,
}


def _path_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ConfigError(f"{name} must be a nonempty path string")
    return value


def load_graph_navigation_config(path: str | Path) -> dict[str, Any]:
    """Load exact diagnostic settings, without importing Habitat or writing files.

    ``base_config`` resolves relative to this YAML file. ``output_dir`` resolves
    relative to the base configuration's project root and must remain within
    that checkout's outputs directory. The returned ``base`` is a fully resolved
    regular CMNav environment configuration; budgets and methods are reusable
    across explicit runs, while navigation manifests freeze selected tasks.

    Older schema-1 input files may omit the random-walk count, node count, and
    explicit device; those receive the documented defaults. Balanced replay
    uses cml_edge_visits rather than cml_random_walks to determine its size.
    Their supplied
    ``cml_epochs`` remains unchanged. ``cml_training_protocol`` is derived from
    the current learner protocol so legacy input files are not mistaken for a
    request to execute the former transition-by-transition training procedure.
    Device syntax is checked here; device availability is checked at execution.
    """
    source = Path(path).expanduser().resolve()
    try:
        config = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"Cannot read graph navigation config {source}: {exc}") from exc
    validate_keys(config, _FIELDS, "graph navigation config", optional=set(_CML_DEFAULTS))
    for key, default in _CML_DEFAULTS.items():
        config.setdefault(key, default)
    if type(config["schema_version"]) is not int or config["schema_version"] != 1:
        raise ConfigError("Graph navigation schema_version must be 1")
    if config["protocol"] != PROTOCOL:
        raise ConfigError(f"Graph navigation protocol must be {PROTOCOL}")
    validate_number(config["seed"], "seed", 0, 2**32 - 1, integer=True)
    for key in ("learning_episodes", "evaluation_episodes", "max_local_steps",
                "max_graph_hops", "cml_dimension", "cml_epochs", "cml_random_walks"):
        validate_number(config[key], key, 1, integer=True)
    validate_number(config["cml_walk_length"], "cml_walk_length", 2, integer=True)
    validate_number(config["cml_edge_visits"], "cml_edge_visits", 1, integer=True)
    if (not isinstance(config["cml_sampling"], str)
            or config["cml_sampling"] not in ("random_walks", "balanced_edge_walks")):
        raise ConfigError("cml_sampling must be 'random_walks' or 'balanced_edge_walks'")
    device = config["cml_device"]
    if (not isinstance(device, str)
            or (device != "cpu" and re.fullmatch(r"cuda:[0-9]+", device) is None)):
        raise ConfigError("cml_device must be 'cpu' or 'cuda:N' with a nonnegative integer N")
    validate_number(config["max_steps"], "max_steps", 2, integer=True)
    for key in ("merge_radius", "attachment_radius"):
        validate_number(config[key], key, 0.001, 1.0)
    validate_number(config["local_goal_radius"], "local_goal_radius", 0.001, 0.2)
    if config["merge_radius"] > config["attachment_radius"]:
        raise ConfigError("merge_radius cannot exceed attachment_radius")
    for key in ("learning_rate", "state_learning_rate", "inverse_learning_rate"):
        validate_number(config[key], key, 0, 1)
        if config[key] <= 0:
            raise ConfigError(f"{key} must be positive")
    methods = config["methods"]
    if (not isinstance(methods, list) or not methods
            or any(not isinstance(method, str) or method not in METHODS for method in methods)
            or len(set(methods)) != len(methods)):
        raise ConfigError(f"methods must be a nonempty, unique list from {sorted(METHODS)}")

    base_value = Path(_path_string(config["base_config"], "base_config")).expanduser()
    base_path = (source.parent / base_value).resolve()
    base = load_config(base_path)
    dataset = base["dataset"]
    if dataset["name"] != "mp3d" or dataset["split"] != "train":
        raise ConfigError("This diagnostic requires official MP3D training tasks")
    if len(dataset["scene_ids"]) != 1 or dataset["scene_ids"] == ["*"]:
        raise ConfigError("This diagnostic requires exactly one explicit MP3D scene")
    scene = dataset["scene_ids"][0]
    if any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
           for character in scene):
        raise ConfigError("dataset.scene_ids must contain a bare MP3D scene identifier")
    base["seed"] = config["seed"]
    base["environment"]["max_episode_steps"] = config["max_steps"]
    validate_config(base)
    root = Path(base["project_root"])
    output = (root / Path(_path_string(config["output_dir"], "output_dir")).expanduser()).resolve()
    if not output.is_relative_to(root / "outputs"):
        raise ConfigError(f"Graph navigation outputs must remain under {root / 'outputs'}")
    config["base_config"] = str(base_path)
    config["base"] = base
    config["project_root"] = str(root)
    config["output_dir"] = str(output)
    config["cml_training_protocol"] = (
        CML_BALANCED_TRAINING_PROTOCOL if config["cml_sampling"] == "balanced_edge_walks"
        else CML_TRAINING_PROTOCOL
    )
    return config
