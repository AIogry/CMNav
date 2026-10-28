"""Explicit settings for the first frozen-graph GCML PointNav adaptation."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from cmnav.configuration.graph_navigation import load_graph_navigation_config
from cmnav.configuration.validation import ConfigError, validate_keys, validate_number


GCML_PROTOCOL = "gcml_pointnav_residual_readout_v1"
METHODS = {"dijkstra", "gcml", "gcml_graph_affordance", "gcml_greedy", "oracle_follower"}
FIELDS = {
    "schema_version", "protocol", "graph_config", "methods", "output_dir",
    "gcml_inverse_learning_rate", "gcml_affordance_learning_rate", "gcml_readout_step_safety",
    "gcml_candidates", "gcml_horizon", "gcml_noise_scale", "gcml_affordance_threshold",
    "gcml_association_radius", "gcml_association_margin", "gcml_cycle_policy",
}


def load_gcml_navigation_config(path: str | Path) -> dict[str, Any]:
    """Reuse the fixed graph/controller/replay settings, with distinct GCML settings."""
    source = Path(path).expanduser().resolve()
    settings = yaml.safe_load(source.read_text(encoding="utf-8"))
    validate_keys(settings, FIELDS, "GCML navigation config")
    if type(settings["schema_version"]) is not int or settings["schema_version"] != 1:
        raise ConfigError("GCML schema_version must be 1")
    if settings["protocol"] != GCML_PROTOCOL:
        raise ConfigError(f"GCML protocol must be {GCML_PROTOCOL}")
    for key in ("graph_config", "output_dir"):
        if not isinstance(settings[key], str) or not settings[key].strip():
            raise ConfigError(f"{key} must be a path string")
    graph_config = (source.parent / settings["graph_config"]).resolve()
    config = load_graph_navigation_config(graph_config)
    if config["cml_sampling"] != "balanced_edge_walks":
        raise ConfigError("GCML v1 requires balanced replay with unique batch indices")
    for key in ("gcml_inverse_learning_rate", "gcml_affordance_learning_rate"):
        validate_number(settings[key], key, 1e-12, 1)
    validate_number(settings["gcml_readout_step_safety"], "gcml_readout_step_safety", 1e-12, 1.99)
    for key in ("gcml_candidates", "gcml_horizon"):
        validate_number(settings[key], key, 1, integer=True)
    validate_number(settings["gcml_noise_scale"], "gcml_noise_scale", 0, 10)
    validate_number(settings["gcml_affordance_threshold"], "gcml_affordance_threshold", 0, 1)
    validate_number(settings["gcml_association_radius"], "gcml_association_radius", 1e-12, 10)
    validate_number(settings["gcml_association_margin"], "gcml_association_margin", 1e-12, 1)
    if settings["gcml_cycle_policy"] != "terminate_on_retrieved_state":
        raise ConfigError("GCML v1 declares candidate-local retrieved-state cycle termination")
    methods = settings["methods"]
    if (not isinstance(methods, list) or not methods
            or any(not isinstance(item, str) or item not in METHODS for item in methods)
            or len(set(methods)) != len(methods)):
        raise ConfigError(f"methods must be a unique nonempty list from {sorted(METHODS)}")
    output = (Path(config["project_root"]) / settings["output_dir"]).resolve()
    if not output.is_relative_to(Path(config["project_root"]) / "outputs"):
        raise ConfigError("GCML outputs must stay inside this checkout's outputs")
    config.update({key: value for key, value in settings.items() if key.startswith("gcml_")})
    config.update(gcml_protocol=GCML_PROTOCOL, graph_config=str(graph_config),
                  gcml_config=str(source), methods=methods, output_dir=str(output))
    return config
