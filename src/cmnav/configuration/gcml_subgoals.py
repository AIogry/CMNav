"""Frozen-checkpoint configuration for teacher-subgoal GCML diagnostics."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from cmnav.configuration.validation import ConfigError, validate_keys, validate_number


PROTOCOL = "gcml_teacher_subgoal_diagnostic_v1"


def load_gcml_subgoal_config(path: str | Path) -> dict[str, Any]:
    """Inherit all model, sampling and simulator settings from a completed run."""
    source = Path(path).expanduser().resolve()
    settings = yaml.safe_load(source.read_text(encoding="utf-8"))
    validate_keys(settings, {"schema_version", "protocol", "project_root", "source_run",
                             "intervals", "methods", "output_dir"}, "GCML subgoal config")
    if type(settings["schema_version"]) is not int or settings["schema_version"] != 1:
        raise ConfigError("GCML subgoal schema_version must be 1")
    if settings["protocol"] != PROTOCOL:
        raise ConfigError(f"GCML subgoal protocol must be {PROTOCOL}")
    for key in ("project_root", "source_run", "output_dir"):
        if not isinstance(settings[key], str) or not settings[key].strip():
            raise ConfigError(f"{key} must be a path string")
    root = (source.parent / settings["project_root"]).resolve()
    origin = (root / settings["source_run"]).resolve()
    output = (root / settings["output_dir"]).resolve()
    if not all(value.is_relative_to(root / "outputs") for value in (origin, output)):
        raise ConfigError("Source and output must remain in this checkout's outputs")
    original = json.loads((origin / "config.json").read_text())
    summary = json.loads((origin / "summary.json").read_text())
    if (summary.get("status") != "completed" or Path(original["project_root"]).resolve() != root
            or original.get("gcml_protocol") != "gcml_pointnav_residual_readout_v1"):
        raise ConfigError("Source must be a completed GCML v1 run in this checkout")
    intervals = settings["intervals"]
    if not isinstance(intervals, list) or not intervals:
        raise ConfigError("intervals must be a nonempty list")
    for value in intervals:
        validate_number(value, "subgoal interval", 2, integer=True)
    if len(set(intervals)) != len(intervals):
        raise ConfigError("subgoal intervals must be unique")
    methods = settings["methods"]
    if (not isinstance(methods, list) or not methods
            or any(item not in ("gcml", "gcml_graph_affordance") for item in methods)
            or len(set(methods)) != len(methods)):
        raise ConfigError("methods must be a unique list of GCML imagination modes")
    if any(method not in summary["methods"] for method in methods):
        raise ConfigError("Source must contain final-goal results for each selected mode")
    original.update(source_run=str(origin), output_dir=str(output), methods=methods,
                    subgoal_protocol=PROTOCOL, subgoal_intervals=intervals,
                    subgoal_config=str(source), rerun_final_goal=True)
    return original
