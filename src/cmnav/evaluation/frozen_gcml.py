"""Restore and audit frozen GCML models without choosing navigation policy."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import torch
from cmnav.data.navigation_plans import load_navigation_plan
from cmnav.data.supported_navigation_tasks import load_supported_navigation_tasks
from cmnav.methods.gcml import GenerativeCognitiveMapLearner
from cmnav.runtime.gcml_artifacts import load_json, parameter_hash

TRAINING_COUNTERS = ("updates", "trajectory_updates", "epochs_completed")

def training_counters(model) -> dict[str, int]:
    return {name: getattr(model, name) for name in TRAINING_COUNTERS}

def restore_model(state, graph, device, *, expected_hash=None, require_cuda=False):
    model = GenerativeCognitiveMapLearner.from_state_dict(state, device=device)
    if model.n_states != len(graph.positions) or list(model.edges) != graph.edges:
        raise ValueError("Model and graph vocabularies differ")
    if require_cuda:
        require_cuda_model(model, torch.device(device))
    if expected_hash is not None and parameter_hash(model) != expected_hash:
        raise RuntimeError("Task model did not restore the frozen parameters")
    return model

def assert_frozen(model, expected_hash, expected_counters, *, cuda_device=None):
    if cuda_device is not None:
        require_cuda_model(model, torch.device(cuda_device))
    observed_hash = parameter_hash(model)
    counters = training_counters(model)
    if observed_hash != expected_hash or counters != expected_counters:
        raise RuntimeError("Inference changed frozen parameters or training counters")
    return observed_hash, counters

def load_frozen_tasks(source: Path, config: dict[str, Any], graph_sha256: str):
    """Validate original sensor/task contract before renderer overrides."""
    base = config["base"]
    plan = load_navigation_plan(source / "plan.json", base)
    raw = load_json(source / "supported_tasks.json")
    graph_path = Path(raw["payload"]["graph_path"])
    tasks = load_supported_navigation_tasks(source / "supported_tasks.json", base, plan, graph_path)
    if graph_sha256 != tasks["payload"]["graph_sha256"]:
        raise ValueError("Checkpoint graph differs from the frozen task graph")
    if tasks["payload"]["attachment_radius"] != config["attachment_radius"]:
        raise ValueError("Task and runtime attachment radii differ")
    return plan, tasks, graph_path

def require_cuda_model(model: GenerativeCognitiveMapLearner, device: torch.device) -> None:
    if model.device != device or any(getattr(model, name).device != device for name in ("Q", "V", "W", "G")):
        raise RuntimeError(f"Model inference must stay on {device}; CPU fallback is forbidden")

