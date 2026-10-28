"""Artifact identities and tensor storage shared by GCML training and evaluation.

Reading a checkpoint never selects its inference device. Workflow protocols
explicitly reconstruct models on CPU or CUDA and retain the original schema.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any
import torch
from cmnav.runtime.artifacts import sha256_file, write_json

TRAINING_INPUT_NAMES = (
    "config", "graph", "training_walks", "algorithm_reference", "model",
    "training", "summary", "plan", "supported_tasks", "reference_source",
    "source_hashes", "runtime",
)
EVALUATION_INPUT_NAMES = (
    "config", "graph", "model", "training", "summary", "plan", "supported_tasks",
    "algorithm_reference", "reference_source", "source_hashes", "runtime",
)
SHARED_TRAINING_OUTPUT_NAMES = (
    "graph", "training_walks", "algorithm_reference", "plan", "supported_tasks", "reference_source",
)

def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))

def fingerprint_json_inputs(source: Path, names: tuple[str, ...]) -> dict[str, str]:
    return {f"{name}.json": sha256_file(source / f"{name}.json") for name in names}

def training_input_hashes(source: Path) -> dict[str, str]:
    return fingerprint_json_inputs(source, TRAINING_INPUT_NAMES)

def evaluation_input_hashes(source: Path) -> dict[str, str]:
    return fingerprint_json_inputs(source, EVALUATION_INPUT_NAMES)

def tensor_hash(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def parameter_hash(model: Any) -> str:
    digest = hashlib.sha256()
    for name in ("Q", "V", "W", "G"):
        digest.update(name.encode())
        digest.update(getattr(model, name).detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_checkpoint(path: Path) -> dict[str, Any]:
    """Read local tensor storage on CPU, then reconstruct inference on CUDA."""
    if path.suffix.lower() == ".pt":
        state = torch.load(path, map_location="cpu", weights_only=True)
    elif path.suffix.lower() == ".json":
        state = load_json(path)
    else:
        raise ValueError("Checkpoint must be a local .pt or historical .json file")
    if not isinstance(state, dict):
        raise ValueError("Checkpoint must contain a GCML state dictionary")
    return state


def tensor_state(model):
    state = model.state_dict()
    for name in ("Q", "V", "W", "G"):
        state[name] = getattr(model, name).detach().cpu().clone()
    return state


def publish_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_json(temporary, value)
    temporary.replace(path)

