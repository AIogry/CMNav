"""File integrity, path boundaries, and explicit JSON artifact serialization."""
from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from pathlib import Path
from typing import Any

def sha256_file(path: Path) -> str:
    """Hash file contents in bounded chunks without loading the entire file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_within(path: Path, root: Path) -> Path:
    """Resolve a path and reject escapes from the explicitly supplied root."""
    path = path.resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Asset path escapes {root}: {path}")
    return path


def json_safe(value: Any) -> Any:
    """Convert recorded arrays and non-finite diagnostics to JSON-safe values."""
    import numpy as np

    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value: Any) -> None:
    """Write a normalized diagnostic artifact, retaining insertion order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")


def write_json_line(stream: Any, value: Any) -> None:
    """Append and flush one normalized diagnostic JSON record."""
    stream.write(json.dumps(json_safe(value), allow_nan=False) + "\n")
    stream.flush()


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize strict sorted-key JSON for frozen manifests and their hashes."""
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def atomic_write_json(path: Path, value: Any) -> None:
    """Atomically replace a strict manifest after flushing it to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(canonical_json_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
