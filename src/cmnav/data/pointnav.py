"""Bounded official PointNav dataset parsing and pose validation.

These helpers read task definitions without importing Habitat or starting an
environment. World-space poses remain privileged setup and audit information.
"""
from __future__ import annotations

import gzip
import json
import math
from pathlib import Path
from typing import Any


SCENES_PATH_PREFIX = "data/scene_datasets/"
DEFAULT_CONTENT_SCENES_PATH = "{data_path}/content/{scene}.json.gz"
MAX_JSON_BYTES = 128 * 1024 * 1024


class EpisodeDatasetTooLarge(ValueError):
    """The decompressed source exceeds the caller's explicit read budget."""


class EpisodeDatasetInvalid(ValueError):
    """The decoded source does not contain an episode list."""


def read_episode_dataset(path: Path, *, max_bytes: int | None = None) -> dict[str, Any]:
    """Read a bounded gzip source and require the official episode-list shape."""
    if max_bytes is None:
        max_bytes = MAX_JSON_BYTES
    with gzip.open(path, "rb") as stream:
        contents = stream.read(max_bytes + 1)
    if len(contents) > max_bytes:
        raise EpisodeDatasetTooLarge(f"Episode JSON exceeds {max_bytes} bytes: {path}")
    value = json.loads(contents)
    if not isinstance(value, dict) or not isinstance(value.get("episodes"), list):
        raise EpisodeDatasetInvalid(f"Invalid episode dataset: {path}")
    return value


def validate_pose(value: Any, length: int, name: str) -> list[float]:
    """Validate finite positions or normalized xyzw quaternions unchanged."""
    if (not isinstance(value, list) or len(value) != length
            or any(type(item) not in (int, float) or not math.isfinite(item)
                   for item in value)):
        raise ValueError(f"Invalid {name}: expected {length} finite numbers")
    result = [float(item) for item in value]
    if length == 4 and abs(sum(item * item for item in result) - 1) > 1e-3:
        raise ValueError("Start quaternion must be normalized, in xyzw order")
    return result

