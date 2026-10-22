"""Version 1, goal-free RGB-D trajectories and verified offline access.

Each directory contains metadata.json, observations.jsonl, transitions.jsonl,
and frames/{index:06d}.{rgb.png,depth.npy}. T transitions link T+1 observations.
Depth is float32 in metres; PNG preserves RGB/RGBA uint8 values. manifest.json
is the final, atomic completion marker and covers every learning-data file.
Diagnostic simulator truth belongs outside the trajectory directory.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any, Iterator

import numpy as np
from PIL import Image


SCHEMA_VERSION = 1
ACTIONS = frozenset({"MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP"})
_PRIVATE_FIELDS = frozenset({
    "pointgoal", "pointgoal_with_gps_compass", "goal_position", "goals",
    "start_position", "start_rotation", "agent_state", "ground_truth", "truth",
    "oracle", "metrics", "evaluation_metrics", "gps", "compass",
    "collision", "collisions", "distance_to_goal", "shortest_paths",
})


def _json_copy(value: Any) -> Any:
    """Require ordinary, finite JSON values; do not silently coerce arrays."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Trajectory JSON cannot contain NaN or infinity")
        return value
    if isinstance(value, list):
        return [_json_copy(item) for item in value]
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("Trajectory JSON object keys must be strings")
        return {key: _json_copy(item) for key, item in value.items()}
    raise ValueError(f"Unsupported trajectory JSON value: {type(value).__name__}")


def _metadata_copy(metadata: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        raise ValueError("Trajectory metadata must be an object")
    result = _json_copy(metadata)
    for key in ("trajectory_id", "scene_id"):
        if not isinstance(result.get(key), str) or not result[key].strip():
            raise ValueError(f"Metadata requires a nonempty {key}")

    def check_fields(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key.lower() in _PRIVATE_FIELDS:
                    raise ValueError(f"Private field {key!r} belongs in a separate audit")
                check_fields(item)
        elif isinstance(value, list):
            for item in value:
                check_fields(item)

    check_fields(result)
    return result


def _atomic_json(path: Path, value: Any) -> None:
    payload = json.dumps(_json_copy(value), ensure_ascii=False, indent=2,
                         allow_nan=False) + "\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8",
                                         dir=path.parent, prefix=".pending-",
                                         delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _fingerprint(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return {"size_bytes": size, "sha256": digest.hexdigest()}


def _arrays(observation: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(observation, dict) or set(observation) != {"rgb", "depth"}:
        raise ValueError("Learning observations must contain exactly rgb and depth")
    rgb, depth = np.asarray(observation["rgb"]), np.asarray(observation["depth"])
    if rgb.ndim != 3 or rgb.shape[2] not in (3, 4) or rgb.dtype != np.uint8:
        raise ValueError("RGB must be uint8 HxWx3 or HxWx4")
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[:, :, 0]
    if (depth.ndim != 2 or depth.shape != rgb.shape[:2]
            or depth.dtype != np.float32 or min(depth.shape) < 1):
        raise ValueError("Depth must be float32 HxW or HxWx1, aligned with RGB")
    if not np.isfinite(depth).all() or (depth < 0).any():
        raise ValueError("Depth must contain finite, nonnegative metres")
    return rgb, depth


def _boundary(action: str, terminated: bool, truncated: bool,
              end_reason: str | None) -> None:
    if not isinstance(action, str) or action not in ACTIONS:
        raise ValueError(f"Unknown trajectory action: {action!r}")
    if type(terminated) is not bool or type(truncated) is not bool:
        raise ValueError("Boundary flags must be booleans")
    if terminated and truncated:
        raise ValueError("Choose termination or truncation, not both")
    if action == "STOP" and not terminated:
        raise ValueError("STOP must be an explicit termination")
    if terminated or truncated:
        if not isinstance(end_reason, str) or not end_reason.strip():
            raise ValueError("A trajectory boundary requires an end reason")
    elif end_reason is not None:
        raise ValueError("An end reason requires a trajectory boundary")


def _frame_paths(index: int) -> dict[str, str]:
    return {"rgb": f"frames/{index:06d}.rgb.png",
            "depth": f"frames/{index:06d}.depth.npy"}


class TrajectoryWriter:
    """Write one reset-delimited trajectory; finish only after a boundary."""

    def __init__(self, directory: Path, metadata: dict[str, Any]):
        self.directory = Path(directory)
        self._metadata = _metadata_copy(metadata)
        reserved = {"schema_version", "status", "observation_count", "transition_count",
                    "frame_shape", "rgb_channels", "depth_unit", "end_reason",
                    "terminated", "truncated", "abort_reason"}
        if reserved.intersection(self._metadata):
            raise ValueError("Caller metadata contains reserved trajectory fields")
        self.directory.mkdir(parents=True, exist_ok=False)
        (self.directory / "frames").mkdir()
        self._observations = 0
        self._transitions = 0
        self._shape: tuple[int, ...] | None = None
        self._last: dict[str, Any] | None = None
        self._status = "collecting"
        self._files = ["observations.jsonl", "transitions.jsonl"]
        self._observation_stream = (self.directory / "observations.jsonl").open("w", encoding="utf-8")
        self._transition_stream = (self.directory / "transitions.jsonl").open("w", encoding="utf-8")
        _atomic_json(self.directory / "metadata.json", self.metadata)

    @property
    def metadata(self) -> dict[str, Any]:
        result = dict(self._metadata, schema_version=SCHEMA_VERSION,
                      status=self._status, observation_count=self._observations,
                      transition_count=self._transitions, depth_unit="metres")
        if self._shape is not None:
            result.update(frame_shape=list(self._shape[:2]), rgb_channels=self._shape[2])
        if self._last is not None:
            result.update({key: self._last[key]
                           for key in ("terminated", "truncated", "end_reason")})
        return _json_copy(result)

    @property
    def summary(self) -> dict[str, Any]:
        return {"status": self._status, "observation_count": self._observations,
                "transition_count": self._transitions}

    def _require_collecting(self) -> None:
        if self._status != "collecting":
            raise RuntimeError(f"Trajectory is {self._status}")

    def _record_observation(self, observation: dict[str, Any]) -> None:
        rgb, depth = _arrays(observation)
        if self._shape is not None and rgb.shape != self._shape:
            raise ValueError("RGB-D shape must remain constant within a trajectory")
        paths = _frame_paths(self._observations)
        Image.fromarray(rgb).save(self.directory / paths["rgb"])
        np.save(self.directory / paths["depth"], depth, allow_pickle=False)
        self._observation_stream.write(json.dumps({"index": self._observations,
                                                  **paths}) + "\n")
        self._observation_stream.flush()
        self._files.extend(paths.values())
        self._shape = rgb.shape
        self._observations += 1

    def start(self, observation: dict[str, Any]) -> None:
        self._require_collecting()
        if self._observations:
            raise RuntimeError("start() is only valid once; use a new writer after reset")
        self._record_observation(observation)

    def append(self, action: str, observation: dict[str, Any], *,
               terminated: bool = False, truncated: bool = False,
               end_reason: str | None = None) -> None:
        self._require_collecting()
        if not self._observations:
            raise RuntimeError("Call start() before appending transitions")
        if self._last and (self._last["terminated"] or self._last["truncated"]):
            raise RuntimeError("Cannot append across a trajectory boundary")
        _boundary(action, terminated, truncated, end_reason)
        record = {"index": self._transitions, "observation_index": self._transitions,
                  "next_observation_index": self._transitions + 1, "action": action,
                  "terminated": terminated, "truncated": truncated, "end_reason": end_reason}
        self._record_observation(observation)
        self._transition_stream.write(json.dumps(record, allow_nan=False) + "\n")
        self._transition_stream.flush()
        self._last = record
        self._transitions += 1

    def _close(self) -> None:
        for stream in (self._observation_stream, self._transition_stream):
            if not stream.closed:
                stream.flush()
                os.fsync(stream.fileno())
                stream.close()

    def finish(self) -> None:
        self._require_collecting()
        if self._last is None or not (self._last["terminated"] or self._last["truncated"]):
            raise RuntimeError("finish() requires a final terminated or truncated transition")
        if self._observations != self._transitions + 1:
            raise RuntimeError("Trajectory alignment is incomplete; abort the writer")
        self._close()
        self._status = "complete"
        try:
            _atomic_json(self.directory / "metadata.json", self.metadata)
            files = {name: _fingerprint(self.directory / name)
                     for name in ["metadata.json", *self._files]}
            _atomic_json(self.directory / "manifest.json",
                         {"schema_version": SCHEMA_VERSION, "status": "complete", "files": files})
        except Exception:
            self._status = "collecting"
            raise

    def abort(self, reason: str) -> None:
        if self._status == "complete":
            raise RuntimeError("A committed trajectory cannot be aborted")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("An abort reason is required")
        self._close()
        self._status = "aborted"
        _atomic_json(self.directory / "metadata.json", dict(self.metadata, abort_reason=reason))


def _read_json(path: Path) -> Any:
    return _json_copy(json.loads(path.read_text(encoding="utf-8")))


def _safe_path(directory: Path, name: str) -> Path:
    if not isinstance(name, str) or "\\" in name:
        raise ValueError("Invalid trajectory file path")
    relative = PurePosixPath(name)
    if relative.is_absolute() or relative.as_posix() != name or ".." in relative.parts:
        raise ValueError("Trajectory paths must be canonical, relative paths")
    path = directory.joinpath(*relative.parts)
    for component in (path, *path.parents):
        if component == directory:
            break
        if component.is_symlink():
            raise ValueError("Symlinks are not allowed inside a trajectory")
    if not path.is_file() or directory not in path.resolve().parents:
        raise ValueError(f"Missing or unsafe trajectory file: {name}")
    return path


def _records(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            record = _json_copy(json.loads(line))
            if not isinstance(record, dict):
                raise ValueError("Trajectory records must be JSON objects")
            records.append(record)
    return records


class TrajectoryReader:
    """Read complete trajectories only, verifying all checksums once at open.

    Dataset files must remain immutable while the reader is open. Arrays are
    loaded on demand and each returned observation owns independent storage.
    windows(length, stride) yields lists containing `length` transitions.
    """

    def __init__(self, directory: Path):
        self.directory = Path(directory).resolve()
        try:
            manifest = _read_json(_safe_path(self.directory, "manifest.json"))
        except (OSError, ValueError) as error:
            raise ValueError("Trajectory is partial or has no readable completion manifest") from error
        if (not isinstance(manifest, dict) or set(manifest) != {"schema_version", "status", "files"}
                or type(manifest["schema_version"]) is not int
                or manifest["schema_version"] != SCHEMA_VERSION or manifest["status"] != "complete"
                or not isinstance(manifest["files"], dict)):
            raise ValueError("Unsupported or incomplete trajectory manifest")
        files = manifest["files"]
        if not {"metadata.json", "observations.jsonl", "transitions.jsonl"}.issubset(files):
            raise ValueError("Manifest is missing required trajectory files")
        for name, expected in files.items():
            if (not isinstance(expected, dict) or set(expected) != {"size_bytes", "sha256"}
                    or type(expected["size_bytes"]) is not int or expected["size_bytes"] < 0
                    or not isinstance(expected["sha256"], str)):
                raise ValueError(f"Invalid file fingerprint for {name}")
            if _fingerprint(_safe_path(self.directory, name)) != expected:
                raise ValueError(f"Trajectory checksum mismatch: {name}")
        self._metadata = _metadata_copy(_read_json(self.directory / "metadata.json"))
        if (type(self._metadata.get("schema_version")) is not int
                or self._metadata["schema_version"] != SCHEMA_VERSION
                or self._metadata.get("status") != "complete"
                or self._metadata.get("depth_unit") != "metres"):
            raise ValueError("Unsupported or incomplete trajectory metadata")
        self._observations = _records(self.directory / "observations.jsonl")
        self._transitions = _records(self.directory / "transitions.jsonl")
        count = len(self._transitions)
        if (count < 1 or len(self._observations) != count + 1
                or type(self._metadata.get("transition_count")) is not int
                or self._metadata["transition_count"] != count
                or type(self._metadata.get("observation_count")) is not int
                or self._metadata["observation_count"] != count + 1):
            raise ValueError("Trajectory observation/transition counts do not align")
        shape = self._metadata.get("frame_shape")
        channels = self._metadata.get("rgb_channels")
        if (not isinstance(shape, list) or len(shape) != 2
                or any(type(size) is not int or size < 1 for size in shape)
                or type(channels) is not int or channels not in (3, 4)):
            raise ValueError("Invalid trajectory frame dimensions")
        self._shape = (*shape, channels)
        expected_files = {"metadata.json", "observations.jsonl", "transitions.jsonl"}
        for index, record in enumerate(self._observations):
            paths = _frame_paths(index)
            if (set(record) != {"index", "rgb", "depth"}
                    or type(record["index"]) is not int or record != {"index": index, **paths}):
                raise ValueError("Observation records are misaligned or contain unexpected fields")
            expected_files.update(paths.values())
        if set(files) != expected_files:
            raise ValueError("Manifest must cover exactly the declared learning-data files")
        transition_keys = {"index", "observation_index", "next_observation_index", "action",
                           "terminated", "truncated", "end_reason"}
        for index, record in enumerate(self._transitions):
            if set(record) != transition_keys:
                raise ValueError("Unexpected transition fields")
            for key, value in (("index", index), ("observation_index", index),
                               ("next_observation_index", index + 1)):
                if type(record[key]) is not int or record[key] != value:
                    raise ValueError("Transition indices must be consecutive within one trajectory")
            _boundary(record["action"], record["terminated"], record["truncated"], record["end_reason"])
            if (record["terminated"] or record["truncated"]) != (index == count - 1):
                raise ValueError("Only the final transition may end a trajectory")
        if (any(type(self._metadata.get(key)) is not bool for key in ("terminated", "truncated"))
                or any(self._metadata.get(key) != self._transitions[-1][key]
                       for key in ("terminated", "truncated", "end_reason"))):
            raise ValueError("Metadata and final transition boundaries differ")

    def __len__(self) -> int:
        return len(self._transitions)

    @property
    def metadata(self) -> dict[str, Any]:
        return _json_copy(self._metadata)

    @property
    def summary(self) -> dict[str, Any]:
        return {"status": "complete", "observation_count": len(self._observations),
                "transition_count": len(self)}

    def observation(self, index: int) -> dict[str, np.ndarray]:
        if type(index) is not int or not 0 <= index < len(self._observations):
            raise IndexError("Observation index is outside the trajectory")
        record = self._observations[index]
        with Image.open(_safe_path(self.directory, record["rgb"])) as image:
            rgb = np.array(image)
        depth = np.load(_safe_path(self.directory, record["depth"]), allow_pickle=False)
        rgb, depth = _arrays({"rgb": rgb, "depth": depth})
        if rgb.shape != self._shape:
            raise ValueError("Stored frame dimensions differ from trajectory metadata")
        return {"rgb": rgb.copy(), "depth": depth.copy()}

    def _transition(self, index: int) -> dict[str, Any]:
        record = self._transitions[index]
        return {"trajectory_id": self._metadata["trajectory_id"],
                "collection_id": self._metadata.get("collection_id"),
                "index": index, "observation_index": index, "next_observation_index": index + 1,
                "obs": self.observation(index), "action": record["action"],
                "next_obs": self.observation(index + 1),
                "terminated": record["terminated"], "truncated": record["truncated"],
                "end_reason": record["end_reason"]}

    def transitions(self) -> Iterator[dict[str, Any]]:
        for index in range(len(self)):
            yield self._transition(index)

    def windows(self, length: int, stride: int = 1) -> Iterator[list[dict[str, Any]]]:
        if type(length) is not int or length < 1 or type(stride) is not int or stride < 1:
            raise ValueError("Window length and stride must be positive integers")
        for start in range(0, len(self) - length + 1, stride):
            yield [self._transition(index) for index in range(start, start + length)]
