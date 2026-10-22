"""Scene-memory snapshots and task-local working copies, independent of Habitat.

This module manages state; it does not implement CML, localization, or online
learning. Shared neural-network parameters and task-local localization state
belong outside this store. A snapshot may contain scene-specific learned arrays.

State is a string-keyed dictionary of nested dictionaries, lists, JSON scalars,
and numeric/bool NumPy arrays. Serialization uses JSON plus NPZ with
``allow_pickle=False``. Reads and updates copy their data, including arrays.

The first experiment uses ``map_mode="online", retention="reset"``: load a fixed
snapshot, update a working copy within the task, and discard it on finish.
``map_mode="frozen"`` rejects map updates. Future cross-task retention uses
``retention="persistent"`` and an explicit ``finish(commit=True)``; the returned
snapshot ID must then be supplied to the next task. No implicit latest pointer
or shared mutable map exists.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


CONTEXT_FIELDS = (
    "scene_id", "method_id", "goal_protocol", "data_version",
    "model_version", "protocol_version",
)
_SNAPSHOT_ID = re.compile(r"[0-9a-f]{32}\Z")


class MemoryStoreError(ValueError):
    """Invalid state, incompatible context, or corrupt snapshot."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_bytes(value: Any) -> bytes:
    try:
        return (json.dumps(value, ensure_ascii=False, allow_nan=False,
                           sort_keys=True, indent=2) + "\n").encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise MemoryStoreError(f"Value is not finite JSON data: {exc}") from exc


def _metadata(value: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise MemoryStoreError("metadata must be a string-keyed dictionary")
    for key in CONTEXT_FIELDS:
        if not isinstance(value.get(key), str) or not value[key].strip():
            raise MemoryStoreError(f"metadata.{key} must be a nonempty string")
    if value["goal_protocol"] not in ("A", "B", "C"):
        raise MemoryStoreError("metadata.goal_protocol must be A, B, or C")
    return json.loads(_json_bytes(value))


def _check_context(actual: dict[str, Any], expected: dict[str, Any] | None) -> None:
    if expected is None:
        return
    expected = _metadata(expected)
    for key, value in expected.items():
        if key not in actual or actual[key] != value:
            raise MemoryStoreError(f"Snapshot context does not match expected {key!r}")


def _pack(state: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if not isinstance(state, dict):
        raise MemoryStoreError("Scene memory state must be a dictionary")
    arrays: dict[str, np.ndarray] = {}
    ancestors: set[int] = set()

    def encode(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            if value.dtype.kind not in "biufc" or value.dtype.hasobject:
                raise MemoryStoreError("Only numeric/bool NumPy arrays are supported")
            key = f"array_{len(arrays):08d}"
            arrays[key] = np.array(value, copy=True)
            return {"kind": "array", "key": key}
        if value is None or type(value) in (str, bool, int):
            return value
        if type(value) is float:
            if not math.isfinite(value):
                raise MemoryStoreError("JSON scalar floats must be finite")
            return value
        if type(value) not in (dict, list):
            raise MemoryStoreError(f"Unsupported memory value: {type(value).__name__}")
        identity = id(value)
        if identity in ancestors:
            raise MemoryStoreError("Memory state cannot contain reference cycles")
        ancestors.add(identity)
        try:
            if isinstance(value, dict):
                if any(not isinstance(key, str) for key in value):
                    raise MemoryStoreError("Memory dictionary keys must be strings")
                return {"kind": "dict", "items": {key: encode(item)
                                                    for key, item in value.items()}}
            return {"kind": "list", "items": [encode(item) for item in value]}
        finally:
            ancestors.remove(identity)

    return encode(state), arrays


def _unpack(tree: Any, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    used: set[str] = set()

    def decode(value: Any) -> Any:
        if value is None or type(value) in (str, bool, int):
            return value
        if type(value) is float and math.isfinite(value):
            return value
        if not isinstance(value, dict):
            raise MemoryStoreError("Malformed memory node")
        if value.get("kind") == "array" and set(value) == {"kind", "key"}:
            key = value["key"]
            if not isinstance(key, str) or key not in arrays:
                raise MemoryStoreError("Snapshot references a missing array")
            array = arrays[key]
            if array.dtype.kind not in "biufc" or array.dtype.hasobject:
                raise MemoryStoreError("Snapshot contains an unsupported array dtype")
            used.add(key)
            return np.array(array, copy=True)
        if value.get("kind") == "dict" and set(value) == {"kind", "items"}:
            if not isinstance(value["items"], dict):
                raise MemoryStoreError("Malformed dictionary node")
            return {key: decode(item) for key, item in value["items"].items()}
        if value.get("kind") == "list" and set(value) == {"kind", "items"}:
            if not isinstance(value["items"], list):
                raise MemoryStoreError("Malformed list node")
            return [decode(item) for item in value["items"]]
        raise MemoryStoreError("Unknown memory node")

    state = decode(tree)
    if not isinstance(state, dict) or used != set(arrays):
        raise MemoryStoreError("Snapshot root or array references are inconsistent")
    return state


def _copy_state(state: dict[str, Any]) -> dict[str, Any]:
    tree, arrays = _pack(state)
    return _unpack(tree, arrays)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    data = _json_bytes(value)
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _replace_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        _write_json(temporary, value)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class SceneMemoryStore:
    """Immutable snapshots and separate, durable task lifecycle records.

    ``metadata`` requires all ``CONTEXT_FIELDS``. Extra JSON provenance fields
    are retained. ``begin_task`` requires an expected context containing every
    required field, so changing scene, method, model, data, or A/B/C protocol
    cannot silently reuse an incompatible snapshot.

    Snapshot files are immutable through this API, with checksums for corruption
    detection. They are local research artifacts, not an authentication or
    access-control boundary. Tasks have UUID-scoped logs, even for reused task
    names. A process crash leaves an active log; it never commits automatically.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.snapshots = self.root / "snapshots"
        self.sessions = self.root / "sessions"
        self.snapshots.mkdir(parents=True, exist_ok=True)
        self.sessions.mkdir(parents=True, exist_ok=True)

    def _path(self, snapshot_id: str) -> Path:
        if not isinstance(snapshot_id, str) or not _SNAPSHOT_ID.fullmatch(snapshot_id):
            raise MemoryStoreError("snapshot_id must be 32 lowercase hexadecimal characters")
        path = self.snapshots / snapshot_id
        if path.is_symlink():
            raise MemoryStoreError("Snapshot directories cannot be symbolic links")
        return path

    def create_snapshot(self, state: dict[str, Any], metadata: dict[str, Any]) -> str:
        """Publish a new snapshot; existing snapshots are never overwritten."""
        return self._create_snapshot(state, metadata)

    def _create_snapshot(self, state: dict[str, Any], metadata: dict[str, Any], *,
                         parent_snapshot_id: str | None = None,
                         session_id: str | None = None) -> str:
        tree, arrays = _pack(state)
        context = _metadata(metadata)
        snapshot_id = uuid.uuid4().hex
        target = self._path(snapshot_id)
        staging = Path(tempfile.mkdtemp(prefix=".snapshot-", dir=self.snapshots))
        try:
            _write_json(staging / "state.json", tree)
            with (staging / "arrays.npz").open("xb") as stream:
                np.savez_compressed(stream, **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            manifest = {
                "schema_version": 1, "snapshot_id": snapshot_id,
                "created_at": _now(), "metadata": context,
                "parent_snapshot_id": parent_snapshot_id,
                "session_id": session_id,
                "files": {name: _sha256(staging / name)
                          for name in ("state.json", "arrays.npz")},
            }
            _write_json(staging / "manifest.json", manifest)
            if target.exists():
                raise MemoryStoreError("Snapshot ID collision; refusing to overwrite")
            staging.rename(target)
            return snapshot_id
        finally:
            if staging.exists():
                shutil.rmtree(staging)

    def _read_snapshot(self, snapshot_id: str,
                       expected_metadata: dict[str, Any] | None
                       ) -> tuple[dict[str, Any], dict[str, Any]]:
        directory = self._path(snapshot_id)
        try:
            for name in ("manifest.json", "state.json", "arrays.npz"):
                if (directory / name).is_symlink():
                    raise MemoryStoreError("Snapshot files cannot be symbolic links")
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            if (not isinstance(manifest, dict) or manifest.get("schema_version") != 1
                    or manifest.get("snapshot_id") != snapshot_id):
                raise MemoryStoreError("Snapshot identity or schema is invalid")
            context = _metadata(manifest.get("metadata"))
            _check_context(context, expected_metadata)
            checksums = manifest.get("files")
            if not isinstance(checksums, dict) or set(checksums) != {"state.json", "arrays.npz"}:
                raise MemoryStoreError("Snapshot file manifest is invalid")
            for name, expected_hash in checksums.items():
                if _sha256(directory / name) != expected_hash:
                    raise MemoryStoreError(f"Snapshot checksum mismatch: {name}")
            tree = json.loads((directory / "state.json").read_text(encoding="utf-8"))
            with np.load(directory / "arrays.npz", allow_pickle=False) as archive:
                if len(archive.files) != len(set(archive.files)):
                    raise MemoryStoreError("Duplicate array names in snapshot")
                arrays = {key: archive[key] for key in archive.files}
            return _unpack(tree, arrays), context
        except MemoryStoreError:
            raise
        except (OSError, ValueError, TypeError, KeyError, EOFError, zipfile.BadZipFile) as exc:
            raise MemoryStoreError(f"Cannot load snapshot {snapshot_id}: {exc}") from exc

    def load_snapshot(self, snapshot_id: str, *,
                      expected_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        """Read a detached state; optional context checking is for offline inspection.

        Task execution must use ``begin_task``, which makes context checking
        mandatory. Supplying expected metadata here also requires all fields.
        """
        state, _ = self._read_snapshot(snapshot_id, expected_metadata)
        return state

    def begin_task(self, snapshot_id: str, task_id: str, *,
                   expected_metadata: dict[str, Any], map_mode: str = "online",
                   retention: str = "reset") -> MemorySession:
        """Load and check a base snapshot, then start an isolated working copy."""
        if map_mode not in ("online", "frozen"):
            raise MemoryStoreError("map_mode must be online or frozen")
        if retention not in ("reset", "persistent"):
            raise MemoryStoreError("retention must be reset or persistent")
        if not isinstance(task_id, str) or not task_id.strip():
            raise MemoryStoreError("task_id must be a nonempty string")
        context = _metadata(expected_metadata)  # None is never allowed for tasks.
        state, actual_metadata = self._read_snapshot(snapshot_id, context)
        return MemorySession(self, snapshot_id, task_id, state, actual_metadata,
                             map_mode=map_mode, retention=retention)


class MemorySession:
    """Task-local scene-memory working copy, created by ``begin_task``.

    ``finish()`` discards updates by default. Only
    ``retention="persistent"`` plus ``finish(commit=True)`` publishes a new
    snapshot. ``abort()`` always discards. Access after closure raises
    ``RuntimeError``. No method changes the base snapshot.
    """

    def __init__(self, store: SceneMemoryStore, snapshot_id: str, task_id: str,
                 state: dict[str, Any], metadata: dict[str, Any], *,
                 map_mode: str, retention: str) -> None:
        self._store = store
        self._state = _copy_state(state)
        self._metadata = _metadata(metadata)
        self._base_snapshot_id = snapshot_id
        self._map_mode = map_mode
        self._retention = retention
        self._closed = False
        self._pending_commit: str | None = None
        self._session_id = uuid.uuid4().hex
        directory = store.sessions / self._session_id
        directory.mkdir(exist_ok=False)
        self._log_path = directory / "lifecycle.json"
        self._record = {
            "schema_version": 1, "session_id": self._session_id,
            "task_id": task_id, "base_snapshot_id": snapshot_id,
            "map_mode": map_mode, "retention": retention,
            "metadata": self._metadata, "started_at": _now(),
            "status": "active", "updates": 0,
            "final_committed_snapshot_id": None,
        }
        _write_json(self._log_path, self._record)

    @property
    def log_path(self) -> Path:
        """Lifecycle record remains inspectable after the session closes."""
        return self._log_path

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("Memory session is closed")
        if self._pending_commit is not None:
            raise RuntimeError("A commit is pending; retry finish(commit=True) or abort")

    def read(self) -> dict[str, Any]:
        """Return a detached copy; mutating it never changes this session."""
        self._require_open()
        return _copy_state(self._state)

    def update(self, state: dict[str, Any]) -> None:
        """Replace task map state with a validated copy; frozen maps reject this."""
        self._require_open()
        if self._map_mode == "frozen":
            raise PermissionError("Frozen Map forbids scene-memory updates")
        candidate = _copy_state(state)
        record = dict(self._record, updates=self._record["updates"] + 1,
                      last_update_at=_now())
        _replace_json(self._log_path, record)
        self._state = candidate
        self._record = record

    def finish(self, *, commit: bool = False) -> dict[str, Any]:
        """Close normally, discarding unless persistent retention explicitly commits."""
        if self._closed:
            raise RuntimeError("Memory session is closed")
        if type(commit) is not bool:
            raise MemoryStoreError("commit must be a boolean")
        if commit and self._retention != "persistent":
            raise MemoryStoreError("Only persistent retention permits commit=True")
        if self._pending_commit is not None and not commit:
            raise RuntimeError("A commit is pending; retry finish(commit=True) or abort")
        if commit and self._pending_commit is None:
            self._pending_commit = self._store._create_snapshot(
                self._state, self._metadata, parent_snapshot_id=self._base_snapshot_id,
                session_id=self._session_id,
            )
        record = dict(self._record, status="committed" if commit else "discarded",
                      finished_at=_now(), final_committed_snapshot_id=self._pending_commit)
        _replace_json(self._log_path, record)
        self._record = record
        self._closed = True
        self._state = {}
        return json.loads(_json_bytes(record))

    def abort(self, reason: str = "aborted") -> dict[str, Any]:
        """Discard task state; an interrupted publication is logged as an orphan."""
        if self._closed:
            raise RuntimeError("Memory session is closed")
        if not isinstance(reason, str):
            raise MemoryStoreError("abort reason must be a string")
        record = dict(self._record, status="aborted", finished_at=_now(), reason=reason,
                      final_committed_snapshot_id=None)
        if self._pending_commit is not None:
            record["orphan_snapshot_id"] = self._pending_commit
        _replace_json(self._log_path, record)
        self._record = record
        self._closed = True
        self._state = {}
        return json.loads(_json_bytes(record))
