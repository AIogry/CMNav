"""Memory persistence and experiment-isolation regression checks (CPU only)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from cmnav.memory.snapshots import CONTEXT_FIELDS, MemoryStoreError, SceneMemoryStore


@pytest.fixture
def context():
    return {
        "scene_id": "17DRP5sb8fy", "method_id": "cml-v1", "goal_protocol": "A",
        "data_version": "pilot-v1", "model_version": "frozen-v1",
        "protocol_version": "memory-v1", "provenance": {"seed": 7},
    }


@pytest.fixture
def state():
    return {
        "nodes": [{"name": "first", "feature": np.arange(6, dtype=np.float32).reshape(2, 3)}],
        "edges": np.array([[False, True], [True, False]], dtype=bool),
        "count": 1, "confidence": 0.5, "optional": None,
        "nested": {"kind": "array", "key": "ordinary user strings"},
    }


@pytest.fixture
def store(tmp_path):
    return SceneMemoryStore(tmp_path / "memory")


def test_roundtrip_preserves_structure_dtype_and_detaches_copies(store, state, context):
    snapshot = store.create_snapshot(state, context)
    state["nodes"][0]["feature"][:] = -9
    state["nodes"].append({"name": "unpublished"})
    context["provenance"]["seed"] = 999
    first = store.load_snapshot(snapshot)
    np.testing.assert_array_equal(first["nodes"][0]["feature"], np.arange(6).reshape(2, 3))
    assert first["nodes"][0]["feature"].dtype == np.float32
    assert first["edges"].dtype == bool
    assert first["nested"] == {"kind": "array", "key": "ordinary user strings"}
    assert first["optional"] is None
    first["nodes"][0]["feature"][:] = 100
    first["nodes"].clear()
    second = store.load_snapshot(snapshot)
    assert len(second["nodes"]) == 1
    np.testing.assert_array_equal(second["nodes"][0]["feature"], np.arange(6).reshape(2, 3))
    manifest = json.loads((store.snapshots / snapshot / "manifest.json").read_text())
    assert manifest["metadata"]["provenance"]["seed"] == 7


def test_independent_store_instance_loads_snapshot(store, state, context):
    snapshot = store.create_snapshot(state, context)
    reopened = SceneMemoryStore(store.root)
    loaded = reopened.load_snapshot(snapshot, expected_metadata=context)
    np.testing.assert_array_equal(loaded["edges"], state["edges"])


@pytest.mark.parametrize("field", CONTEXT_FIELDS)
def test_create_rejects_missing_context_field(store, state, context, field):
    del context[field]
    with pytest.raises(MemoryStoreError, match=field):
        store.create_snapshot(state, context)
    assert list(store.snapshots.iterdir()) == []


@pytest.mark.parametrize("field", CONTEXT_FIELDS)
def test_begin_requires_all_expected_context_fields(store, state, context, field):
    snapshot = store.create_snapshot(state, context)
    del context[field]
    with pytest.raises(MemoryStoreError, match=field):
        store.begin_task(snapshot, "task", expected_metadata=context)
    assert list(store.sessions.iterdir()) == []


@pytest.mark.parametrize("field", CONTEXT_FIELDS)
def test_begin_rejects_context_mismatch(store, state, context, field):
    snapshot = store.create_snapshot(state, context)
    context[field] = "B" if field == "goal_protocol" else "different"
    with pytest.raises(MemoryStoreError, match=field):
        store.begin_task(snapshot, "task", expected_metadata=context)
    assert list(store.sessions.iterdir()) == []


@pytest.mark.parametrize("protocol", ["A", "B", "C"])
def test_all_goal_protocols_are_supported_and_isolated(store, state, context, protocol):
    context["goal_protocol"] = protocol
    snapshot = store.create_snapshot(state, context)
    session = store.begin_task(snapshot, "task", expected_metadata=context)
    session.finish()
    for other in set("ABC") - {protocol}:
        with pytest.raises(MemoryStoreError, match="goal_protocol"):
            store.begin_task(snapshot, "task", expected_metadata=dict(context, goal_protocol=other))


def test_begin_rejects_missing_or_none_expected_context(store, state, context):
    snapshot = store.create_snapshot(state, context)
    with pytest.raises(TypeError):
        store.begin_task(snapshot, "task")
    with pytest.raises(MemoryStoreError):
        store.begin_task(snapshot, "task", expected_metadata=None)


def test_online_reset_discards_updates_and_preserves_baseline(store, state, context):
    snapshot = store.create_snapshot(state, context)
    session = store.begin_task(snapshot, "first", expected_metadata=context)
    working = session.read()
    working["count"] = 2
    working["nodes"][0]["feature"][:] = 9
    assert session.read()["count"] == 1  # Read is not a mutation handle.
    session.update(working)
    working["count"] = 3
    working["nodes"][0]["feature"][:] = -1
    assert session.read()["count"] == 2
    np.testing.assert_array_equal(session.read()["nodes"][0]["feature"], np.full((2, 3), 9))
    with pytest.raises(MemoryStoreError, match="persistent"):
        session.finish(commit=True)
    assert session.read()["count"] == 2  # Rejected commit leaves session usable.
    result = session.finish()
    assert result["status"] == "discarded"
    assert result["updates"] == 1
    assert result["final_committed_snapshot_id"] is None
    assert json.loads(session.log_path.read_text()) == result
    assert len(list(store.snapshots.iterdir())) == 1
    next_session = store.begin_task(snapshot, "second", expected_metadata=context)
    assert next_session.read()["count"] == 1
    np.testing.assert_array_equal(next_session.read()["nodes"][0]["feature"], state["nodes"][0]["feature"])
    next_session.finish()


def test_frozen_rejects_updates_and_exposes_only_copies(store, state, context):
    snapshot = store.create_snapshot(state, context)
    session = store.begin_task(snapshot, "frozen", map_mode="frozen", expected_metadata=context)
    working = session.read()
    working["count"] = 999
    working["nodes"][0]["feature"][:] = 999
    with pytest.raises(PermissionError, match="Frozen Map"):
        session.update(working)
    assert session.read()["count"] == 1
    np.testing.assert_array_equal(session.read()["nodes"][0]["feature"], state["nodes"][0]["feature"])
    result = session.finish()
    assert result["updates"] == 0
    assert result["status"] == "discarded"


def test_persistent_requires_explicit_commit_and_records_lineage(store, state, context):
    snapshot = store.create_snapshot(state, context)
    discarded = store.begin_task(snapshot, "no-commit", retention="persistent", expected_metadata=context)
    discarded.update({"count": 12})
    assert discarded.finish()["status"] == "discarded"
    assert len(list(store.snapshots.iterdir())) == 1
    session = store.begin_task(snapshot, "commit", retention="persistent", expected_metadata=context)
    session.update({"count": 2, "weights": np.eye(2, dtype=np.float32)})
    result = session.finish(commit=True)
    published = result["final_committed_snapshot_id"]
    assert result["status"] == "committed"
    assert published != snapshot
    manifest = json.loads((store.snapshots / published / "manifest.json").read_text())
    assert manifest["parent_snapshot_id"] == snapshot
    assert manifest["session_id"] == result["session_id"]
    assert store.load_snapshot(snapshot)["count"] == 1
    next_session = store.begin_task(published, "next", retention="persistent", expected_metadata=context)
    assert next_session.read()["count"] == 2
    np.testing.assert_array_equal(next_session.read()["weights"], np.eye(2))
    next_session.finish()


def test_abort_never_commits_and_logs_reason(store, state, context):
    snapshot = store.create_snapshot(state, context)
    session = store.begin_task(snapshot, "aborted", retention="persistent", expected_metadata=context)
    session.update({"count": 999})
    result = session.abort("collector interrupted")
    assert result["status"] == "aborted"
    assert result["reason"] == "collector interrupted"
    assert result["final_committed_snapshot_id"] is None
    assert len(list(store.snapshots.iterdir())) == 1
    assert store.load_snapshot(snapshot)["count"] == 1
    assert json.loads(session.log_path.read_text()) == result


@pytest.mark.parametrize("close", ["finish", "abort"])
def test_closed_sessions_reject_all_operations(store, state, context, close):
    snapshot = store.create_snapshot(state, context)
    session = store.begin_task(snapshot, "task", expected_metadata=context)
    getattr(session, close)()
    operations = [session.read, lambda: session.update({}), session.finish, session.abort]
    for operation in operations:
        with pytest.raises(RuntimeError, match="closed"):
            operation()
    assert session.log_path.is_file()


@pytest.mark.parametrize("filename", ["state.json", "arrays.npz"])
def test_checksum_detects_modified_snapshot_payload(store, state, context, filename):
    snapshot = store.create_snapshot(state, context)
    path = store.snapshots / snapshot / filename
    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(MemoryStoreError, match="checksum mismatch"):
        store.load_snapshot(snapshot)
    with pytest.raises(MemoryStoreError, match="checksum mismatch"):
        store.begin_task(snapshot, "task", expected_metadata=context)


@pytest.mark.parametrize("invalid", ["../outside", "0" * 31, "G" * 32, "0" * 33, None])
def test_invalid_snapshot_ids_are_rejected(store, invalid):
    with pytest.raises(MemoryStoreError, match="snapshot_id"):
        store.load_snapshot(invalid)


@pytest.mark.parametrize("bad_state", [
    {"object": np.array([object()], dtype=object)},
    {"strings": np.array(["a"], dtype="U1")},
    {"scalar": float("nan")},
    {"tuple": (1, 2)},
    {1: "non-string key"},
])
def test_unsupported_state_does_not_publish_snapshot(store, context, bad_state):
    with pytest.raises(MemoryStoreError):
        store.create_snapshot(bad_state, context)
    assert list(store.snapshots.iterdir()) == []


def test_cyclic_state_is_rejected(store, context):
    state = {}
    state["self"] = state
    with pytest.raises(MemoryStoreError, match="cycles"):
        store.create_snapshot(state, context)


def test_reused_task_names_still_have_independent_lifecycle_logs(store, state, context):
    snapshot = store.create_snapshot(state, context)
    first = store.begin_task(snapshot, "same", expected_metadata=context)
    second = store.begin_task(snapshot, "same", expected_metadata=context)
    assert first.log_path != second.log_path
    first.update({"count": 2})
    assert second.read()["count"] == 1
    first.finish()
    second.finish()
