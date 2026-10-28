"""Trajectory storage contracts, corruption handling, and offline provenance."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from cmnav.data.trajectories import TrajectoryReader, TrajectoryWriter


def observation(index=0, *, channels=3, singleton_depth=False):
    rgb = np.full((3, 4, channels), index, dtype=np.uint8)
    depth = np.full((3, 4), index + 0.25, dtype=np.float32)
    if singleton_depth:
        depth = depth[..., None]
    return {"rgb": rgb, "depth": depth}


def metadata():
    return {"trajectory_id": "trajectory-1", "scene_id": "scene-1",
            "collection_id": "collection-1", "sensor_spec": {"height": 3}}


def write_complete(path, *, count=4, channels=3, singleton_depth=False):
    writer = TrajectoryWriter(path, metadata())
    writer.start(observation(channels=channels, singleton_depth=singleton_depth))
    for index in range(count):
        writer.append("MOVE_FORWARD", observation(index + 1, channels=channels,
                                                 singleton_depth=singleton_depth),
                      truncated=index == count - 1,
                      end_reason="collector_budget" if index == count - 1 else None)
    writer.finish()
    return writer


def refresh_fingerprint(path: Path, name: str):
    """Allow structural validation to be exercised independently of checksums."""
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    payload = (path / name).read_bytes()
    manifest["files"][name] = {"size_bytes": len(payload),
                               "sha256": hashlib.sha256(payload).hexdigest()}
    manifest_path.write_text(json.dumps(manifest))


def rewrite_record(path, filename, index, update):
    records = [json.loads(line) for line in (path / filename).read_text().splitlines()]
    records[index].update(update)
    (path / filename).write_text("".join(json.dumps(record) + "\n" for record in records))
    refresh_fingerprint(path, filename)


@pytest.mark.parametrize("channels,singleton_depth", [(3, False), (4, False), (3, True)])
def test_round_trip_preserves_t_plus_one_and_exact_arrays(tmp_path, channels, singleton_depth):
    path = tmp_path / "trajectory"
    writer = write_complete(path, channels=channels, singleton_depth=singleton_depth)
    reader = TrajectoryReader(path)
    assert writer.summary == reader.summary == {
        "status": "complete", "observation_count": 5, "transition_count": 4}
    assert len(reader) == 4
    assert reader.metadata["depth_unit"] == "metres"
    assert reader.metadata["frame_shape"] == [3, 4]
    assert reader.metadata["rgb_channels"] == channels
    for index in range(5):
        actual = reader.observation(index)
        expected = observation(index, channels=channels)
        np.testing.assert_array_equal(actual["rgb"], expected["rgb"])
        np.testing.assert_array_equal(actual["depth"], expected["depth"])
        assert actual["rgb"].dtype == np.uint8
        assert actual["depth"].dtype == np.float32
    transitions = list(reader.transitions())
    assert [item["observation_index"] for item in transitions] == [0, 1, 2, 3]
    assert [item["next_observation_index"] for item in transitions] == [1, 2, 3, 4]
    assert all(not item["terminated"] for item in transitions)
    assert [item["truncated"] for item in transitions] == [False, False, False, True]
    assert transitions[-1]["end_reason"] == "collector_budget"
    assert transitions[-1]["action"] == "MOVE_FORWARD"


def test_explicit_stop_is_termination_not_budget_truncation(tmp_path):
    writer = TrajectoryWriter(tmp_path / "trajectory", metadata())
    writer.start(observation())
    writer.append("STOP", observation(1), terminated=True, end_reason="agent_stop")
    writer.finish()
    transition = next(TrajectoryReader(writer.directory).transitions())
    assert transition["terminated"] is True
    assert transition["truncated"] is False


@pytest.mark.parametrize("field", ["gps", "compass", "collisions", "agent_state",
                                    "start_position", "start_rotation", "goals",
                                    "pointgoal", "pointgoal_with_gps_compass",
                                    "ground_truth", "distance_to_goal", "METRICS"])
def test_private_metadata_rejected_recursively_before_creating_files(tmp_path, field):
    value = metadata()
    value["nested"] = [{"nested_again": {field: [1, 2, 3]}}]
    with pytest.raises(ValueError, match="Private field"):
        TrajectoryWriter(tmp_path / "trajectory", value)
    assert not (tmp_path / "trajectory").exists()


@pytest.mark.parametrize("update", [{"trajectory_id": " "}, {"scene_id": None},
                                    {"status": "complete"}, {"transition_count": 5},
                                    {"extra": float("nan")}, {"extra": float("inf")},
                                    {"extra": np.array([1])}, {"extra": {1: "bad"}}])
def test_invalid_metadata_rejected(tmp_path, update):
    value = metadata()
    value.update(update)
    with pytest.raises(ValueError):
        TrajectoryWriter(tmp_path / "trajectory", value)


@pytest.mark.parametrize("case", ["extra_truth", "missing_rgb", "float_rgb", "gray_rgb",
                                   "two_channels", "float64_depth", "unaligned_depth",
                                   "negative_depth", "nan_depth", "infinite_depth", "empty"])
def test_invalid_observations_rejected_without_advancing_writer(tmp_path, case):
    value = observation()
    if case == "extra_truth":
        value["gps"] = np.zeros(2)
    elif case == "missing_rgb":
        del value["rgb"]
    elif case == "float_rgb":
        value["rgb"] = value["rgb"].astype(np.float32)
    elif case == "gray_rgb":
        value["rgb"] = value["rgb"][:, :, 0]
    elif case == "two_channels":
        value["rgb"] = value["rgb"][:, :, :2]
    elif case == "float64_depth":
        value["depth"] = value["depth"].astype(np.float64)
    elif case == "unaligned_depth":
        value["depth"] = np.zeros((2, 4), dtype=np.float32)
    elif case == "empty":
        value = {"rgb": np.zeros((0, 4, 3), dtype=np.uint8),
                 "depth": np.zeros((0, 4), dtype=np.float32)}
    else:
        value["depth"][0, 0] = {"negative_depth": -1, "nan_depth": np.nan,
                                "infinite_depth": np.inf}[case]
    writer = TrajectoryWriter(tmp_path / "trajectory", metadata())
    try:
        with pytest.raises(ValueError):
            writer.start(value)
        assert writer.summary["observation_count"] == 0
    finally:
        writer.abort("test cleanup")


@pytest.mark.parametrize("action,boundary", [
    ("TELEPORT", {}), ("STOP", {}),
    ("STOP", {"truncated": True, "end_reason": "budget"}),
    ("MOVE_FORWARD", {"terminated": True, "truncated": True, "end_reason": "both"}),
    ("MOVE_FORWARD", {"truncated": 1, "end_reason": "budget"}),
    ("MOVE_FORWARD", {"truncated": True}),
    ("MOVE_FORWARD", {"terminated": True, "end_reason": " "}),
    ("MOVE_FORWARD", {"end_reason": "not a boundary"}),
])
def test_invalid_boundary_does_not_write_a_next_frame(tmp_path, action, boundary):
    writer = TrajectoryWriter(tmp_path / "trajectory", metadata())
    writer.start(observation())
    try:
        with pytest.raises(ValueError):
            writer.append(action, observation(1), **boundary)
        assert writer.summary["observation_count"] == 1
        assert writer.summary["transition_count"] == 0
    finally:
        writer.abort("test cleanup")


def test_writer_reset_boundaries_and_finish_state(tmp_path):
    writer = TrajectoryWriter(tmp_path / "trajectory", metadata())
    with pytest.raises(RuntimeError, match="start"):
        writer.append("MOVE_FORWARD", observation())
    with pytest.raises(RuntimeError, match="final"):
        writer.finish()
    writer.start(observation())
    with pytest.raises(RuntimeError, match="only valid once"):
        writer.start(observation())
    with pytest.raises(ValueError, match="shape"):
        writer.append("MOVE_FORWARD", observation(1, channels=4))
    writer.append("TURN_LEFT", observation(1))
    with pytest.raises(RuntimeError, match="final"):
        writer.finish()
    writer.append("TURN_RIGHT", observation(2), truncated=True, end_reason="budget")
    with pytest.raises(RuntimeError, match="boundary"):
        writer.append("MOVE_FORWARD", observation(3))
    writer.finish()
    with pytest.raises(RuntimeError, match="complete"):
        writer.append("MOVE_FORWARD", observation(3))
    with pytest.raises(RuntimeError, match="complete"):
        writer.finish()
    with pytest.raises(RuntimeError, match="committed"):
        writer.abort("should not alter complete data")
    assert len(TrajectoryReader(writer.directory)) == 2


def test_incomplete_and_aborted_trajectories_are_not_readable(tmp_path):
    writer = TrajectoryWriter(tmp_path / "trajectory", metadata())
    writer.start(observation())
    writer.append("MOVE_FORWARD", observation(1))
    with pytest.raises(ValueError, match="partial"):
        TrajectoryReader(writer.directory)
    writer.abort("interrupted collection")
    with pytest.raises(ValueError, match="partial"):
        TrajectoryReader(writer.directory)
    assert not (writer.directory / "manifest.json").exists()
    with pytest.raises(RuntimeError, match="aborted"):
        writer.append("MOVE_FORWARD", observation(2))


@pytest.mark.parametrize("name", ["metadata.json", "observations.jsonl", "transitions.jsonl",
                                  "frames/000000.rgb.png", "frames/000001.depth.npy"])
def test_checksum_corruption_rejected_at_open(tmp_path, name):
    path = tmp_path / "trajectory"
    write_complete(path)
    with (path / name).open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        TrajectoryReader(path)


def test_missing_frame_rejected_at_open(tmp_path):
    path = tmp_path / "trajectory"
    write_complete(path)
    (path / "frames/000002.depth.npy").unlink()
    with pytest.raises(ValueError, match="Missing or unsafe"):
        TrajectoryReader(path)


@pytest.mark.parametrize("filename,index,update", [
    ("observations.jsonl", 0, {"index": True}),
    ("observations.jsonl", 0, {"gps": [0, 0]}),
    ("observations.jsonl", 0, {"rgb": "../outside.png"}),
    ("transitions.jsonl", 0, {"observation_index": 1}),
    ("transitions.jsonl", 0, {"index": False}),
    ("transitions.jsonl", 0, {"truncated": True, "end_reason": "early"}),
    ("transitions.jsonl", 3, {"truncated": False, "end_reason": None}),
    ("transitions.jsonl", 3, {"end_reason": "mismatch"}),
    ("transitions.jsonl", 0, {"collision": False}),
])
def test_rehashed_invalid_records_still_rejected(tmp_path, filename, index, update):
    path = tmp_path / "trajectory"
    write_complete(path)
    rewrite_record(path, filename, index, update)
    with pytest.raises(ValueError):
        TrajectoryReader(path)


@pytest.mark.parametrize("update", [{"schema_version": True}, {"observation_count": 4},
                                    {"transition_count": True}, {"depth_unit": "mm"},
                                    {"frame_shape": [True, 4]}, {"rgb_channels": 2},
                                    {"truncated": 1}, {"goals": [1, 2, 3]}])
def test_rehashed_invalid_metadata_still_rejected(tmp_path, update):
    path = tmp_path / "trajectory"
    write_complete(path)
    value = json.loads((path / "metadata.json").read_text())
    value.update(update)
    (path / "metadata.json").write_text(json.dumps(value))
    refresh_fingerprint(path, "metadata.json")
    with pytest.raises(ValueError):
        TrajectoryReader(path)


def test_symlinked_learning_file_rejected_even_if_content_matches(tmp_path):
    path = tmp_path / "trajectory"
    write_complete(path)
    source = path / "frames/000000.depth.npy"
    target = tmp_path / "outside.npy"
    source.rename(target)
    source.symlink_to(target)
    with pytest.raises(ValueError, match="Symlinks"):
        TrajectoryReader(path)


def test_rehashed_wrong_frame_dtype_rejected_when_loaded(tmp_path):
    path = tmp_path / "trajectory"
    write_complete(path)
    name = "frames/000000.depth.npy"
    np.save(path / name, np.zeros((3, 4), dtype=np.float64), allow_pickle=False)
    refresh_fingerprint(path, name)
    reader = TrajectoryReader(path)
    with pytest.raises(ValueError, match="float32"):
        reader.observation(0)


def test_windows_preserve_provenance_order_and_do_not_cross_trajectory(tmp_path):
    path = tmp_path / "trajectory"
    write_complete(path, count=5)
    reader = TrajectoryReader(path)
    windows = list(reader.windows(3, stride=2))
    assert [[item["index"] for item in window] for window in windows] == [[0, 1, 2], [2, 3, 4]]
    assert all(item["trajectory_id"] == "trajectory-1" and
               item["collection_id"] == "collection-1"
               for window in windows for item in window)
    assert list(reader.windows(6)) == []
    assert windows[-1][-1]["truncated"] is True
    assert all(not item["truncated"] for item in windows[0])
    for window in windows:
        for item in window:
            assert item["next_observation_index"] == item["observation_index"] + 1
            np.testing.assert_array_equal(item["obs"]["rgb"], observation(item["index"])["rgb"])
    windows[0][-1]["obs"]["depth"].fill(99)
    assert (windows[1][0]["obs"]["depth"] == 2.25).all()


def test_metadata_and_loaded_arrays_are_detached_from_callers(tmp_path):
    path = tmp_path / "trajectory"
    source_metadata = metadata()
    source_observation = observation()
    writer = TrajectoryWriter(path, source_metadata)
    source_metadata["sensor_spec"]["height"] = 999
    writer.start(source_observation)
    source_observation["rgb"].fill(255)
    writer.metadata["sensor_spec"]["height"] = 500
    writer.append("MOVE_FORWARD", observation(1), truncated=True, end_reason="budget")
    writer.finish()
    reader = TrajectoryReader(path)
    reader.metadata["sensor_spec"]["height"] = 100
    assert reader.metadata["sensor_spec"]["height"] == 3
    first = reader.observation(0)
    first["rgb"].fill(200)
    first["depth"].fill(50)
    np.testing.assert_array_equal(reader.observation(0)["rgb"], observation()["rgb"])
    np.testing.assert_array_equal(reader.observation(0)["depth"], observation()["depth"])


@pytest.mark.parametrize("index", [-1, 5, True, 0.0, "0"])
def test_reader_invalid_observation_index(tmp_path, index):
    path = tmp_path / "trajectory"
    write_complete(path)
    with pytest.raises(IndexError):
        TrajectoryReader(path).observation(index)


@pytest.mark.parametrize("length,stride", [(0, 1), (-1, 1), (1, 0), (1, -1),
                                           (True, 1), (1, True), (1.5, 1)])
def test_reader_invalid_window_arguments(tmp_path, length, stride):
    path = tmp_path / "trajectory"
    write_complete(path)
    with pytest.raises(ValueError):
        list(TrajectoryReader(path).windows(length, stride))
