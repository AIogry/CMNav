"""Goal-free collector contracts using tiny assets and a deterministic fake env."""
from copy import deepcopy
import gzip
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from cmnav import collection
from cmnav.collection_config import data_path, load_collection_config
from cmnav.config import ConfigError
from cmnav.exploration_policy import DepthWalkPolicy
from cmnav.trajectories import TrajectoryReader


PROJECT = Path(__file__).resolve().parents[1]


def write_gzip(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump(value, stream)


@pytest.fixture
def setup(tmp_path):
    project = tmp_path / "project"
    configs = project / "configs"
    configs.mkdir(parents=True)
    base = yaml.safe_load((PROJECT / "configs/mp3d_smoke.yaml").read_text())
    base["dataset"]["scene_ids"] = ["sceneB", "sceneA"]
    base["dataset"].pop("scene_dataset_config", None)
    base["environment"]["rgbd"].update(width=8, height=8)
    config = yaml.safe_load((PROJECT / "configs/mp3d_collect.yaml").read_text())
    config["base_config"] = "base.yaml"
    config["collection"].update(trajectories_per_scene=2, steps_per_trajectory=4)
    for scene in base["dataset"]["scene_ids"]:
        folder = project / "data/scene_datasets/mp3d" / scene
        folder.mkdir(parents=True)
        (folder / f"{scene}.glb").write_bytes(b"tiny-mesh")
        (folder / f"{scene}.navmesh").write_bytes(b"tiny-navmesh")
        episodes = [{"episode_id": str(index),
                     "scene_id": f"data/scene_datasets/mp3d/{scene}/{scene}.glb",
                     "start_position": [float(index), 0.0, 0.0],
                     "start_rotation": [0.0, 0.0, 0.0, 1.0],
                     "goals": [{"position": [99.0, 0.0, 0.0]}]}
                    for index in range(5)]
        write_gzip(project / f"data/datasets/pointnav/mp3d/v1/train/content/{scene}.json.gz",
                   {"episodes": episodes})
    write_gzip(project / "data/datasets/pointnav/mp3d/v1/train/train.json.gz",
               {"episodes": [], "content_scenes_path": "{data_path}/content/{scene}.json.gz"})
    return {"project": project, "base": base, "raw": config,
            "path": configs / "collection.yaml"}


def load(setup):
    setup["path"].write_text(yaml.safe_dump(setup["raw"]))
    (setup["path"].parent / "base.yaml").write_text(yaml.safe_dump(setup["base"]))
    return load_collection_config(setup["path"])


def observation(index=0, depth=2.0):
    return {"rgb": np.full((8, 8, 3), index, dtype=np.uint8),
            "depth": np.full((8, 8), depth, dtype=np.float32)}


class FakeEnvironment:
    def __init__(self, config, starts, *, fail_at=None, ends_at=None,
                 wrong_reset=False, close_failure=False):
        self.config = config
        self.starts = deepcopy(starts)
        self.index = -1
        self.step_index = 0
        self.reset_ids = []
        self.actions = []
        self.fail_at = fail_at
        self.ends_at = ends_at
        self.wrong_reset = wrong_reset
        self.close_failure = close_failure
        self.closed = False

    def reset(self):
        self.index += 1
        self.step_index = 0
        self.reset_ids.append(self.starts[self.index]["trajectory_id"])
        return observation()

    def episode_info(self):
        start = self.starts[self.index]
        return {"episode_id": "incorrect" if self.wrong_reset else start["trajectory_id"],
                "scene_id": start["scene_id"]}

    def step(self, action):
        self.step_index += 1
        if self.fail_at == self.step_index:
            raise RuntimeError("injected simulator failure")
        self.actions.append((self.index, action))
        return observation(self.step_index)

    @property
    def episode_over(self):
        return self.ends_at == self.step_index

    def audit(self):
        return {"position": [123.0, 0.0, float(self.step_index)],
                "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                "collisions": {"is_collision": True, "count": self.step_index}}

    def resolved_config(self):
        return {"simulator": "fake", "max_episode_steps": self.config["environment"]["max_episode_steps"]}

    def close(self):
        self.closed = True
        if self.close_failure:
            raise RuntimeError("injected cleanup failure")


@pytest.fixture
def fake_preflight(monkeypatch):
    monkeypatch.setattr(collection, "inspect_environment", lambda config: {"ready_for_run": True})
    monkeypatch.setattr(collection, "_versions", lambda: {"runtime": "test"})


def test_config_resolves_paths_and_overrides_seed_and_budget(setup, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    config = load(setup)
    assert config["environment_config"]["seed"] == config["seed"]
    assert config["environment_config"]["environment"]["max_episode_steps"] == 4
    assert config["collection"]["output_dir"] == str(setup["project"] / "data/trajectories")


@pytest.mark.parametrize("section,key,value", [
    ("collection", "trajectories_per_scene", 0),
    ("collection", "steps_per_trajectory", True),
    ("collection", "output_dir", "../outside"),
    ("collection", "plan_dir", "outputs"),
    ("policy", "depth_quantile", 1.1),
    ("policy", "min_valid_fraction", 0),
    ("policy", "obstacle_distance", float("nan")),
    ("policy", "min_turn_steps", 4),
    ("policy", "name", "oracle"),
])
def test_config_rejects_invalid_collection_settings(setup, section, key, value):
    setup["raw"][section][key] = value
    with pytest.raises(ConfigError):
        load(setup)


@pytest.mark.parametrize("field,value", [("split", "val"), ("split", "test"),
                                        ("scene_ids", ["*"]), ("scene_ids", ["../sceneA"])])
def test_development_collection_rejects_heldout_starts_and_wildcards(setup, field, value):
    setup["base"]["dataset"][field] = value
    with pytest.raises(ConfigError):
        load(setup)


@pytest.mark.parametrize("field,value", [("normalize_depth", True), ("min_depth", 0.1),
                                        ("max_depth", 0.5)])
def test_config_rejects_depth_units_or_threshold_incompatibility(setup, field, value):
    setup["base"]["environment"]["rgbd"][field] = value
    with pytest.raises(ConfigError):
        load(setup)


def test_data_path_rejects_symlink_escape(setup, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (setup["project"] / "data/link").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ConfigError):
        data_path(setup["project"], "data/link/result")


def test_plan_selection_is_deterministic_pins_assets_and_omits_goal(setup):
    config = load(setup)
    first = json.loads(collection.create_collection_plan(config).read_text())
    second = json.loads(collection.create_collection_plan(config).read_text())
    assert first["plan_id"] != second["plan_id"]
    assert first["payload"] == second["payload"]
    payload = first["payload"]
    assert payload["budget_actions"] == 16
    assert [entry["scene_key"] for entry in payload["starts"]] == ["sceneA", "sceneA", "sceneB", "sceneB"]
    assert len({(entry["scene_key"], entry["source_episode_id"]) for entry in payload["starts"]}) == 4
    assert [entry["policy_seed"] for entry in payload["starts"]] == [7, 8, 9, 10]
    assert "goals" not in json.dumps(payload)
    assert len(payload["fingerprints"]) == 7  # Root, two shards, two meshes, two navmeshes.
    config["seed"] += 1
    different = json.loads(collection.create_collection_plan(config).read_text())
    assert payload["starts"] != different["payload"]["starts"]


@pytest.mark.parametrize("mutation", ["asset", "source", "config", "pose"])
def test_collection_rejects_changed_plan_config_or_source_before_environment(setup, fake_preflight, mutation):
    config = load(setup)
    plan_path = collection.create_collection_plan(config)
    plan = json.loads(plan_path.read_text())
    if mutation == "asset":
        Path(plan["payload"]["starts"][0]["scene_id"]).write_bytes(b"changed")
    elif mutation == "source":
        root = Path(config["environment_config"]["dataset"]["episodes_path"])
        write_gzip(root, {"episodes": [], "extra_provenance": "changed"})
    elif mutation == "config":
        config["collection"]["steps_per_trajectory"] += 1
    else:
        plan["payload"]["starts"][0]["start_position"][0] += 1
        plan_path.write_text(json.dumps(plan))
    def forbidden_factory(*args):
        pytest.fail("Environment should never initialize for a mismatched plan")
    with pytest.raises(ValueError, match="mismatch"):
        collection.collect_trajectories(config, plan_path, env_factory=forbidden_factory)


@pytest.mark.parametrize("mutation", ["quaternion", "nan_pose", "duplicate_id", "insufficient_starts", "scene_mismatch"])
def test_plan_rejects_invalid_source_episodes(setup, mutation):
    config = load(setup)
    shard = setup["project"] / "data/datasets/pointnav/mp3d/v1/train/content/sceneA.json.gz"
    with gzip.open(shard, "rt") as stream:
        data = json.load(stream)
    if mutation == "quaternion":
        for episode in data["episodes"]:
            episode["start_rotation"] = [0, 0, 0, 2]
    elif mutation == "nan_pose":
        for episode in data["episodes"]:
            episode["start_position"] = [float("nan"), 0, 0]
    elif mutation == "duplicate_id":
        data["episodes"][1]["episode_id"] = data["episodes"][0]["episode_id"]
    elif mutation == "insufficient_starts":
        data["episodes"] = data["episodes"][:1]
    else:
        data["episodes"][0]["scene_id"] = "mp3d/sceneB/sceneB.glb"
    write_gzip(shard, data)
    with pytest.raises(ValueError):
        collection.create_collection_plan(config)


def test_policy_uses_reproducible_segments_and_no_stop(setup):
    policy_config = load(setup)["policy"]
    first = DepthWalkPolicy(17, policy_config)
    second = DepthWalkPolicy(17, policy_config)
    actions = [first.act(observation()) for _ in range(100)]
    assert actions == [second.act(observation()) for _ in range(100)]
    assert set(actions) == {"MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"}
    assert actions[0] == "MOVE_FORWARD"


@pytest.mark.parametrize("depth", [0, -1, 0.2, float("nan"), float("inf")])
def test_policy_turns_for_obstacle_or_unknown_depth(setup, depth):
    policy = DepthWalkPolicy(7, load(setup)["policy"])
    assert policy.act(observation(depth=depth)) in {"TURN_LEFT", "TURN_RIGHT"}


def test_policy_rejects_goal_or_diagnostic_input(setup):
    policy = DepthWalkPolicy(7, load(setup)["policy"])
    with pytest.raises(ValueError, match="exactly rgb and depth"):
        policy.act(dict(observation(), pointgoal=[1, 2, 3]))


def test_collection_exact_starts_actions_boundaries_and_audit_separation(setup, fake_preflight):
    config = load(setup)
    plan_path = collection.create_collection_plan(config)
    environments = []
    def factory(*args):
        environments.append(FakeEnvironment(*args))
        return environments[-1]
    directory = collection.collect_trajectories(config, plan_path, env_factory=factory)
    manifest = json.loads((directory / "dataset.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["expected_actions"] == 16
    env = environments[0]
    assert env.closed
    assert env.reset_ids == [f"traj_{index:06d}" for index in range(4)]
    assert len(env.actions) == 16  # Even every collision counts against the budget.
    assert len(manifest["trajectories"]) == 4
    for entry in manifest["trajectories"]:
        path = directory / entry["path"]
        reader = TrajectoryReader(path)
        assert len(reader) == 4
        assert reader.summary["observation_count"] == 5
        assert reader.metadata["trajectory_id"] == entry["trajectory_id"]
        rows = list(reader.transitions())
        assert [row["truncated"] for row in rows] == [False, False, False, True]
        assert not any(row["terminated"] for row in rows)
        assert rows[-1]["end_reason"] == "collector_budget"
        assert all(row["action"] != "STOP" for row in rows)
        for index, row in enumerate(rows):
            assert set(row["obs"]) == set(row["next_obs"]) == {"rgb", "depth"}
            assert np.all(row["obs"]["rgb"] == index)
            assert np.all(row["next_obs"]["rgb"] == index + 1)
        windows = list(reader.windows(3))
        assert [[row["index"] for row in window] for window in windows] == [[0, 1, 2], [1, 2, 3]]
        learning_text = " ".join(file.read_text() for file in path.glob("*.json*"))
        for forbidden in ("start_position", "start_rotation", "collisions", "rotation_xyzw", "goal_position"):
            assert forbidden not in learning_text
        audit = [json.loads(line) for line in (directory / "audit" / f"{entry['trajectory_id']}.jsonl").read_text().splitlines()]
        assert len(audit) == 5
        assert audit[-1]["position"] == [123.0, 0.0, 4.0]
        assert audit[-1]["collisions"]["count"] == 4


@pytest.mark.parametrize("options,message", [
    ({"fail_at": 2}, "injected simulator failure"),
    ({"ends_at": 2}, "ended before external truncation"),
    ({"wrong_reset": True}, "did not follow"),
])
def test_collection_aborts_partial_data_and_closes_environment(setup, fake_preflight, options, message):
    config = load(setup)
    plan_path = collection.create_collection_plan(config)
    environments = []
    def factory(*args):
        environments.append(FakeEnvironment(*args, **options))
        return environments[-1]
    with pytest.raises(RuntimeError, match=message):
        collection.collect_trajectories(config, plan_path, env_factory=factory)
    assert environments[0].closed
    directory, = Path(config["collection"]["output_dir"]).iterdir()
    manifest = json.loads((directory / "dataset.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["trajectories"] == []
    for partial in (directory / "trajectories").glob("*"):
        with pytest.raises(ValueError, match="partial"):
            TrajectoryReader(partial)
        assert json.loads((partial / "metadata.json").read_text())["status"] == "aborted"


def test_collection_preflight_failure_never_constructs_environment(setup, monkeypatch):
    config = load(setup)
    plan_path = collection.create_collection_plan(config)
    monkeypatch.setattr(collection, "inspect_environment", lambda config: {"ready_for_run": False})
    def factory(*args):
        pytest.fail("Preflight failure must precede environment construction")
    with pytest.raises(RuntimeError, match="preflight failed"):
        collection.collect_trajectories(config, plan_path, env_factory=factory)
    assert not Path(config["collection"]["output_dir"]).exists()


def test_cleanup_failure_marks_dataset_failed(setup, fake_preflight):
    config = load(setup)
    plan_path = collection.create_collection_plan(config)
    with pytest.raises(RuntimeError, match="cleanup failure"):
        collection.collect_trajectories(config, plan_path,
                                        env_factory=lambda *args: FakeEnvironment(*args, close_failure=True))
    directory, = Path(config["collection"]["output_dir"]).iterdir()
    manifest = json.loads((directory / "dataset.json").read_text())
    assert manifest["status"] == "failed"
    assert "cleanup_error" in manifest
    assert len(manifest["trajectories"]) == 4
