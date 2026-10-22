"""Exercise logging and cleanup without conflating test doubles with Habitat."""
import json
from pathlib import Path

import numpy as np
import pytest

from cmnav.config import load_config
from cmnav.runner import run_experiment


class FakeEnvironment:
    def __init__(self, config):
        self.closed = False
        self.episode_over = False
        self.actions = []
        self.step_number = 0

    def reset(self):
        self.episode_over = False
        self.step_number = 0
        return self.observation()

    def observation(self):
        return {"rgb": np.full((4, 4, 3), self.step_number, dtype=np.uint8),
                "depth": np.full((4, 4, 1), 2.5, dtype=np.float32),
                "pointgoal": np.array([1.0, 0.0, -2.0])}

    def step(self, action):
        assert not self.episode_over
        self.actions.append(action)
        self.step_number += 1
        self.episode_over = action == "STOP"
        return self.observation()

    def episode_info(self):
        return {"episode_id": "fixture", "scene_id": "fixture-only"}

    def metrics(self):
        return {"success": 0.0, "spl": 0.0, "distance_to_goal": np.float64(3.0),
                "unreachable_example": float("inf")}

    def resolved_config(self):
        return {"test_double": True}

    def close(self):
        self.closed = True


@pytest.fixture
def config(tmp_path):
    source = Path(__file__).parents[1] / "configs" / "mp3d_smoke.yaml"
    return load_config(source, overrides={"run.output_dir": str(tmp_path),
                                          "environment.max_episode_steps": 3})


def test_episode_alignment_stop_and_metric_depth(config):
    env = FakeEnvironment(config)
    config["run"]["save_every"] = 2
    directory = run_experiment(config, env_factory=lambda _: env)
    summary = json.loads((directory / "summary.json").read_text())
    rows = [json.loads(line) for line in (directory / "episode_0000/steps.jsonl").read_text().splitlines()]
    assert env.closed
    assert env.actions[-1] == "STOP"
    assert len(env.actions) == 3
    assert [row["observation_step"] for row in rows] == [0, 1, 2, 3]
    assert rows[0]["action_from_previous_observation"] is None
    assert rows[1]["frames"] == {}
    assert rows[-1]["action_from_previous_observation"] == "STOP"
    assert rows[-1]["episode_over"]
    assert rows[-1]["frames"]  # terminal frame survives save_every sampling
    assert rows[0]["evaluation_metrics"]["unreachable_example"] is None
    depth = np.load(directory / "episode_0000" / rows[-1]["frames"]["depth_metres"])
    np.testing.assert_array_equal(depth, np.full((4, 4), 2.5, dtype=np.float32))
    assert summary["execution_status"] == "completed"
    assert summary["result_scope"] == "environment validation only"
    assert summary["episodes"][0]["termination_reason"] == "stop"
    assert summary["aggregate"]["mean_success"] == 0.0  # execution != navigation success
    assert (directory / "episode_0000/preview.gif").is_file()


def test_step_failure_preserves_error_and_closes_environment(config):
    class FailingEnvironment(FakeEnvironment):
        def step(self, action):
            raise RuntimeError("step failed")

        def close(self):
            super().close()
            raise RuntimeError("cleanup failed too")

    env = FailingEnvironment(config)
    with pytest.raises(RuntimeError, match="step failed"):
        run_experiment(config, env_factory=lambda _: env)
    summary_file = next(Path(config["run"]["output_dir"]).glob("*/summary.json"))
    summary = json.loads(summary_file.read_text())
    assert env.closed
    assert summary["execution_status"] == "failed"
    assert "step failed" in summary["error"]
    assert "cleanup failed too" in summary["cleanup_error"]
    assert (summary_file.parent / "error.txt").is_file()


def test_constructor_failure_is_recorded(config):
    def fail(_):
        raise RuntimeError("renderer unavailable")

    with pytest.raises(RuntimeError, match="renderer unavailable"):
        run_experiment(config, env_factory=fail)
    summary_file = next(Path(config["run"]["output_dir"]).glob("*/summary.json"))
    assert json.loads(summary_file.read_text())["execution_status"] == "failed"


def test_seeded_runs_record_same_actions_without_overwriting(config):
    config["environment"]["max_episode_steps"] = 20
    first, second = FakeEnvironment(config), FakeEnvironment(config)
    first_path = run_experiment(config, env_factory=lambda _: first)
    second_path = run_experiment(config, env_factory=lambda _: second)
    assert first.actions == second.actions
    assert first_path != second_path
    assert (first_path / "summary.json").exists()
    assert (second_path / "summary.json").exists()
