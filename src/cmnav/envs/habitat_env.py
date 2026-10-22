"""Habitat-Lab 0.3.3 adapter for CMNav's RGB-D PointNav protocol.

Only RGB, depth, and an episode-start goal vector enter observations. Habitat's
ground-truth measurements are available separately for experiment recording.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


class HabitatEnvironment:
    """Run PointNav without exposing simulator state to a policy.

    ``config`` is CMNav's validated configuration, with absolute dataset paths.
    Habitat and Habitat-Sim are deliberately imported only on construction so
    that configuration tools remain usable before installing the simulator.
    """

    _ACTIONS = {
        "STOP": "stop",
        "MOVE_FORWARD": "move_forward",
        "TURN_LEFT": "turn_left",
        "TURN_RIGHT": "turn_right",
    }

    def __init__(
        self,
        config: dict[str, Any],
        *,
        episodes: list[Any] | None = None,
        exploration: bool = False,
    ) -> None:
        if episodes is not None and not episodes:
            raise ValueError("Explicit episodes must be nonempty.")
        if exploration and episodes is None:
            raise ValueError("Exploration requires an explicit ordered start list.")
        try:
            import habitat
            import habitat_sim  # noqa: F401 -- fail early if only Lab is installed
            from habitat.config import read_write
            from habitat.config.default_structured_configs import (
                CollisionsMeasurementConfig,
                DistanceToGoalMeasurementConfig,
                HabitatSimDepthSensorConfig,
                HabitatSimRGBSensorConfig,
                MoveForwardActionConfig,
                PointGoalSensorConfig,
                SoftSPLMeasurementConfig,
                SPLMeasurementConfig,
                StopActionConfig,
                SuccessMeasurementConfig,
                TurnLeftActionConfig,
                TurnRightActionConfig,
            )
            from omegaconf import OmegaConf
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                f"Habitat runtime dependency {exc.name!r} is unavailable. "
                "Use the CMNav environment with Python 3.9, Habitat-Sim 0.3.3 "
                "headless and Habitat-Lab 0.3.3; see README.md for installation."
            ) from exc

        dataset = config["dataset"]
        environment = config["environment"]
        rgbd = environment["rgbd"]
        if not exploration and environment["goal_sensor"] != "fixed":
            raise ValueError("This adapter requires environment.goal_sensor='fixed'.")
        if not exploration and environment["goal_dimensions"] != 3:
            raise ValueError("This adapter requires a three-dimensional fixed goal.")
        for name, value in (
            ("turn_angle", environment["turn_angle"]),
            ("rgbd.hfov", rgbd["hfov"]),
        ):
            if not float(value).is_integer():
                raise ValueError(f"Habitat 0.3.3 requires an integer {name} in degrees.")

        benchmarks = {
            "mp3d": "benchmark/nav/pointnav/pointnav_mp3d.yaml",
            "habitat_test": "benchmark/nav/pointnav/pointnav_habitat_test.yaml",
        }
        if dataset["name"] not in benchmarks:
            raise ValueError(f"Unsupported PointNav dataset {dataset['name']!r}.")
        habitat_config = habitat.get_config(benchmarks[dataset["name"]])
        with read_write(habitat_config):
            cfg = habitat_config.habitat
            cfg.seed = config["seed"]
            cfg.dataset.split = dataset["split"]
            cfg.dataset.data_path = dataset["episodes_path"]
            cfg.dataset.scenes_dir = dataset["scenes_dir"]
            cfg.dataset.content_scenes = list(dataset["scene_ids"])
            if "scene_dataset_config" in dataset:
                # The official MP3D config specifies mesh orientation, flat
                # shading, and the matching navigation/semantic resources.
                cfg.simulator.scene_dataset = dataset["scene_dataset_config"]
            # Collection ends at its external action budget. Keep Habitat's
            # safety limit beyond that boundary so it cannot turn truncation
            # into a task termination on the final collected transition.
            cfg.environment.max_episode_steps = (
                environment["max_episode_steps"] + int(exploration)
            )
            # Deterministic iteration makes smoke runs and later replays comparable.
            cfg.environment.iterator_options.shuffle = False
            cfg.environment.iterator_options.cycle = episodes is None
            if episodes is not None:
                # A manifest's order is authoritative, including scene order.
                cfg.environment.iterator_options.group_by_scene = False
                cfg.environment.iterator_options.num_episode_sample = -1
                cfg.environment.iterator_options.max_scene_repeat_episodes = -1
                cfg.environment.iterator_options.max_scene_repeat_steps = -1
            if exploration:
                cfg.environment.max_episode_seconds = 0
            cfg.simulator.forward_step_size = environment["forward_step_size"]
            cfg.simulator.turn_angle = int(environment["turn_angle"])
            cfg.simulator.habitat_sim_v0.gpu_device_id = environment["gpu_device_id"]
            cfg.simulator.habitat_sim_v0.gpu_gpu = False
            cfg.simulator.habitat_sim_v0.enable_physics = False

            camera = {
                "width": rgbd["width"],
                "height": rgbd["height"],
                "hfov": int(rgbd["hfov"]),
                "position": [0.0, rgbd["sensor_height"], 0.0],
            }
            cfg.simulator.agents.main_agent.sim_sensors = {
                "rgb_sensor": HabitatSimRGBSensorConfig(**camera),
                "depth_sensor": HabitatSimDepthSensorConfig(
                    **camera,
                    min_depth=rgbd["min_depth"],
                    max_depth=rgbd["max_depth"],
                    normalize_depth=rgbd["normalize_depth"],
                ),
            }
            cfg.task.actions = {
                "stop": StopActionConfig(),
                "move_forward": MoveForwardActionConfig(),
                "turn_left": TurnLeftActionConfig(),
                "turn_right": TurnRightActionConfig(),
            }
            # Replace the default dynamic GPS+compass goal completely. In
            # Habitat's 3D Cartesian convention x is right, y up, and -z forward.
            cfg.task.lab_sensors = {
                "pointgoal_sensor": PointGoalSensorConfig(
                    goal_format="CARTESIAN", dimensionality=3
                ),
            }
            cfg.task.goal_sensor_uuid = "pointgoal"
            cfg.task.measurements = {
                "distance_to_goal": DistanceToGoalMeasurementConfig(),
                "success": SuccessMeasurementConfig(success_distance=0.2),
                "spl": SPLMeasurementConfig(),
                "soft_spl": SoftSPLMeasurementConfig(),
                "collisions": CollisionsMeasurementConfig(),
            }
            cfg.task.success_measure = "success"
            cfg.task.reward_measure = None
            if exploration:
                cfg.task.actions = {
                    "move_forward": MoveForwardActionConfig(),
                    "turn_left": TurnLeftActionConfig(),
                    "turn_right": TurnRightActionConfig(),
                }
                cfg.task.lab_sensors = {}
                cfg.task.goal_sensor_uuid = ""
                cfg.task.measurements = {
                    "collisions": CollisionsMeasurementConfig(),
                }
                cfg.task.success_measure = ""

        # Snapshot before Env substitutes individual episode scene/start values.
        self._resolved_config = OmegaConf.to_container(habitat_config, resolve=True)
        loaded_dataset = None
        if episodes is not None:
            # PointNavDatasetV1(config=None) creates an empty dataset. Supplying
            # the manifest directly avoids reloading the full episode archive.
            loaded_dataset = habitat.make_dataset(
                id_dataset=habitat_config.habitat.dataset.type,
                config=None,
            )
            loaded_dataset.episodes = list(episodes)
        elif "scene_dataset_config" in dataset:
            loaded_dataset = habitat.make_dataset(
                id_dataset=habitat_config.habitat.dataset.type,
                config=habitat_config.habitat.dataset,
            )
        if loaded_dataset is not None and "scene_dataset_config" in dataset:
            # Lab 0.3.3 replaces simulator.scene_dataset with the first
            # episode's value at initialization. Keep both sources aligned.
            for episode in loaded_dataset.episodes:
                episode.scene_dataset_config = dataset["scene_dataset_config"]
        self._env = habitat.Env(config=habitat_config, dataset=loaded_dataset)
        self._env.seed(config["seed"])
        self._closed = False
        self._steps = 0
        self._exploration = exploration

    def reset(self) -> dict[str, Any]:
        """Start the next dataset episode and return permitted observations."""
        self._require_open()
        observation = dict(self._env.reset())
        self._steps = 0
        return self._permitted_observations(observation)

    def step(self, action: str) -> dict[str, Any]:
        """Apply a CMNav action; STOP counts as an environment step."""
        self._require_open()
        if self._exploration and action == "STOP":
            raise ValueError("Goal-free exploration does not expose STOP.")
        if action not in self._ACTIONS:
            raise ValueError(f"Unknown action {action!r}; expected {tuple(self._ACTIONS)}.")
        observation = dict(self._env.step(self._ACTIONS[action]))
        self._steps += 1
        return self._permitted_observations(observation)

    def _permitted_observations(self, observation: dict[str, Any]) -> dict[str, Any]:
        if self._exploration:
            return {name: observation[name] for name in ("rgb", "depth")}
        return observation

    @property
    def episode_over(self) -> bool:
        self._require_open()
        return bool(self._env.episode_over)

    def episode_info(self) -> dict[str, str]:
        """Return identifiers, without privileged poses or goal world positions."""
        self._require_open()
        episode = self._env.current_episode
        return {"episode_id": str(episode.episode_id), "scene_id": str(episode.scene_id)}

    def metrics(self) -> dict[str, Any]:
        """Return ground-truth evaluation data, to be kept out of policy inputs."""
        self._require_open()
        result = deepcopy(self._env.get_metrics())
        # Count locally: this does not depend on optional task registrations.
        result["num_steps"] = self._steps
        return result

    def resolved_config(self) -> dict[str, Any]:
        """Return the complete Habitat configuration for reproducibility."""
        return deepcopy(self._resolved_config)

    def audit(self) -> dict[str, Any]:
        """Return privileged diagnostics for a separate, explicitly gated log."""
        self._require_open()
        from habitat.utils.geometry_utils import quaternion_to_list

        state = self._env.sim.get_agent_state()
        return {
            "position": [float(value) for value in state.position],
            "rotation_xyzw": [float(value) for value in quaternion_to_list(state.rotation)],
            "collisions": deepcopy(self._env.get_metrics().get("collisions")),
        }

    def close(self) -> None:
        """Release the renderer; repeated closes are harmless."""
        if not self._closed:
            self._env.close()
            self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("The Habitat environment has already been closed.")
