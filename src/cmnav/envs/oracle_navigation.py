"""Explicitly privileged official PointNav path following.

This wrapper is for reference collection and an oracle navigation baseline.
Its follower consumes simulator state and NavMesh geometry. It must never be
represented as an RGB-D-only policy, even though ``reset`` and ``step`` retain
the ordinary observation interface of :class:`HabitatEnvironment`.
"""

from __future__ import annotations

import math
from copy import deepcopy
from numbers import Real
from typing import Any

from cmnav.envs.habitat_env import HabitatEnvironment


class OracleFollowerError(RuntimeError):
    """The reference follower could not provide a valid next action.

    A caller should record the rollout as a follower failure. This exception
    must not be converted to a successful STOP or an invented transition.
    """


class OracleNavigationEnvironment:
    """Run actual Habitat actions selected by its privileged path follower.

    ``episodes`` uses the same ordered-episode interface as the base adapter.
    Each episode must have exactly one finite, three-dimensional point goal.
    The default follower radius is inside the base adapter's 0.2 m success
    threshold; task success is still determined by Habitat's measurements
    after STOP, never inferred from the follower returning STOP.

    ``current_episode`` and ``reference_target`` are privileged provenance
    interfaces, not policy inputs. World-state diagnostics additionally require
    explicit ``oracle_state_diagnostic=True``. The wrapper does not inject any
    of these values into observations.
    """

    ACTIONS = ("STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT")

    def __init__(
        self,
        config: dict[str, Any],
        *,
        episodes: list[Any] | None = None,
        oracle_state_diagnostic: bool = False,
        goal_radius: float = 0.15,
    ) -> None:
        if isinstance(goal_radius, bool) or not isinstance(goal_radius, (int, float)):
            raise ValueError("Reference goal_radius must be a finite number in (0, 0.2].")
        if not math.isfinite(goal_radius) or not 0.0 < goal_radius <= 0.2:
            raise ValueError("Reference goal_radius must be a finite number in (0, 0.2].")
        if not isinstance(oracle_state_diagnostic, bool):
            raise ValueError("oracle_state_diagnostic must be an explicit boolean.")
        try:
            from habitat.sims.habitat_simulator.actions import HabitatSimActions
            from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
            from habitat_sim.errors import GreedyFollowerError
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                f"Habitat runtime dependency {exc.name!r} is unavailable. "
                "Use the CMNav .env environment; see README.md."
            ) from exc

        self._backend = HabitatEnvironment(config, episodes=episodes)
        self._follower_class = ShortestPathFollower
        self._follower_error_class = GreedyFollowerError
        self._action_names = {
            HabitatSimActions.stop: "STOP",
            HabitatSimActions.move_forward: "MOVE_FORWARD",
            HabitatSimActions.turn_left: "TURN_LEFT",
            HabitatSimActions.turn_right: "TURN_RIGHT",
        }
        self._goal_radius = float(goal_radius)
        self._oracle_state_diagnostic = oracle_state_diagnostic
        self._follower: Any = None
        self._target: list[float] | None = None
        self._closed = False

    def reset(self) -> dict[str, Any]:
        """Start a real episode and clear the previous follower's history."""
        self._require_open()
        self._target = None
        self._follower = None
        observation = self._backend.reset()
        episode = self._backend._env.current_episode
        if len(episode.goals) != 1:
            raise ValueError("Reference PointNav requires exactly one point goal per episode.")
        target = self._position(episode.goals[0].position, "Reference point goal")
        # Habitat's follower retains an internal action history for its
        # anti-thrashing logic. Independent episodes must not share that state.
        self._follower = self._follower_class(
            self._backend._env.sim,
            goal_radius=self._goal_radius,
            return_one_hot=False,
            stop_on_error=False,
        )
        self._target = target
        return observation

    def get_reference_action(self, target_position: Any = None) -> str:
        """Replan from the actual state to the episode goal or an explicit waypoint.

        This operation does not advance or reposition the agent. STOP for an
        intermediate waypoint means that waypoint was reached: a caller must
        select its next waypoint instead of executing STOP, which terminates
        the entire Habitat task. Only Habitat metrics establish task success.
        """
        self._require_active()
        if self._backend.episode_over:
            raise RuntimeError("Cannot request a reference action after episode termination.")
        target = (
            self.reference_target
            if target_position is None
            else self._position(target_position, "Reference waypoint")
        )
        try:
            action = self._follower.get_next_action(target)
        except self._follower_error_class as exc:
            info = self._backend.episode_info()
            raise OracleFollowerError(
                "Official path follower failed for "
                f"scene {info['scene_id']!r}, episode {info['episode_id']!r}: {exc}"
            ) from exc
        try:
            return self._action_names[action]
        except (KeyError, TypeError) as exc:
            raise OracleFollowerError(
                f"Official path follower returned unsupported action {action!r}."
            ) from exc

    def step(self, action: str) -> dict[str, Any]:
        """Execute the requested action and return the actual successor observation."""
        self._require_active()
        if self._backend.episode_over:
            raise RuntimeError("Cannot step a terminated reference episode.")
        return self._backend.step(action)

    @property
    def episode_over(self) -> bool:
        self._require_active()
        return self._backend.episode_over

    @property
    def current_episode(self) -> Any:
        """Return a detached copy of privileged episode provenance."""
        self._require_active()
        return deepcopy(self._backend._env.current_episode)

    @property
    def reference_target(self) -> list[float]:
        """Return the single goal in world coordinates; this is privileged."""
        self._require_active()
        return list(self._target)

    def episode_info(self) -> dict[str, str]:
        """Return identifiers without poses or world-coordinate goals."""
        self._require_active()
        return self._backend.episode_info()

    def metrics(self) -> dict[str, Any]:
        """Return evaluation measurements, including actual STOP-based success."""
        self._require_active()
        return self._backend.metrics()

    def privileged_state(self) -> dict[str, Any]:
        """Return world pose and goal only for explicitly enabled diagnostics."""
        self._require_active()
        self._require_diagnostics()
        audit = self._backend.audit()
        return {
            "position": audit["position"],
            "rotation_xyzw": audit["rotation_xyzw"],
            "goal_position": self.reference_target,
        }

    def geodesic_distance(self, source: Any, target: Any) -> float:
        """Return privileged NavMesh distance without moving or rendering.

        Unreachable pairs return positive infinity, as in Habitat. Passing no
        episode cache ensures an intermediate target cannot reuse a path to
        the task's final goal. This oracle query requires the same explicit
        gate as world-state diagnostics.
        """
        self._require_active()
        self._require_diagnostics()
        source_position = self._position(source, "Geodesic source")
        target_position = self._position(target, "Geodesic target")
        distance = float(
            self._backend._env.sim.geodesic_distance(
                source_position, target_position, episode=None
            )
        )
        if math.isnan(distance) or distance < 0.0:
            raise RuntimeError(f"Habitat returned invalid geodesic distance {distance!r}.")
        return distance

    def resolved_config(self) -> dict[str, Any]:
        """Return the unchanged base adapter's resolved Habitat configuration."""
        return self._backend.resolved_config()

    def close(self) -> None:
        if not self._closed:
            self._backend.close()
            self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("The reference environment has already been closed.")

    def _require_active(self) -> None:
        self._require_open()
        if self._target is None or self._follower is None:
            raise RuntimeError("Call reset() before using a reference episode.")

    def _require_diagnostics(self) -> None:
        if not self._oracle_state_diagnostic:
            raise PermissionError(
                "World-state diagnostics require oracle_state_diagnostic=True."
            )

    @staticmethod
    def _position(value: Any, label: str) -> list[float]:
        try:
            coordinates = list(value)
        except TypeError as exc:
            raise ValueError(f"{label} must contain three finite real coordinates.") from exc
        if len(coordinates) != 3 or any(
            isinstance(coordinate, bool) or not isinstance(coordinate, Real)
            for coordinate in coordinates
        ):
            raise ValueError(f"{label} must contain three finite real coordinates.")
        result = [float(coordinate) for coordinate in coordinates]
        if not all(math.isfinite(coordinate) for coordinate in result):
            raise ValueError(f"{label} must contain three finite real coordinates.")
        return result
