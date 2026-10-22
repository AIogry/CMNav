"""Goal-free RGB-D collection over an explicit, finite start manifest."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from cmnav.envs.habitat_env import HabitatEnvironment


class ExplorationEnvironment:
    """Keep start poses and simulator truth outside the policy interface.

    ``starts`` is already sampled and validated by the collection coordinator.
    Each entry contains ``source_episode_id``, ``scene_id`` (an absolute scene
    path), ``start_position``, ``start_rotation`` (xyzw), and optionally a
    ``trajectory_id``. Their given order is used once, without scene regrouping.

    The environment budget is a safety limit one action longer than the
    collector's budget. The collector records its own budget truncation and
    never issues an artificial STOP. ``audit`` is privileged and must only be
    used by the separate diagnostic writer.
    """

    ACTIONS = ("MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT")

    def __init__(self, config: dict[str, Any], starts: list[dict[str, Any]]) -> None:
        if not starts:
            raise ValueError("Exploration requires at least one selected start.")
        try:
            from habitat.tasks.nav.nav import NavigationEpisode
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                f"Habitat runtime dependency {exc.name!r} is unavailable. "
                "Use the CMNav .env environment; see README.md."
            ) from exc

        self._starts = deepcopy(starts)
        episodes = []
        for index, start in enumerate(self._starts):
            trajectory_id = str(start.get("trajectory_id", f"trajectory_{index:06d}"))
            start["trajectory_id"] = trajectory_id
            episodes.append(
                NavigationEpisode(
                    episode_id=trajectory_id,
                    scene_id=str(start["scene_id"]),
                    start_position=list(start["start_position"]),
                    start_rotation=list(start["start_rotation"]),
                    # Nav-v0 requires the field, but goal-free sensors and
                    # collision-only measurements do not consume any goal.
                    goals=[],
                )
            )
        self._backend = HabitatEnvironment(config, episodes=episodes, exploration=True)
        self._index = -1
        self._closed = False

    def reset(self) -> dict[str, Any]:
        """Begin the next manifest start; exhaustion never silently cycles."""
        self._require_open()
        next_index = self._index + 1
        if next_index >= len(self._starts):
            raise StopIteration("All exploration starts have been consumed.")
        observation = self._backend.reset()
        self._index = next_index
        return observation

    def step(self, action: str) -> dict[str, Any]:
        self._require_active()
        if action not in self.ACTIONS:
            raise ValueError(f"Unknown exploration action {action!r}; expected {self.ACTIONS}.")
        return self._backend.step(action)

    @property
    def episode_over(self) -> bool:
        self._require_active()
        return self._backend.episode_over

    def episode_info(self) -> dict[str, str]:
        """Return identity/provenance only, without start poses or goals."""
        self._require_active()
        start = self._starts[self._index]
        return {
            **self._backend.episode_info(),
            "trajectory_id": str(start["trajectory_id"]),
            "source_episode_id": str(start["source_episode_id"]),
        }

    def audit(self) -> dict[str, Any]:
        self._require_active()
        return self._backend.audit()

    def resolved_config(self) -> dict[str, Any]:
        return self._backend.resolved_config()

    def close(self) -> None:
        if not self._closed:
            self._backend.close()
            self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("The exploration environment has already been closed.")

    def _require_active(self) -> None:
        self._require_open()
        if self._index < 0:
            raise RuntimeError("Call reset() before using an exploration episode.")
