"""A reproducible depth-reactive exploration heuristic; no simulator queries."""
from __future__ import annotations

import random
from typing import Any

import numpy as np


class DepthWalkPolicy:
    """Alternate persistent forward/turn segments; turn when depth looks blocked.

    The central 30% x 30% image region supplies the configured depth quantile.
    Missing/nonpositive depth is unknown, not evidence of free space. This is
    a collection heuristic, not a coverage guarantee or a navigation baseline.
    """

    version = "depth_walk_v1"

    def __init__(self, seed: int, config: dict[str, Any]):
        self.rng = random.Random(seed)
        self.config = dict(config)
        self.forward_remaining = self._length("forward")
        self.turn_remaining = 0
        self.turn_action = "TURN_LEFT"

    def _length(self, kind: str) -> int:
        return self.rng.randint(self.config[f"min_{kind}_steps"], self.config[f"max_{kind}_steps"])

    def act(self, observation: dict[str, Any]) -> str:
        if set(observation) != {"rgb", "depth"}:
            raise ValueError("Exploration policy accepts exactly rgb and depth")
        if self.turn_remaining:
            return self._turn()
        depth = np.asarray(observation["depth"])
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        if depth.ndim != 2 or min(depth.shape) < 1:
            raise ValueError("Expected an HxW metric depth image")
        height, width = depth.shape
        top, left = int(height * 0.35), int(width * 0.35)
        roi = depth[top:max(top + 1, int(height * 0.65)),
                    left:max(left + 1, int(width * 0.65))]
        valid = np.isfinite(roi) & (roi > 0)
        clear = (float(valid.mean()) >= self.config["min_valid_fraction"]
                 and float(np.quantile(roi[valid], self.config["depth_quantile"]))
                 > self.config["obstacle_distance"])
        if clear and self.forward_remaining:
            self.forward_remaining -= 1
            return "MOVE_FORWARD"
        self.turn_action = self.rng.choice(("TURN_LEFT", "TURN_RIGHT"))
        self.turn_remaining = self._length("turn")
        return self._turn()

    def _turn(self) -> str:
        self.turn_remaining -= 1
        if self.turn_remaining == 0:
            self.forward_remaining = self._length("forward")
        return self.turn_action
