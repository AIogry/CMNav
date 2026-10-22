"""Minimal policy for checking the environment, not a navigation baseline."""
from __future__ import annotations

import random
from typing import Any


class RandomPolicy:
    def __init__(self, seed: int, forward_probability: float, max_steps: int):
        self.rng = random.Random(seed)
        self.forward_probability = forward_probability
        self.max_steps = max_steps
        self.steps = 0

    def act(self, observation: dict[str, Any]) -> str:
        # Metrics and simulator state are deliberately not accepted by this API.
        self.steps += 1
        if self.steps >= self.max_steps:
            return "STOP"
        sample = self.rng.random()
        if sample < self.forward_probability:
            return "MOVE_FORWARD"
        if sample < (1 + self.forward_probability) / 2:
            return "TURN_LEFT"
        return "TURN_RIGHT"

