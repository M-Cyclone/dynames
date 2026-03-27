from __future__ import annotations

import numpy as np

from core.behaviors.base import Behavior
from sim.state import State


class RepulsiveShell(Behavior):
    def __init__(self, radius: float, strength: float = 1.0) -> None:
        self.radius = radius
        self.strength = strength

    def compute(self, state: State) -> np.ndarray:
        positions = state.positions
        forces = np.zeros_like(positions)

        for i, point in enumerate(positions):
            delta = point - positions
            distances = np.linalg.norm(delta, axis=1, keepdims=True)
            mask = (distances > 0.0) & (distances < self.radius)
            safe_distances = np.where(mask, distances, 1.0)
            direction = np.divide(delta, safe_distances, where=safe_distances != 0.0)
            magnitude = np.where(mask, (self.radius - safe_distances) / self.radius, 0.0)
            forces[i] = (direction * magnitude * self.strength).sum(axis=0)

        return forces
