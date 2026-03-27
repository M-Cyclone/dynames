from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from sim.state import State


class Behavior(ABC):
    @abstractmethod
    def compute(self, state: State) -> np.ndarray:
        """Return a force contribution for each simulated agent."""
