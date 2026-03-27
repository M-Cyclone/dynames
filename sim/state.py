from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class State:
    positions: np.ndarray
    velocities: np.ndarray
    phase: np.ndarray | None = None


@dataclass
class Action:
    desired_velocity: np.ndarray
