from __future__ import annotations

import numpy as np

from sim.state import Action, State


class PFNNModel:
    """Minimal PFNN placeholder for wiring the framework."""

    def predict(self, state: State) -> Action:
        desired_velocity = np.copy(state.velocities)
        if state.phase is not None:
            desired_velocity = desired_velocity + 0.1 * np.sin(state.phase)[:, None]
        return Action(desired_velocity=desired_velocity)
