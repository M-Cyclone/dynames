from __future__ import annotations

import numpy as np

from sim.state import Action, State


def integrate(state: State, action: Action, forces: np.ndarray, dt: float) -> State:
    acceleration = forces
    next_velocities = action.desired_velocity + acceleration * dt
    next_positions = state.positions + next_velocities * dt
    return State(positions=next_positions, velocities=next_velocities, phase=state.phase)
