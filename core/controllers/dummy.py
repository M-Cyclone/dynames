from __future__ import annotations

import numpy as np

from core.controllers.base import Controller
from sim.state import Action, State


class DummyController(Controller):
    def act(self, state: State) -> Action:
        return Action(desired_velocity=np.zeros_like(state.velocities))
