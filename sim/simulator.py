from __future__ import annotations

import numpy as np

from core.behaviors.base import Behavior
from core.controllers.base import Controller
from sim.integrator import integrate
from sim.state import State


class Simulator:
    def __init__(self, controller: Controller, behaviors: list[Behavior] | None = None, dt: float = 1 / 60) -> None:
        self.controller = controller
        self.behaviors = behaviors or []
        self.dt = dt

    def step(self, state: State) -> State:
        action = self.controller.act(state)
        forces = np.zeros_like(state.positions)

        for behavior in self.behaviors:
            forces += behavior.compute(state)

        return integrate(state, action, forces, self.dt)
