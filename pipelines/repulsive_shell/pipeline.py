from __future__ import annotations

import numpy as np

from core.behaviors import RepulsiveShell
from core.controllers import DummyController
from sim import Simulator, State


def build_pipeline() -> tuple[Simulator, State]:
    controller = DummyController()
    behaviors = [RepulsiveShell(radius=1.0, strength=2.0)]
    simulator = Simulator(controller=controller, behaviors=behaviors)

    initial_state = State(
        positions=np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=float),
        velocities=np.zeros((2, 3), dtype=float),
    )
    return simulator, initial_state
