from __future__ import annotations

import numpy as np

from core.controllers import PFNNController
from core.models import PFNNModel
from sim import Simulator, State


def build_pipeline() -> tuple[Simulator, State]:
    model = PFNNModel()
    controller = PFNNController(model)
    simulator = Simulator(controller=controller, behaviors=[])

    initial_state = State(
        positions=np.zeros((1, 3), dtype=float),
        velocities=np.zeros((1, 3), dtype=float),
        phase=np.array([0.0], dtype=float),
    )
    return simulator, initial_state
