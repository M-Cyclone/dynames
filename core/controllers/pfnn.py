from __future__ import annotations

from core.controllers.base import Controller
from sim.state import Action, State


class PFNNController(Controller):
    def __init__(self, model) -> None:
        self.model = model

    def act(self, state: State) -> Action:
        return self.model.predict(state)
