from __future__ import annotations

from abc import ABC, abstractmethod

from sim.state import Action, State


class Controller(ABC):
    @abstractmethod
    def act(self, state: State) -> Action:
        """Produce the desired action for the current state."""
