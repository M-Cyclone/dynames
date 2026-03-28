from __future__ import annotations

from abc import ABC, abstractmethod


class System(ABC):
    @abstractmethod
    def step(self, dt: float) -> None:
        raise NotImplementedError
