from __future__ import annotations

from collections.abc import Iterable

from latticeon.core.system import System


class Scene:
    def __init__(self, systems: Iterable[System] | None = None) -> None:
        self.systems: list[System] = list(systems or [])

    def add_system(self, system: System) -> None:
        self.systems.append(system)

    def step(self, dt: float) -> None:
        for system in self.systems:
            system.step(dt)
