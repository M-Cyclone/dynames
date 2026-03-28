from __future__ import annotations

from collections.abc import Callable, Iterable


class Pipeline:
    def __init__(self, passes: Iterable[Callable[[float], None]]) -> None:
        self.passes = list(passes)

    def run(self, dt: float) -> None:
        for pipeline_pass in self.passes:
            pipeline_pass(dt)
