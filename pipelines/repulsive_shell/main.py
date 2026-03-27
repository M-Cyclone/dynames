from __future__ import annotations

from pipelines.repulsive_shell.pipeline import build_pipeline


def main() -> None:
    simulator, state = build_pipeline()
    next_state = simulator.step(state)
    print("Repulsive shell pipeline ready.")
    print(f"next positions: {next_state.positions}")
