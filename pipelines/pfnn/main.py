from __future__ import annotations

from pipelines.pfnn.pipeline import build_pipeline


def main() -> None:
    simulator, state = build_pipeline()
    next_state = simulator.step(state)
    print("PFNN pipeline ready.")
    print(f"next positions: {next_state.positions}")
