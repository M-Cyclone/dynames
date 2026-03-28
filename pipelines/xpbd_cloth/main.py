from __future__ import annotations

from pipelines.xpbd_cloth.pipeline import build_pipeline
from viz.viewer import launch_viewer as launch_simulation_viewer


def launch_viewer() -> None:
    launch_simulation_viewer(
        scene_factory=build_pipeline,
        title="Dynames XPBD Viewer",
        scene_name="XPBD Cloth",
    )


def main() -> None:
    scene, particles = build_pipeline()
    dt = 1.0 / 60.0

    for frame in range(120):
        scene.step(dt)
        if frame % 30 == 0:
            positions = particles.positions_numpy()
            center_index = positions.shape[0] // 2
            center = positions[center_index]
            print(
                f"frame={frame:03d} center=({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})"
            )

    positions = particles.positions_numpy()
    print("XPBD cloth pipeline ready.")
    print(f"pinned corners: {positions[0]}, {positions[11]}")
    print(f"last particle: {positions[-1]}")
